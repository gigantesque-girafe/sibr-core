/*
 * GaussianLiveViewV42 implementation  (v4.2, double-buffered pipeline)
 *
 * Transport upgrade over v4.0:
 *   v4.0: Python → TCP → CPU staging → cudaMemcpy → GPU (every frame, ~2 MB)
 *   v4.2: Python → CUDA IPC shared buffer (zero-copy, no host involvement)
 *
 * Per-frame sequence (after handshake), pipelined across two buffers:
 *   1. Python (free-running, no ping): deform → extract features → write into
 *      buffer[i%2] → cudaEventRecord(dataReadyEvt) → sendall(frame_id, buf_idx, N_live).
 *   2. C++'s network thread receives that announcement and atomically publishes it —
 *      this never blocks the render thread.
 *   3. C++'s render thread (beginFrame(), once per frame) snapshots the latest
 *      announcement. onRenderIBR() (once per eye) cudaEventSynchronize()s that
 *      buffer's dataReadyEvt (near-instant — already recorded) and runs Color MLP
 *      + CudaRasterizer directly on its IPC pointers.
 *   4. After both eyes are done, C++ records that buffer's readCompleteEvt.
 *      Python waits on it before reusing that buffer slot for a future frame.
 *
 * Because Python no longer waits for a reply, it can deform frame N+1 while C++
 * is still rendering frame N — the two are no longer strictly alternating turns.
 */

#include "GaussianLiveViewV42.hpp"

#include <core/graphics/GUI.hpp>
#include <rasterizer.h>

#include <boost/asio.hpp>
#include <GL/glew.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>

using boost::asio::ip::tcp;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error helper that reports:
// - the CUDA error message,
// - the CUDA call that failed,
// - the line number.
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess)                                                    \
            SIBR_ERR << "CUDA error: " << cudaGetErrorString(_e)                 \
                     << "  (" #call ")  line " << __LINE__ << std::endl;          \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Resizable CUDA buffer helper (rasterizer scratch)
// if capacite needed is bigger than capacite current, free the current then alloc new memory
// templae pour que ca fonctionne avec tous type de valeurs (int, ptr etc)


template<typename T>
static std::function<char*(size_t)> resizeFunctional(T** ptr, size_t& cap)
{
    return [ptr, &cap](size_t needed) -> char* {
        if (needed > cap) {
            if (*ptr) cudaFree(*ptr);
            cudaMalloc(ptr, needed * 2);
            cap = needed * 2;
        }
        return reinterpret_cast<char*>(*ptr);
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Copy shader (float SSBO → render target)

static const char* COPY_VERT_SRC = R"GLSL(
#version 450
layout(location = 0) in vec4 in_vertex;
layout(location = 1) in vec4 in_texcoord;
layout(location = 2) in vec4 in_color;
out vec4 texcoord;
out vec4 color;
void main(void) {
    gl_Position = in_vertex;
    texcoord    = in_texcoord;
    color       = in_color;
}
)GLSL";

static const char* COPY_FRAG_SRC = R"GLSL(
#version 450
layout(location = 0) out vec4 out_color;
layout(std430, binding = 0) buffer colorLayout { float data[]; } source;
uniform bool flip   = false;
uniform int  width  = 1000;
uniform int  height = 800;
in vec4 texcoord;
void main(void) {
    int x = int(texcoord.x * width);
    int y = flip ? (height - 1 - int(texcoord.y * height)) : int(texcoord.y * height);
    float r = source.data[0 * width * height + (y * width + x)];
    float g = source.data[1 * width * height + (y * width + x)];
    float b = source.data[2 * width * height + (y * width + x)];
    out_color = vec4(r, g, b, 1.0);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
// TCP helpers
// ─────────────────────────────────────────────────────────────────────────────

static void tcp_recv_all(tcp::socket& sock, void* buf, size_t n)
{
    boost::asio::read(sock, boost::asio::buffer(buf, n));
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

sibr::GaussianLiveViewV42::GaussianLiveViewV42(
    const std::string& ip, int port,
    uint render_w, uint render_h,
    bool white_bg, int device)
    : sibr::ViewBase(render_w, render_h)
    , _ip(ip), _port(port)
    , _white_bg(white_bg), _device(device)
{
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (device >= num_devices)
        SIBR_ERR << "[V42] Invalid CUDA device " << device << std::endl;
    CUDA_CHECK(cudaSetDevice(device));

    CUDA_CHECK(cudaMalloc(&_view_cuda,   16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_proj_cuda,   16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_camPos_cuda,  3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_bg_cuda,      3 * sizeof(float)));

    float bg[3] = { white_bg ? 1.f : 0.f,
                    white_bg ? 1.f : 0.f,
                    white_bg ? 1.f : 0.f };
    CUDA_CHECK(cudaMemcpy(_bg_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    _geomBuf = resizeFunctional(&_geomPtr, _allocGeom);
    _binBuf  = resizeFunctional(&_binPtr,  _allocBin);
    _imgBuf  = resizeFunctional(&_imgPtr,  _allocImg);

    recreateImageBuffer(render_w, render_h);
    initShader();

    SIBR_LOG << "[V42] Connecting to Python at " << ip << ":" << port << std::endl;
    connectTCP();
}

sibr::GaussianLiveViewV42::~GaussianLiveViewV42()
{
    // Stop the network thread first — it must not touch `this` while we tear
    // down CUDA/IPC state below.
    stopNetworkThread();

    // IPC cleanup — must happen before destroying CUDA context
    for (auto& b : _buffers)
    {
        if (b.dataReadyEvt)    { cudaEventDestroy(b.dataReadyEvt);    b.dataReadyEvt    = nullptr; }
        if (b.readCompleteEvt) { cudaEventDestroy(b.readCompleteEvt); b.readCompleteEvt = nullptr; }
        if (b.base)            { cudaIpcCloseMemHandle(b.base);       b.base            = nullptr; }
    }

    if (_cudaBuffer)   cudaGraphicsUnregisterResource(_cudaBuffer);
    if (_glBuffer)     glDeleteBuffers(1, &_glBuffer);
    if (_fallbackCuda) cudaFree(_fallbackCuda);

    if (_view_cuda)   cudaFree(_view_cuda);
    if (_proj_cuda)   cudaFree(_proj_cuda);
    if (_camPos_cuda) cudaFree(_camPos_cuda);
    if (_bg_cuda)     cudaFree(_bg_cuda);

    if (_geomPtr) cudaFree(_geomPtr);
    if (_binPtr)  cudaFree(_binPtr);
    if (_imgPtr)  cudaFree(_imgPtr);
}

// Private helpers
void sibr::GaussianLiveViewV42::initShader()
{
    _copyShader.init("GaussianLiveV42Copy", COPY_VERT_SRC, COPY_FRAG_SRC);
    _flipU.init(_copyShader,   "flip");
    _widthU.init(_copyShader,  "width");
    _heightU.init(_copyShader, "height");
}

void sibr::GaussianLiveViewV42::recreateImageBuffer(uint w, uint h)
{
    if (_cudaBuffer)   { cudaGraphicsUnregisterResource(_cudaBuffer); _cudaBuffer = nullptr; }
    if (_glBuffer)     { glDeleteBuffers(1, &_glBuffer); _glBuffer = 0; }
    if (_fallbackCuda) { cudaFree(_fallbackCuda); _fallbackCuda = nullptr; }

    glCreateBuffers(1, &_glBuffer);
    glNamedBufferStorage(_glBuffer,
        (GLsizeiptr)(w * h * 3 * sizeof(float)),
        nullptr, GL_DYNAMIC_STORAGE_BIT);

    if (_useInterop) {
        cudaGraphicsGLRegisterBuffer(&_cudaBuffer, _glBuffer,
                                     cudaGraphicsRegisterFlagsWriteDiscard);
        _useInterop = (cudaGetLastError() == cudaSuccess);
    }
    if (!_useInterop) {
        _interopFailed = true;
        _fallbackBytes.resize(w * h * 3 * sizeof(float));
        cudaMalloc(&_fallbackCuda, _fallbackBytes.size());
        SIBR_LOG << "[V42] CUDA-GL interop unavailable, using fallback." << std::endl;
    }
}

bool sibr::GaussianLiveViewV42::connectTCP()
{
    try {
        _socket = std::make_unique<tcp::socket>(_io);
        tcp::endpoint ep(boost::asio::ip::address::from_string(_ip),
                         (unsigned short)_port);
        boost::system::error_code ec;
        do {
            _socket->connect(ep, ec);
            if (ec) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } while (ec);
        _connected = true;
        SIBR_LOG << "[V42] TCP connected to Python." << std::endl;
        return true;
    } catch (const std::exception& e) {
        SIBR_LOG << "[V42] TCP connect error: " << e.what() << std::endl;
        _connected = false;
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Startup handshake
// return 
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV42::handshakeWithPython()
{
    try {
        // 1. Send 4-byte magic (V42D = double-buffer pipelined protocol)
        const char magic[4] = {'V','4','2','D'};
        boost::asio::write(*_socket, boost::asio::buffer(magic, 4));

        // 2. Receive N_max, K, model_size
        uint32_t N_max = 0, K = 0, model_size = 0;
        tcp_recv_all(*_socket, &N_max,      sizeof(uint32_t));
        tcp_recv_all(*_socket, &K,          sizeof(uint32_t));
        tcp_recv_all(*_socket, &model_size, sizeof(uint32_t));
        _N_max = (int)N_max;
        _K     = (int)K;
        SIBR_LOG << "[V42] N_max=" << _N_max << "  K=" << _K
                 << "  model=" << model_size << " bytes" << std::endl;

        // 3. Receive and load TorchScript Color MLP
        std::vector<char> model_buf(model_size);
        tcp_recv_all(*_socket, model_buf.data(), model_size);
        std::istringstream ss(std::string(model_buf.data(), model_size));
        _colorMLP = torch::jit::load(ss, torch::Device(torch::kCUDA, _device));
        _colorMLP.eval();
        _mlpLoaded = true;
        SIBR_LOG << "[V42] Color MLP loaded." << std::endl;

        // 4. Receive source device index (int32)
        int32_t src_device = 0;
        tcp_recv_all(*_socket, &src_device, sizeof(int32_t));

        CUDA_CHECK(cudaSetDevice(_device));

        // 5. Receive IPC handles for each of the 2 buffers and open them
        for (int i = 0; i < 2; ++i)
        {
            uint64_t ipc_offset = 0;
            tcp_recv_all(*_socket, &ipc_offset, sizeof(uint64_t));

            cudaIpcMemHandle_t mem_handle;
            tcp_recv_all(*_socket, &mem_handle, sizeof(cudaIpcMemHandle_t));

            cudaIpcEventHandle_t data_ready_handle;
            tcp_recv_all(*_socket, &data_ready_handle, sizeof(cudaIpcEventHandle_t));

            cudaIpcEventHandle_t read_complete_handle;
            tcp_recv_all(*_socket, &read_complete_handle, sizeof(cudaIpcEventHandle_t));

            IpcBuffer& b = _buffers[i];
            CUDA_CHECK(cudaIpcOpenMemHandle(&b.base, mem_handle, cudaIpcMemLazyEnablePeerAccess));
            b.ptr = reinterpret_cast<float*>(static_cast<char*>(b.base) + ipc_offset);
            CUDA_CHECK(cudaIpcOpenEventHandle(&b.dataReadyEvt, data_ready_handle));
            CUDA_CHECK(cudaIpcOpenEventHandle(&b.readCompleteEvt, read_complete_handle));

            // Pre-compute field pointers into this buffer.
            // Layout: xyz[N_max*3] | feat[N_max*K] | R_bwd[N_max*9] | opa[N_max] | scale[N_max*3] | rot[N_max*4]
            ptrdiff_t off = 0;
            b.xyzPtr   = b.ptr + off; off += (ptrdiff_t)_N_max * 3;
            b.featPtr  = b.ptr + off; off += (ptrdiff_t)_N_max * _K;
            b.rbwdPtr  = b.ptr + off; off += (ptrdiff_t)_N_max * 9;
            b.opaPtr   = b.ptr + off; off += (ptrdiff_t)_N_max * 1;
            b.scalePtr = b.ptr + off; off += (ptrdiff_t)_N_max * 3;
            b.rotPtr   = b.ptr + off;

            SIBR_LOG << "[V42] Buffer " << i << " ready. ipc_offset=" << ipc_offset
                     << "  src_device=" << src_device << std::endl;
        }

        SIBR_LOG << "[V42] Both IPC buffers ready. Total floats per buffer: "
                 << (ptrdiff_t)_N_max * (_K + 20) << std::endl;

        _handshakeDone = true;
        return true;

    } catch (const std::exception& e) {
        SIBR_ERR << "[V42] Handshake failed: " << e.what() << std::endl;
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// setResolution
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveViewV42::setResolution(const sibr::Vector2i& size)
{
    if (size == getResolution()) return;
    ViewBase::setResolution(size);
    recreateImageBuffer((uint)size.x(), (uint)size.y());
    _widthU.get()  = size.x();
    _heightU.get() = size.y();
}

// ─────────────────────────────────────────────────────────────────────────────
// Network thread — receives Python's per-frame announcements (frame_id,
// buf_idx, N_live) and atomically publishes them. Runs independently of the
// render thread; never touches CUDA or GL.
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveViewV42::networkThreadFunc()
{
    while (_networkRunning.load(std::memory_order_relaxed))
    {
        uint64_t frame_id  = 0;
        uint32_t buf_idx_u = 0;
        uint32_t n_live_u  = 0;
        try {
            tcp_recv_all(*_socket, &frame_id,  sizeof(uint64_t));
            tcp_recv_all(*_socket, &buf_idx_u, sizeof(uint32_t));
            tcp_recv_all(*_socket, &n_live_u,  sizeof(uint32_t));
        } catch (const std::exception&) {
            // Python disconnected. Don't touch non-atomic members from this
            // thread — beginFrame() on the render thread detects this via
            // _networkRunning and handles teardown/reconnect.
            _networkRunning.store(false, std::memory_order_relaxed);
            return;
        }
        // Write order matters: n_live/frame_id first, buf_idx last (release).
        // beginFrame() reads buf_idx first (acquire), so if it observes a new
        // buf_idx it's guaranteed to also observe the matching n_live/frame_id —
        // no torn (buf_idx, n_live) combination.
        _latestNLive.store((int)n_live_u, std::memory_order_relaxed);
        _latestFrameId.store(frame_id, std::memory_order_relaxed);
        _latestReadyBuf.store((int)buf_idx_u, std::memory_order_release);
    }
}

void sibr::GaussianLiveViewV42::startNetworkThread()
{
    // read the latest buffer available: -1 is nothing has rendered yet, 0 and 1 are our double buffers
    _latestReadyBuf.store(-1, std::memory_order_relaxed);

    // Allow the network thread's receive loop to run
    _networkRunning.store(true, std::memory_order_relaxed);

    // Launch the background network thread, which executes
    // GaussianLiveViewV42::networkThreadFunc() on this object.
    _networkThread = std::thread(&GaussianLiveViewV42::networkThreadFunc, this);
}

void sibr::GaussianLiveViewV42::stopNetworkThread()
{
    // Unallow the network thread's receive loop to run
    _networkRunning.store(false, std::memory_order_relaxed);

    // The network thread is blocked in a synchronous read; closing the socket
    // is what actually unblocks it (the atomic flag alone won't interrupt it).
    if (_socket) {
        boost::system::error_code ec;
        _socket->close(ec);
    }
    if (_networkThread.joinable()) _networkThread.join();
}

// ─────────────────────────────────────────────────────────────────────────────
// beginFrame — call once per frame. Check the connection and select which GPU buffer
// to render 
// return true if ok; false if connection failed
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV42::beginFrame()
{
    // Detect a network-thread death (Python disconnected) and tear down.
    if (_connected && _handshakeDone && !_networkRunning.load(std::memory_order_relaxed))
    {
        SIBR_LOG << "[V42] Lost connection to Python — reconnecting." << std::endl;
        if (_networkThread.joinable()) _networkThread.join();
        _socket.reset();
        _connected     = false;
        _handshakeDone = false;
        _mlpLoaded     = false;
        _hasData       = false;
        _latestReadyBuf.store(-1, std::memory_order_relaxed);
    }

    // if not connected, return false
    if (!_connected) {
        if (!connectTCP()) { _hasData = false; return false; }
    }

    // if handshake not done, then do the handshake. If handshake fail, return false, if succeed, create network thread
    if (!_handshakeDone) {
        if (!handshakeWithPython()) {
            _connected = false;
            _hasData   = false;
            return false;
        }
        startNetworkThread();
    }

    auto t0 = std::chrono::steady_clock::now();

    int buf = _latestReadyBuf.load(std::memory_order_acquire);
    if (buf < 0) {
        // Connected, handshake done, but no frame announced yet — normal for
        // the first few milliseconds after connecting.
        _hasData = false;
        return true;
    }

    _currentBufIdx  = buf;
    _currentNLive   = _latestNLive.load(std::memory_order_relaxed);
    _currentFrameId = _latestFrameId.load(std::memory_order_relaxed);
    _hasData        = true;

    auto t1 = std::chrono::steady_clock::now();
    _beginFrameMsThisFrame = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// onRenderIBR — called per eye by OpenXRRdrMode
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveViewV42::onRenderIBR(sibr::IRenderTarget& dst,
                                              const sibr::Camera& eye)
{
    if (!_mlpLoaded || !_hasData || _currentBufIdx < 0 || _currentNLive == 0) return;

    IpcBuffer& buf = _buffers[_currentBufIdx];

    uint w = (uint)_resolution.x();
    uint h = (uint)_resolution.y();

    // ── View / projection matrices ────────────────────────────────────────────
    auto view_mat = eye.view();
    auto proj_mat = eye.viewproj();
    view_mat.row(1) *= -1.f;
    view_mat.row(2) *= -1.f;
    proj_mat.row(1) *= -1.f;

    CUDA_CHECK(cudaMemcpy(_view_cuda,   view_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_proj_cuda,   proj_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_camPos_cuda, &eye.position(),  3 * sizeof(float), cudaMemcpyHostToDevice));

    float tan_fovy = std::tan(eye.fovy() * 0.5f);
    float tan_fovx = tan_fovy * eye.aspect();

    // ── Wait for Python to finish writing this buffer ─────────────────────────
    // TCP ordering guarantees the event was recorded before Python's announcement
    // was sent, and we received the announcement before reaching here. This call
    // returns immediately.
    CUDA_CHECK(cudaEventSynchronize(buf.dataReadyEvt));

    // ── COLOR MLP===============================================================
    auto t_mlp0 = std::chrono::steady_clock::now();
    {
        auto dev  = torch::Device(torch::kCUDA, _device);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        // from_blob: LibTorch wraps the IPC pointer without owning it.
        auto feat_t = torch::from_blob(buf.featPtr,  {_currentNLive, _K},    opts);
        auto xyz_t  = torch::from_blob(buf.xyzPtr,   {_currentNLive, 3},     opts);
        auto rbwd_t = torch::from_blob(buf.rbwdPtr,  {_currentNLive, 3, 3},  opts);

        sibr::Vector3f pos = eye.position();
        auto cam_t = torch::tensor({pos.x(), pos.y(), pos.z()}, opts);

        torch::NoGradGuard no_grad;
        _colors_t = _colorMLP.forward({feat_t, xyz_t, cam_t, rbwd_t})
                              .toTensor()
                              .contiguous();   // [N_live, 3] on CUDA
    }
    // LibTorch CUDA ops are async — sync before reading elapsed time, same
    // correctness requirement as the Python-side deform timing.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_mlp1 = std::chrono::steady_clock::now();
    _mlpMsAccum += std::chrono::duration<double, std::milli>(t_mlp1 - t_mlp0).count();

    float* colors_ptr = _colors_t.data_ptr<float>();

    // ── Map GL buffer for CUDA write ──────────────────────────────────────────
    float* image_cuda = nullptr;
    if (!_interopFailed) {
        size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &_cudaBuffer));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            (void**)&image_cuda, &bytes, _cudaBuffer));
    } else {
        image_cuda = _fallbackCuda;
    }

    // ── Rasterize (reads xyz, opa, scale, rot directly from IPC buffer) ───────
    auto t_raster0 = std::chrono::steady_clock::now();
    CudaRasterizer::Rasterizer::forward(
        _geomBuf, _binBuf, _imgBuf,
        _currentNLive, /*D=*/0, /*M=*/0,
        _bg_cuda,
        (int)w, (int)h,
        buf.xyzPtr,            // positions (IPC, zero-copy)
        /*shs=*/nullptr,
        colors_ptr,            // per-eye precomputed RGB from Color MLP
        buf.opaPtr,            // opacity (IPC, zero-copy)
        buf.scalePtr,          // scale (IPC, zero-copy)
        /*scale_modifier=*/1.0f,
        buf.rotPtr,            // rotation quaternion (IPC, zero-copy)
        /*cov3D_precomp=*/nullptr,
        _view_cuda, _proj_cuda, _camPos_cuda,
        tan_fovx, tan_fovy,
        /*prefiltered=*/false,
        image_cuda
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_raster1 = std::chrono::steady_clock::now();
    _rasterMsAccum += std::chrono::duration<double, std::milli>(t_raster1 - t_raster0).count();

    // ── Unmap and blit ────────────────────────────────────────────────────────
    if (!_interopFailed) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cudaBuffer));
    } else {
        cudaMemcpy(_fallbackBytes.data(), _fallbackCuda,
                   _fallbackBytes.size(), cudaMemcpyDeviceToHost);
        glNamedBufferSubData(_glBuffer, 0,
                             (GLsizeiptr)_fallbackBytes.size(),
                             _fallbackBytes.data());
    }

    // ── Copy shader: GL SSBO → IRenderTarget ─────────────────────────────────
    glDisable(GL_DEPTH_TEST);
    _copyShader.begin();
    _flipU.send();
    _widthU.get()  = (int)w;
    _heightU.get() = (int)h;
    _widthU.send();
    _heightU.send();

    dst.clear();
    dst.bind();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _glBuffer);
    sibr::RenderUtility::renderScreenQuad();
    dst.unbind();
    _copyShader.end();

    CHECK_GL_ERROR;

    // onRenderIBR is called once per eye (left, then right) per frame. Once
    // both eyes are done, this buffer slot is safe to overwrite — record the
    // read-complete event so Python's wait_read_complete() can unblock and
    // reuse this slot for a future frame. Log the per-frame total — mlp/raster
    // summed across both eyes — once every other call, matching Python's
    // _LOG_INTERVAL=60 cadence.
    _eyeCallsThisFrame++;
    if (_eyeCallsThisFrame >= 2)
    {
        _eyeCallsThisFrame = 0;
        CUDA_CHECK(cudaEventRecord(buf.readCompleteEvt, 0));

        _frameCount++;
        if (_frameCount % kLogIntervalFrames == 0 || _frameCount <= 3)
        {
            SIBR_LOG << "[V42] frame " << _frameCount
                     << "  pyFrameId=" << _currentFrameId
                     << "  buf=" << _currentBufIdx
                     << "  beginFrame=" << _beginFrameMsThisFrame << "ms"
                     << "  mlp(L+R)=" << _mlpMsAccum << "ms"
                     << "  raster(L+R)=" << _rasterMsAccum << "ms"
                     << "  render_total=" << (_mlpMsAccum + _rasterMsAccum) << "ms"
                     << std::endl;
        }
        _mlpMsAccum    = 0.0;
        _rasterMsAccum = 0.0;
    }
}

void sibr::GaussianLiveViewV42::onGUI()
{
    if (ImGui::Begin("GaussianLiveViewV42")) {
        ImGui::Text("v4.2 — zero-copy CUDA IPC transport (double-buffered, pipelined)");
        ImGui::Text("Gaussians (live): %d / %d", _currentNLive, _N_max);
        ImGui::Text("Feat dim K      : %d",  _K);
        ImGui::Text("Connected       : %s",  _connected     ? "yes" : "no");
        ImGui::Text("MLP loaded      : %s",  _mlpLoaded     ? "yes" : "no");
        ImGui::Text("IPC ready       : %s",  (_buffers[0].ptr != nullptr && _buffers[1].ptr != nullptr) ? "yes" : "no");
        ImGui::Text("Current buffer  : %d",  _currentBufIdx);
        ImGui::Text("Python frame id : %llu", (unsigned long long)_currentFrameId);
    }
    ImGui::End();
}
