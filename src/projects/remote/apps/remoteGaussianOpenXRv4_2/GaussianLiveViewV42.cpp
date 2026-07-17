/*
 * NEW FILE — does not modify any existing SIBR source.
 *
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
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>

using boost::asio::ip::tcp;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error helper
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
// ─────────────────────────────────────────────────────────────────────────────

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
// ─────────────────────────────────────────────────────────────────────────────

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

static void tcp_send_all(tcp::socket& sock, const void* buf, size_t n)
{
    boost::asio::write(sock, boost::asio::buffer(buf, n));
}

// Identity index -> ZJU-MoCap subject id, for display in the GUI. Must match the
// dataset order Python decodes from (appearance_identity in render_vr_v1_modular).
static const int   kNumIdentities   = 8;
static const char* kSubjectNames[kNumIdentities] =
    { "386", "387", "377", "392", "315", "394", "393", "390" };

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

sibr::GaussianLiveViewV42::GaussianLiveViewV42(
    const std::string& ip, int port,
    uint render_w, uint render_h,
    bool white_bg, int device, int views_per_frame)
    : sibr::ViewBase(render_w, render_h)
    , _ip(ip), _port(port)
    , _white_bg(white_bg), _device(device)
    , _viewsPerFrame(views_per_frame < 1 ? 1 : views_per_frame)
    , _nativeRes((int)render_w, (int)render_h)
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

    // CUDA-GL interop is ON by default (zero-copy: rasterizer writes straight into
    // the mapped GL buffer, no D2H+H2D round trip). Set env V42_INTEROP=0 to force
    // the host-copy fallback. recreateImageBuffer() logs the real register error
    // and whether the GL context is on the same CUDA device as the rasterizer, so
    // a failure can still be diagnosed.
    {
        const char* io = std::getenv("V42_INTEROP");
        _useInterop = (io == nullptr) || (io[0] != '0');
    }

    recreateImageBuffer(render_w, render_h);
    initShader();

    // Optional per-stage GPU timing (env V42_TIMING=1). Off by default so the
    // render hot path issues no cudaDeviceSynchronize at all.
    {
        const char* t = std::getenv("V42_TIMING");
        _timingEnabled = (t != nullptr && t[0] != '\0' && t[0] != '0');
    }
    if (_timingEnabled) {
        CUDA_CHECK(cudaEventCreate(&_evtMlpStart));
        CUDA_CHECK(cudaEventCreate(&_evtMlpStop));
        CUDA_CHECK(cudaEventCreate(&_evtRasterStart));
        CUDA_CHECK(cudaEventCreate(&_evtRasterStop));
        SIBR_LOG << "[V42] Per-stage GPU timing ENABLED (V42_TIMING)." << std::endl;
    }

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

    if (_evtMlpStart)    cudaEventDestroy(_evtMlpStart);
    if (_evtMlpStop)     cudaEventDestroy(_evtMlpStop);
    if (_evtRasterStart) cudaEventDestroy(_evtRasterStart);
    if (_evtRasterStop)  cudaEventDestroy(_evtRasterStop);
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

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
        // Which CUDA device backs the *current GL context*? If it isn't the device
        // the rasterizer runs on (_device), interop registers OK but the mapped
        // pointer is cross-device and the CUDA writes never reach the GL buffer —
        // the classic "registers, then black". cudaGLGetDevices needs the GL
        // context current (it is here — this runs on the render thread).
        unsigned int glDevCount = 0;
        int          glDevs[8]  = {0};
        cudaError_t de = cudaGLGetDevices(&glDevCount, glDevs, 8, cudaGLDeviceListAll);
        if (de == cudaSuccess && glDevCount > 0) {
            std::ostringstream oss;
            bool match = false;
            for (unsigned int i = 0; i < glDevCount; ++i) {
                oss << glDevs[i] << (i + 1 < glDevCount ? "," : "");
                if (glDevs[i] == _device) match = true;
            }
            SIBR_LOG << "[V42] GL context is on CUDA device(s) [" << oss.str()
                     << "]; rasterizer uses device " << _device << "." << std::endl;
            if (!match)
                SIBR_LOG << "[V42] *** DEVICE MISMATCH *** GL context and CUDA device "
                            "differ — interop writes will land on the wrong GPU (black "
                            "image). Run the rasterizer on the GL device instead." << std::endl;
        } else {
            SIBR_LOG << "[V42] cudaGLGetDevices failed (" << cudaGetErrorString(de)
                     << ") — current GL context may not be CUDA-interop capable "
                        "(remote session, wrong adapter, or OpenXR-owned context)." << std::endl;
            cudaGetLastError();  // clear sticky error
        }

        cudaError_t re = cudaGraphicsGLRegisterBuffer(
            &_cudaBuffer, _glBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
        if (re != cudaSuccess) {
            SIBR_LOG << "[V42] cudaGraphicsGLRegisterBuffer FAILED: "
                     << cudaGetErrorString(re) << " — falling back." << std::endl;
            cudaGetLastError();  // clear sticky error
            _useInterop = false;
        } else {
            SIBR_LOG << "[V42] CUDA-GL interop ENABLED (buffer registered)." << std::endl;
        }
    }
    if (!_useInterop) {
        _interopFailed = true;
        _fallbackBytes.resize(w * h * 3 * sizeof(float));
        cudaMalloc(&_fallbackCuda, _fallbackBytes.size());
        SIBR_LOG << "[V42] Using host-copy fallback (D2H + glNamedBufferSubData). "
                    "Set V42_INTEROP=1 to attempt zero-copy interop." << std::endl;
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
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV42::handshakeWithPython()
{
    try {
        // 1. Send 4-byte magic (V42E = double-buffer pipelined protocol, wire v2:
        //    cov3D[6] tail instead of scale[3]+rot[4]). Bumping the magic makes an
        //    old Python server refuse the connection loudly rather than mis-parse.
        const char magic[4] = {'V','4','2','E'};
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
            // Layout: xyz[N_max*3] | feat[N_max*K] | R_bwd[N_max*9] | opa[N_max] | cov3D[N_max*6]
            ptrdiff_t off = 0;
            b.xyzPtr   = b.ptr + off; off += (ptrdiff_t)_N_max * 3;
            b.featPtr  = b.ptr + off; off += (ptrdiff_t)_N_max * _K;
            b.rbwdPtr  = b.ptr + off; off += (ptrdiff_t)_N_max * 9;
            b.opaPtr   = b.ptr + off; off += (ptrdiff_t)_N_max * 1;
            b.covPtr   = b.ptr + off;   // cov3D_precomp (6 floats/Gaussian)

            SIBR_LOG << "[V42] Buffer " << i << " ready. ipc_offset=" << ipc_offset
                     << "  src_device=" << src_device << std::endl;
        }

        SIBR_LOG << "[V42] Both IPC buffers ready. Total floats per buffer: "
                 << (ptrdiff_t)_N_max * (_K + 19) << std::endl;

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
    _latestReadyBuf.store(-1, std::memory_order_relaxed);
    _networkRunning.store(true, std::memory_order_relaxed);
    _networkThread = std::thread(&GaussianLiveViewV42::networkThreadFunc, this);
}

void sibr::GaussianLiveViewV42::stopNetworkThread()
{
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
// beginFrame — call once per frame. Non-blocking: snapshots whichever buffer
// the network thread most recently marked ready. Never waits on Python.
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

    if (!_connected) {
        if (!connectTCP()) { _hasData = false; return false; }
    }
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

    // One-shot ground-truth log: where is the eye camera in the same world the
    // avatar lives in (recentered to origin), and where does it look? Useful when
    // re-tuning seatOffset for a new headset/runtime/reference-space.
    static bool s_camLogged = false;
    if (!s_camLogged) {
        s_camLogged = true;
        sibr::Vector3f p = eye.position();
        sibr::Vector3f d = eye.dir();
        SIBR_LOG << "[V42] CAM world pos=(" << p.x() << ", " << p.y() << ", " << p.z()
                 << ")  dir=(" << d.x() << ", " << d.y() << ", " << d.z()
                 << ")  (avatar centered at origin, ~1.66m tall)" << std::endl;
    }

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

    // ── Run Color MLP on IPC data (zero-copy) ─────────────────────────────────
    // No device sync here: the MLP output and the rasterizer below run on the
    // same (default) stream, so ordering is guaranteed without draining the GPU.
    // Timing, when enabled, is done with CUDA events (measured, not blocking).
    if (_timingEnabled) CUDA_CHECK(cudaEventRecord(_evtMlpStart, 0));
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
    if (_timingEnabled) CUDA_CHECK(cudaEventRecord(_evtMlpStop, 0));

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

    // ── Rasterize (reads xyz, opa, cov3D directly from IPC buffer) ────────────
    // Uses the cov3D_precomp path (scales/rotations = nullptr) so the splat shape
    // matches render.py exactly — the covariance already encodes the posed LBS
    // scale/shear that a scale+quaternion pair would lose.
    if (_timingEnabled) CUDA_CHECK(cudaEventRecord(_evtRasterStart, 0));
    CudaRasterizer::Rasterizer::forward(
        _geomBuf, _binBuf, _imgBuf,
        _currentNLive, /*D=*/0, /*M=*/0,
        _bg_cuda,
        (int)w, (int)h,
        buf.xyzPtr,            // positions (IPC, zero-copy)
        /*shs=*/nullptr,
        colors_ptr,            // per-eye precomputed RGB from Color MLP
        buf.opaPtr,            // opacity (IPC, zero-copy)
        /*scales=*/nullptr,    // using cov3D_precomp instead
        /*scale_modifier=*/1.0f,
        /*rotations=*/nullptr,
        buf.covPtr,            // cov3D_precomp (IPC, zero-copy)
        _view_cuda, _proj_cuda, _camPos_cuda,
        tan_fovx, tan_fovy,
        /*prefiltered=*/false,
        image_cuda
    );
    if (_timingEnabled) CUDA_CHECK(cudaEventRecord(_evtRasterStop, 0));

    // Accumulate per-stage GPU times from the events. cudaEventElapsedTime needs
    // the stop event complete, so we sync ONLY on it (a single stream sync, not a
    // full-device drain). When timing is off, nothing here blocks — the GL unmap
    // / fallback D2H copy below provides the only (necessary) synchronization.
    if (_timingEnabled) {
        CUDA_CHECK(cudaEventSynchronize(_evtRasterStop));
        float mlp_ms = 0.f, raster_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&mlp_ms,    _evtMlpStart,    _evtMlpStop));
        CUDA_CHECK(cudaEventElapsedTime(&raster_ms, _evtRasterStart, _evtRasterStop));
        _mlpMsAccum    += mlp_ms;
        _rasterMsAccum += raster_ms;
    }

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

    // onRenderIBR is called _viewsPerFrame times per Python frame: twice under
    // OpenXR (left, then right eye), once for a mono desktop window. Once every
    // view has read this buffer slot it is safe to overwrite — record the
    // read-complete event so Python's wait_read_complete() can unblock and
    // reuse this slot for a future frame. Log the per-frame total — mlp/raster
    // summed across the frame's views — matching Python's _LOG_INTERVAL=60 cadence.
    _eyeCallsThisFrame++;
    if (_eyeCallsThisFrame >= _viewsPerFrame)
    {
        _eyeCallsThisFrame = 0;
        CUDA_CHECK(cudaEventRecord(buf.readCompleteEvt, 0));

        _frameCount++;
        if (_frameCount % kLogIntervalFrames == 0 || _frameCount <= 3)
        {
            std::ostringstream timing;
            if (_timingEnabled) {
                timing << "  mlp(L+R)=" << _mlpMsAccum << "ms"
                       << "  raster(L+R)=" << _rasterMsAccum << "ms"
                       << "  render_total=" << (_mlpMsAccum + _rasterMsAccum) << "ms";
            } else {
                timing << "  (stage timing off; set V42_TIMING=1 to measure)";
            }
            SIBR_LOG << "[V42] frame " << _frameCount
                     << "  pyFrameId=" << _currentFrameId
                     << "  buf=" << _currentBufIdx
                     << "  beginFrame=" << _beginFrameMsThisFrame << "ms"
                     << timing.str()
                     << std::endl;
        }
        _mlpMsAccum    = 0.0;
        _rasterMsAccum = 0.0;
    }
}

bool sibr::GaussianLiveViewV42::sendControl(const char* magic, int payload)
{
    if (!_connected || !_socket) {
        SIBR_LOG << "[V42] control " << magic << "(" << payload
                 << ") ignored — not connected." << std::endl;
        return false;
    }
    // 4-byte magic + int32 payload, little-endian (matches vr_viewer.protocol).
    unsigned char msg[8];
    std::memcpy(msg, magic, 4);
    std::memcpy(msg + 4, &payload, 4);
    try {
        std::lock_guard<std::mutex> lk(_sendMutex);
        tcp_send_all(*_socket, msg, sizeof(msg));
        return true;
    } catch (const std::exception& e) {
        SIBR_LOG << "[V42] control " << magic << " failed: " << e.what() << std::endl;
        return false;
    }
}

void sibr::GaussianLiveViewV42::sendIdentity(int id)
{
    if (id < 0) id = 0;
    if (id >= kNumIdentities) id = kNumIdentities - 1;
    if (sendControl("CTL0", id))
        SIBR_LOG << "[V42] sent identity switch -> " << id
                 << " (subject " << kSubjectNames[id] << ")" << std::endl;
}

void sibr::GaussianLiveViewV42::setPaused(bool paused)
{
    if (!sendControl("CTL1", paused ? 1 : 0))
        return;   // leave _uiPaused alone so the GUI keeps matching Python
    _uiPaused = paused;
    SIBR_LOG << "[V42] animation " << (paused ? "paused" : "resumed") << std::endl;
}

void sibr::GaussianLiveViewV42::togglePause()
{
    setPaused(!_uiPaused);
}

void sibr::GaussianLiveViewV42::stepFrame(int delta)
{
    // Stepping only means something while paused, and Python ignores it otherwise
    // — so pause first rather than dropping the request on the floor.
    if (!_uiPaused) {
        setPaused(true);
        if (!_uiPaused) return;   // send failed; nothing to step
    }
    if (sendControl("CTL2", delta))
        SIBR_LOG << "[V42] step frame " << (delta >= 0 ? "+" : "") << delta << std::endl;
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

        // ── Render resolution (mono desktop only) ────────────────────────────
        // The OpenXR app already has this in OpenXRRdrMode's own panel, driven by
        // the headset resolution — it reasserts setResolution() every frame, so a
        // second slider here would be overridden and misleading. Mono has no such
        // owner: the resolution is set once at construction and never touched.
        if (_viewsPerFrame == 1) {
            ImGui::Separator();
            ImGui::Text("Render resolution");
            ImGui::Text("Window (native) : %ix%i", _nativeRes.x(), _nativeRes.y());
            ImGui::Text("Rendering       : %ix%i",
                        _nativeRes.x() / _uiDownscale, _nativeRes.y() / _uiDownscale);
            // Rasterizing fewer pixels is the cheapest framerate lever here; the
            // copy shader stretches the smaller buffer over the full window.
            if (ImGui::SliderInt("Down scale factor", &_uiDownscale, 1, 8)) {
                if (_uiDownscale < 1) _uiDownscale = 1;
                setResolution(sibr::Vector2i(std::max(1, _nativeRes.x() / _uiDownscale),
                                             std::max(1, _nativeRes.y() / _uiDownscale)));
            }
        }

        // ── Animation pause / frame step ─────────────────────────────────────
        // Freezes the body pose only; the camera stays live so a frozen pose can
        // be inspected from any angle. "Python frame id" above keeps ticking while
        // paused (Python re-announces the held buffer) — watch the pose, not the id.
        ImGui::Separator();
        ImGui::Text("Animation");
        ImGui::Text("State: %s", _uiPaused ? "PAUSED" : "playing");

        if (ImGui::Button(_uiPaused ? "Resume (P)" : "Pause (P)"))
            togglePause();

        // Always live: stepFrame() pauses first, so stepping while playing does the
        // obvious thing. (No BeginDisabled here — this ImGui is 1.60, which predates it.)
        ImGui::SameLine();
        if (ImGui::Button("< Frame (Left)")) stepFrame(-1);
        ImGui::SameLine();
        if (ImGui::Button("Frame > (Right)")) stepFrame(+1);

        // ── Live appearance-identity switch ──────────────────────────────────
        ImGui::Separator();
        ImGui::Text("Appearance identity");
        if (_uiIdentity < 0) _uiIdentity = 0;
        if (_uiIdentity >= kNumIdentities) _uiIdentity = kNumIdentities - 1;
        ImGui::Text("Selected: %d  (subject %s)", _uiIdentity, kSubjectNames[_uiIdentity]);

        if (ImGui::Button("< Prev")) {
            _uiIdentity = (_uiIdentity - 1 + kNumIdentities) % kNumIdentities;
            sendIdentity(_uiIdentity);
        }
        ImGui::SameLine();
        if (ImGui::Button("Next >")) {
            _uiIdentity = (_uiIdentity + 1) % kNumIdentities;
            sendIdentity(_uiIdentity);
        }

        // Direct per-identity buttons.
        for (int i = 0; i < kNumIdentities; ++i) {
            if (i % 4 != 0) ImGui::SameLine();
            char label[16];
            std::snprintf(label, sizeof(label), "%s##id%d", kSubjectNames[i], i);
            if (ImGui::Button(label)) {
                _uiIdentity = i;
                sendIdentity(_uiIdentity);
            }
        }
    }
    ImGui::End();
}
