/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveViewV42 implementation  (v4.2)
 *
 * Transport upgrade over v4.0:
 *   v4.0: Python → TCP → CPU staging → cudaMemcpy → GPU (every frame, ~2 MB)
 *   v4.2: Python → CUDA IPC shared buffer (zero-copy, no host involvement)
 *
 * Per-frame sequence (after handshake):
 *   1. C++ sends 1-byte ping.
 *   2. Python: deform → extract features → attr_buf.write() → cudaEventRecord → send N_live.
 *   3. C++ receives N_live; calls cudaEventSynchronize (instant, event already recorded).
 *   4. C++ runs Color MLP + CudaRasterizer directly on IPC pointers.
 */

#include "GaussianLiveViewV42.hpp"

#include <core/graphics/GUI.hpp>
#include <rasterizer.h>

#include <boost/asio.hpp>
#include <GL/glew.h>

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
    // IPC cleanup — must happen before destroying CUDA context
    if (_ipc_event) { cudaEventDestroy(_ipc_event); _ipc_event = nullptr; }
    if (_ipc_base)  { cudaIpcCloseMemHandle(_ipc_base); _ipc_base = nullptr; }

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
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV42::handshakeWithPython()
{
    try {
        // 1. Send 4-byte magic
        const char magic[4] = {'V','4','2','0'};
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

        // 4. Receive CUDA IPC memory handle (64 bytes)
        cudaIpcMemHandle_t mem_handle;
        tcp_recv_all(*_socket, &mem_handle, sizeof(cudaIpcMemHandle_t));

        // 5. Receive CUDA IPC event handle (64 bytes)
        cudaIpcEventHandle_t evt_handle;
        tcp_recv_all(*_socket, &evt_handle, sizeof(cudaIpcEventHandle_t));

        // 6. Receive IPC offset (uint64) and source device (int32)
        uint64_t ipc_offset = 0;
        int32_t  src_device = 0;
        tcp_recv_all(*_socket, &ipc_offset, sizeof(uint64_t));
        tcp_recv_all(*_socket, &src_device, sizeof(int32_t));
        SIBR_LOG << "[V42] ipc_offset=" << ipc_offset
                 << "  src_device=" << src_device << std::endl;

        // 7. Open IPC memory handle → _ipc_base
        CUDA_CHECK(cudaSetDevice(_device));
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &_ipc_base, mem_handle, cudaIpcMemLazyEnablePeerAccess));
        _ipc_ptr = reinterpret_cast<float*>(
            static_cast<char*>(_ipc_base) + ipc_offset);

        // 8. Open IPC event handle → _ipc_event
        CUDA_CHECK(cudaIpcOpenEventHandle(&_ipc_event, evt_handle));

        // 9. Pre-compute field pointers into the IPC buffer
        //    Layout: xyz[N_max*3] | feat[N_max*K] | R_bwd[N_max*9] | opa[N_max] | scale[N_max*3] | rot[N_max*4]
        ptrdiff_t off = 0;
        _ipc_xyz_ptr   = _ipc_ptr + off; off += (ptrdiff_t)_N_max * 3;
        _ipc_feat_ptr  = _ipc_ptr + off; off += (ptrdiff_t)_N_max * _K;
        _ipc_rbwd_ptr  = _ipc_ptr + off; off += (ptrdiff_t)_N_max * 9;
        _ipc_opa_ptr   = _ipc_ptr + off; off += (ptrdiff_t)_N_max * 1;
        _ipc_scale_ptr = _ipc_ptr + off; off += (ptrdiff_t)_N_max * 3;
        _ipc_rot_ptr   = _ipc_ptr + off;

        SIBR_LOG << "[V42] IPC buffer ready. Total floats: "
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
// fetchFromPython — call once per frame
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV42::fetchFromPython()
{
    if (!_connected) {
        if (!connectTCP()) return false;
    }

    try {
        if (!_handshakeDone) {
            if (!handshakeWithPython()) {
                _connected = false;
                return false;
            }
        }

        // Send 1-byte ping: "ready for next frame"
        const char ping = 0x00;
        boost::asio::write(*_socket, boost::asio::buffer(&ping, 1));

        // Receive N_live. Python sends this AFTER recording the CUDA event,
        // so the event is always already complete when we call cudaEventSynchronize
        // in onRenderIBR — that call returns instantly.
        uint32_t n = 0;
        tcp_recv_all(*_socket, &n, sizeof(uint32_t));
        _N_live  = (int)n;
        _hasData = true;
        return true;

    } catch (const std::exception& e) {
        SIBR_LOG << "[V42] fetch error: " << e.what() << " — reconnecting." << std::endl;
        _connected     = false;
        _handshakeDone = false;
        _hasData       = false;
        _mlpLoaded     = false;
        _socket.reset();
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// onRenderIBR — called per eye by OpenXRRdrMode
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveViewV42::onRenderIBR(sibr::IRenderTarget& dst,
                                              const sibr::Camera& eye)
{
    if (!_mlpLoaded || !_hasData || _N_live == 0) return;

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

    // ── Wait for Python to finish writing IPC buffer ──────────────────────────
    // TCP ordering guarantees the event was recorded before Python sent N_live,
    // and we received N_live before reaching here. This call returns immediately.
    CUDA_CHECK(cudaEventSynchronize(_ipc_event));

    // ── Run Color MLP on IPC data (zero-copy) ─────────────────────────────────
    {
        auto dev  = torch::Device(torch::kCUDA, _device);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        // from_blob: LibTorch wraps the IPC pointer without owning it.
        auto feat_t = torch::from_blob(_ipc_feat_ptr,  {_N_live, _K},    opts);
        auto xyz_t  = torch::from_blob(_ipc_xyz_ptr,   {_N_live, 3},     opts);
        auto rbwd_t = torch::from_blob(_ipc_rbwd_ptr,  {_N_live, 3, 3},  opts);

        sibr::Vector3f pos = eye.position();
        auto cam_t = torch::tensor({pos.x(), pos.y(), pos.z()}, opts);

        torch::NoGradGuard no_grad;
        _colors_t = _colorMLP.forward({feat_t, xyz_t, cam_t, rbwd_t})
                              .toTensor()
                              .contiguous();   // [N_live, 3] on CUDA
    }

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
    CudaRasterizer::Rasterizer::forward(
        _geomBuf, _binBuf, _imgBuf,
        _N_live, /*D=*/0, /*M=*/0,
        _bg_cuda,
        (int)w, (int)h,
        _ipc_xyz_ptr,          // positions (IPC, zero-copy)
        /*shs=*/nullptr,
        colors_ptr,            // per-eye precomputed RGB from Color MLP
        _ipc_opa_ptr,          // opacity (IPC, zero-copy)
        _ipc_scale_ptr,        // scale (IPC, zero-copy)
        /*scale_modifier=*/1.0f,
        _ipc_rot_ptr,          // rotation quaternion (IPC, zero-copy)
        /*cov3D_precomp=*/nullptr,
        _view_cuda, _proj_cuda, _camPos_cuda,
        tan_fovx, tan_fovy,
        /*prefiltered=*/false,
        image_cuda
    );

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
}

void sibr::GaussianLiveViewV42::onGUI()
{
    if (ImGui::Begin("GaussianLiveViewV42")) {
        ImGui::Text("v4.2 — zero-copy CUDA IPC transport");
        ImGui::Text("Gaussians (live): %d / %d", _N_live, _N_max);
        ImGui::Text("Feat dim K      : %d",  _K);
        ImGui::Text("Connected       : %s",  _connected     ? "yes" : "no");
        ImGui::Text("MLP loaded      : %s",  _mlpLoaded     ? "yes" : "no");
        ImGui::Text("IPC ready       : %s",  (_ipc_ptr != nullptr) ? "yes" : "no");
    }
    ImGui::End();
}
