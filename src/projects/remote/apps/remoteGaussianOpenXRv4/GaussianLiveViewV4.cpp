/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveViewV4 implementation  (v4.0)
 *
 * Protocol differences from v3:
 *   Startup :  C++ sends b"V400"; Python responds with K (uint32) + model (TorchScript).
 *   Per frame: receive N, xyz[N*3], feat[N*K], R_bwd[N*9], opacity[N], scale[N*3], rot[N*4].
 *   Per eye  : run _colorMLP(feat, xyz, cam_center, R_bwd) → colors[N*3], then rasterize.
 */

#include "GaussianLiveViewV4.hpp"

#include <core/graphics/GUI.hpp>
#include <rasterizer.h>   // CudaRasterizer::Rasterizer::forward()

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
// Resizable CUDA buffer helper
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
// Embedded copy-shader source (same as v3)
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

sibr::GaussianLiveViewV4::GaussianLiveViewV4(
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
        SIBR_ERR << "[V4] Invalid CUDA device " << device << std::endl;
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

    SIBR_LOG << "[V4] Connecting to Python at " << ip << ":" << port << std::endl;
    connectTCP();
}

sibr::GaussianLiveViewV4::~GaussianLiveViewV4()
{
    if (_cudaBuffer)  cudaGraphicsUnregisterResource(_cudaBuffer);
    if (_glBuffer)    glDeleteBuffers(1, &_glBuffer);
    if (_fallbackCuda) cudaFree(_fallbackCuda);

    if (_pos_cuda)     cudaFree(_pos_cuda);
    if (_feat_cuda)    cudaFree(_feat_cuda);
    if (_rbwd_cuda)    cudaFree(_rbwd_cuda);
    if (_opacity_cuda) cudaFree(_opacity_cuda);
    if (_scale_cuda)   cudaFree(_scale_cuda);
    if (_rot_cuda)     cudaFree(_rot_cuda);

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

void sibr::GaussianLiveViewV4::initShader()
{
    _copyShader.init("GaussianLiveV4Copy", COPY_VERT_SRC, COPY_FRAG_SRC);
    _flipU.init(_copyShader,   "flip");
    _widthU.init(_copyShader,  "width");
    _heightU.init(_copyShader, "height");
}

void sibr::GaussianLiveViewV4::recreateImageBuffer(uint w, uint h)
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
        SIBR_LOG << "[V4] CUDA-GL interop unavailable, using fallback." << std::endl;
    }
}

void sibr::GaussianLiveViewV4::resizeGaussianBuffers(int N)
{
    if (N <= _allocN) return;

    if (_pos_cuda)     cudaFree(_pos_cuda);
    if (_feat_cuda)    cudaFree(_feat_cuda);
    if (_rbwd_cuda)    cudaFree(_rbwd_cuda);
    if (_opacity_cuda) cudaFree(_opacity_cuda);
    if (_scale_cuda)   cudaFree(_scale_cuda);
    if (_rot_cuda)     cudaFree(_rot_cuda);

    CUDA_CHECK(cudaMalloc(&_pos_cuda,     N * 3  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_feat_cuda,    N * _K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_rbwd_cuda,    N * 9  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_opacity_cuda, N      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_scale_cuda,   N * 3  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_rot_cuda,     N * 4  * sizeof(float)));

    _allocN = N;
}

bool sibr::GaussianLiveViewV4::connectTCP()
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
        SIBR_LOG << "[V4] TCP connected to Python." << std::endl;
        return true;
    } catch (const std::exception& e) {
        SIBR_LOG << "[V4] TCP connect error: " << e.what() << std::endl;
        _connected = false;
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Startup handshake: send "V400", receive K + TorchScript model
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV4::handshakeWithPython()
{
    try {
        // Send 4-byte magic
        const char magic[4] = {'V','4','0','0'};
        boost::asio::write(*_socket, boost::asio::buffer(magic, 4));

        // Receive K (feature dim) and model size
        uint32_t K_recv = 0, model_size = 0;
        tcp_recv_all(*_socket, &K_recv,     sizeof(uint32_t));
        tcp_recv_all(*_socket, &model_size, sizeof(uint32_t));

        _K = (int)K_recv;
        SIBR_LOG << "[V4] K=" << _K << "  model_size=" << model_size << " bytes" << std::endl;

        // Receive model bytes
        std::vector<char> model_buf(model_size);
        tcp_recv_all(*_socket, model_buf.data(), model_size);

        // Load TorchScript model from memory
        std::istringstream ss(std::string(model_buf.data(), model_size));
        _colorMLP  = torch::jit::load(ss, torch::Device(torch::kCUDA, _device));
        _colorMLP.eval();
        _mlpLoaded = true;

        SIBR_LOG << "[V4] Color MLP loaded successfully." << std::endl;
        _handshakeDone = true;
        return true;

    } catch (const std::exception& e) {
        SIBR_ERR << "[V4] Handshake failed: " << e.what() << std::endl;
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// setResolution
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveViewV4::setResolution(const sibr::Vector2i& size)
{
    if (size == getResolution()) return;
    ViewBase::setResolution(size);
    recreateImageBuffer((uint)size.x(), (uint)size.y());
    _widthU.get()  = size.x();
    _heightU.get() = size.y();
}

// ─────────────────────────────────────────────────────────────────────────────
// fetchFromPython — call once per frame in the main loop
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::GaussianLiveViewV4::fetchFromPython()
{
    if (!_connected) {
        if (!connectTCP()) return false;
    }

    try {
        // Perform handshake on first successful connection
        if (!_handshakeDone) {
            if (!handshakeWithPython()) {
                _connected = false;
                return false;
            }
        }

        // Send 13-byte frame request (identical to v3 for interop simplicity):
        //   byte 0     : ping
        //   bytes 1-12 : last known eye position (3×float32 LE)
        // Python v4 ignores the eye position (C++ computes view dir from OpenXR directly).
        char req[13] = {};
        memcpy(req + 1, _lastEyePosition.data(), 3 * sizeof(float));
        boost::asio::write(*_socket, boost::asio::buffer(req, 13));

        // Receive N
        uint32_t N = 0;
        tcp_recv_all(*_socket, &N, sizeof(uint32_t));

        resizeGaussianBuffers((int)N);
        _N = (int)N;

        // CPU staging buffers
        std::vector<float> xyz_cpu    (N * 3);
        std::vector<float> feat_cpu   (N * _K);
        std::vector<float> rbwd_cpu   (N * 9);
        std::vector<float> opacity_cpu(N);
        std::vector<float> scale_cpu  (N * 3);
        std::vector<float> rot_cpu    (N * 4);

        tcp_recv_all(*_socket, xyz_cpu.data(),     N * 3  * sizeof(float));
        tcp_recv_all(*_socket, feat_cpu.data(),    N * _K * sizeof(float));
        tcp_recv_all(*_socket, rbwd_cpu.data(),    N * 9  * sizeof(float));
        tcp_recv_all(*_socket, opacity_cpu.data(), N      * sizeof(float));
        tcp_recv_all(*_socket, scale_cpu.data(),   N * 3  * sizeof(float));
        tcp_recv_all(*_socket, rot_cpu.data(),     N * 4  * sizeof(float));

        // Upload to GPU
        CUDA_CHECK(cudaMemcpy(_pos_cuda,     xyz_cpu.data(),     N * 3  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_feat_cuda,    feat_cpu.data(),    N * _K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_rbwd_cuda,    rbwd_cpu.data(),    N * 9  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_opacity_cuda, opacity_cpu.data(), N      * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_scale_cuda,   scale_cpu.data(),   N * 3  * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_rot_cuda,     rot_cpu.data(),     N * 4  * sizeof(float), cudaMemcpyHostToDevice));

        _hasData = true;
        return true;

    } catch (const std::exception& e) {
        SIBR_LOG << "[V4] fetch error: " << e.what() << " — reconnecting." << std::endl;
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

void sibr::GaussianLiveViewV4::onRenderIBR(sibr::IRenderTarget& dst,
                                             const sibr::Camera& eye)
{
    if (!_mlpLoaded || !_hasData || _N == 0) return;

    uint w = (uint)_resolution.x();
    uint h = (uint)_resolution.y();

    // ── View / projection matrices (same convention as v3) ────────────────────
    auto view_mat = eye.view();
    auto proj_mat = eye.viewproj();
    view_mat.row(1) *= -1.f;
    view_mat.row(2) *= -1.f;
    proj_mat.row(1) *= -1.f;

    CUDA_CHECK(cudaMemcpy(_view_cuda,   view_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_proj_cuda,   proj_mat.data(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_camPos_cuda, &eye.position(), 3  * sizeof(float), cudaMemcpyHostToDevice));

    float tan_fovy = std::tan(eye.fovy() * 0.5f);
    float tan_fovx = tan_fovy * eye.aspect();

    // ── Run Color MLP: view-indep features + per-eye cam_center → RGB ─────────
    {
        auto dev  = torch::Device(torch::kCUDA, _device);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        // Zero-copy wraps of CUDA buffers — LibTorch does NOT own these.
        auto feat_t = torch::from_blob(_feat_cuda, {_N, _K},    opts);
        auto xyz_t  = torch::from_blob(_pos_cuda,  {_N, 3},     opts);
        auto rbwd_t = torch::from_blob(_rbwd_cuda, {_N, 3, 3},  opts);

        // Per-eye camera centre as a small CUDA tensor
        sibr::Vector3f pos = eye.position();
        auto cam_t = torch::tensor({pos.x(), pos.y(), pos.z()}, opts);

        // Forward pass — no gradient, keeps _colors_t alive for rasterizer
        torch::NoGradGuard no_grad;
        _colors_t = _colorMLP.forward({feat_t, xyz_t, cam_t, rbwd_t})
                              .toTensor()
                              .contiguous();  // [N, 3] interleaved RGB on CUDA
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

    // ── Rasterize ─────────────────────────────────────────────────────────────
    CudaRasterizer::Rasterizer::forward(
        _geomBuf, _binBuf, _imgBuf,
        _N, /*D=*/0, /*M=*/0,
        _bg_cuda,
        (int)w, (int)h,
        _pos_cuda,
        /*shs=*/nullptr,
        colors_ptr,       // per-eye precomputed RGB from Color MLP
        _opacity_cuda,
        _scale_cuda,
        /*scale_modifier=*/1.0f,
        _rot_cuda,
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

    _lastEyePosition = eye.position();

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

void sibr::GaussianLiveViewV4::onGUI()
{
    if (ImGui::Begin("GaussianLiveViewV4")) {
        ImGui::Text("v4.0 — view-dep MLP on C++");
        ImGui::Text("Gaussians : %d",  _N);
        ImGui::Text("Feat dim K: %d",  _K);
        ImGui::Text("Connected : %s",  _connected     ? "yes" : "no");
        ImGui::Text("MLP loaded: %s",  _mlpLoaded     ? "yes" : "no");
        ImGui::Text("Has data  : %s",  _hasData       ? "yes" : "no");
    }
    ImGui::End();
}
