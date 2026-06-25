/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveView implementation.
 * Mirrors the CUDA-GL interop pattern from gaussianviewer/renderer/GaussianView.cpp
 * but replaces PLY loading + SH rasterization with TCP receive + precomputed colors.
 */

#include "GaussianLiveView.hpp"

#include <core/graphics/GUI.hpp>

#include <rasterizer.h>   // CudaRasterizer::Rasterizer::forward()

#include <boost/asio.hpp>
#include <GL/glew.h>

#include <cstring>
#include <stdexcept>
#include <thread>

using boost::asio::ip::tcp;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error helper (matches GaussianView convention)
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            SIBR_ERR << "CUDA error: " << cudaGetErrorString(_e)               \
                     << "  (" #call ")  line " << __LINE__ << std::endl;        \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Resizable CUDA buffer helper (same pattern as GaussianView.cpp)
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
// Embedded copy shader source
// (identical to gaussianviewer/renderer/shaders/copy.{vert,frag})
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
// TCP helpers (synchronous boost::asio)
// ─────────────────────────────────────────────────────────────────────────────

static void tcp_recv_all(tcp::socket& sock, void* buf, size_t n)
{
    boost::asio::read(sock, boost::asio::buffer(buf, n));
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

sibr::GaussianLiveView::GaussianLiveView(
    const std::string& ip, int port,
    uint render_w, uint render_h,
    bool white_bg, int device)
    : sibr::ViewBase(render_w, render_h)
    , _ip(ip), _port(port)
    , _white_bg(white_bg), _device(device)
{
    // CUDA device
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (device >= num_devices)
        SIBR_ERR << "Invalid CUDA device " << device << std::endl;
    CUDA_CHECK(cudaSetDevice(device));

    // Fixed-size GPU scratch (view/proj/campos/bg)
    CUDA_CHECK(cudaMalloc(&_view_cuda,   16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_proj_cuda,   16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_camPos_cuda,  3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_bg_cuda,      3 * sizeof(float)));

    float bg[3] = { white_bg ? 1.f : 0.f,
                    white_bg ? 1.f : 0.f,
                    white_bg ? 1.f : 0.f };
    CUDA_CHECK(cudaMemcpy(_bg_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Rasterizer scratch buffers
    _geomBuf = resizeFunctional(&_geomPtr, _allocGeom);
    _binBuf  = resizeFunctional(&_binPtr,  _allocBin);
    _imgBuf  = resizeFunctional(&_imgPtr,  _allocImg);

    // GL + CUDA-GL image buffer
    recreateImageBuffer(render_w, render_h);

    // Copy shader (source embedded above)
    initShader();

    SIBR_LOG << "[GaussianLiveView] Connecting to Python at "
             << ip << ":" << port << std::endl;
    connectTCP();
}

sibr::GaussianLiveView::~GaussianLiveView()
{
    if (_cudaBuffer) cudaGraphicsUnregisterResource(_cudaBuffer);
    if (_glBuffer)   glDeleteBuffers(1, &_glBuffer);
    if (_fallbackCuda) cudaFree(_fallbackCuda);

    if (_pos_cuda)     cudaFree(_pos_cuda);
    if (_colors_cuda)  cudaFree(_colors_cuda);
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

void sibr::GaussianLiveView::initShader()
{
    _copyShader.init("GaussianLiveCopy", COPY_VERT_SRC, COPY_FRAG_SRC);
    _flipU.init(_copyShader,   "flip");
    _widthU.init(_copyShader,  "width");
    _heightU.init(_copyShader, "height");
}

void sibr::GaussianLiveView::recreateImageBuffer(uint w, uint h)
{
    if (_cudaBuffer) { cudaGraphicsUnregisterResource(_cudaBuffer); _cudaBuffer = nullptr; }
    if (_glBuffer)   { glDeleteBuffers(1, &_glBuffer); _glBuffer = 0; }
    if (_fallbackCuda) { cudaFree(_fallbackCuda); _fallbackCuda = nullptr; }

    // GL storage buffer: W*H*3 floats (planar RGB — rasterizer output format)
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
        SIBR_LOG << "[GaussianLiveView] CUDA-GL interop failed, using fallback." << std::endl;
    }
}

void sibr::GaussianLiveView::resizeGaussianBuffers(int N)
{
    if (N <= _allocN) return;   // existing capacity is enough

    if (_pos_cuda)     cudaFree(_pos_cuda);
    if (_colors_cuda)  cudaFree(_colors_cuda);
    if (_opacity_cuda) cudaFree(_opacity_cuda);
    if (_scale_cuda)   cudaFree(_scale_cuda);
    if (_rot_cuda)     cudaFree(_rot_cuda);

    CUDA_CHECK(cudaMalloc(&_pos_cuda,     N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_colors_cuda,  N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_opacity_cuda, N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_scale_cuda,   N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&_rot_cuda,     N * 4 * sizeof(float)));

    _allocN = N;
}

bool sibr::GaussianLiveView::connectTCP()
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
        SIBR_LOG << "[GaussianLiveView] Connected to Python." << std::endl;
        return true;
    } catch (const std::exception& e) {
        SIBR_LOG << "[GaussianLiveView] TCP connect error: " << e.what() << std::endl;
        _connected = false;
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// setResolution — called by OpenXRRdrMode when headset resolution changes
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveView::setResolution(const sibr::Vector2i& size)
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

bool sibr::GaussianLiveView::fetchFromPython()
{
    if (!_connected) {
        if (!connectTCP()) return false;
    }

    try {
        // Send 13-byte frame request:
        //   byte 0      : ping (value ignored by Python)
        //   bytes 1–12  : last known eye position (3 × float32 LE)
        // Python uses this camera center to compute view-dependent colors correctly.
        // One-frame lag is unavoidable (eye pose is not yet known before onRenderIBR).
        char req_buf[13] = {};
        memcpy(req_buf + 1, _lastEyePosition.data(), 3 * sizeof(float));
        boost::asio::write(*_socket, boost::asio::buffer(req_buf, 13));

        // Receive N (number of Gaussians)
        uint32_t N = 0;
        tcp_recv_all(*_socket, &N, sizeof(uint32_t));

        // Resize GPU buffers if needed
        resizeGaussianBuffers((int)N);
        _N = (int)N;

        // CPU staging buffers
        std::vector<float> xyz_cpu    (N * 3);
        std::vector<float> colors_cpu (N * 3);
        std::vector<float> opacity_cpu(N);
        std::vector<float> scale_cpu  (N * 3);
        std::vector<float> rot_cpu    (N * 4);

        tcp_recv_all(*_socket, xyz_cpu.data(),     N * 3 * sizeof(float));
        tcp_recv_all(*_socket, colors_cpu.data(),  N * 3 * sizeof(float));
        tcp_recv_all(*_socket, opacity_cpu.data(), N     * sizeof(float));
        tcp_recv_all(*_socket, scale_cpu.data(),   N * 3 * sizeof(float));
        tcp_recv_all(*_socket, rot_cpu.data(),     N * 4 * sizeof(float));

        // Upload to GPU
        CUDA_CHECK(cudaMemcpy(_pos_cuda,     xyz_cpu.data(),     N * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_colors_cuda,  colors_cpu.data(),  N * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_opacity_cuda, opacity_cpu.data(), N     * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_scale_cuda,   scale_cpu.data(),   N * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_rot_cuda,     rot_cpu.data(),     N * 4 * sizeof(float), cudaMemcpyHostToDevice));

        _hasData = true;
        return true;

    } catch (const std::exception& e) {
        SIBR_LOG << "[GaussianLiveView] fetch error: " << e.what()
                 << " — reconnecting." << std::endl;
        _connected = false;
        _hasData   = false;
        _socket.reset();
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// onRenderIBR — called by OpenXRRdrMode per eye
// ─────────────────────────────────────────────────────────────────────────────

void sibr::GaussianLiveView::onRenderIBR(sibr::IRenderTarget& dst,
                                          const sibr::Camera& eye)
{
    if (!_hasData || _N == 0) return;

    uint w = (uint)_resolution.x();
    uint h = (uint)_resolution.y();

    // ── Build view/proj matching 3DGS rasterizer coordinate convention ────────
    // OpenXRRdrMode already applies the Y-flip to the camera pose.
    // GaussianView.cpp applies an additional row-flip to match the rasterizer:
    //   view_mat.row(1) *= -1; view_mat.row(2) *= -1; proj_mat.row(1) *= -1;
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

    // ── Map the GL buffer for CUDA write ─────────────────────────────────────
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
    // Using colors_precomp (D=0, shs=nullptr) — no SH evaluation needed.
    CudaRasterizer::Rasterizer::forward(
        _geomBuf, _binBuf, _imgBuf,
        _N, /*D=*/0, /*M=*/0,
        _bg_cuda,
        (int)w, (int)h,
        _pos_cuda,
        /*shs=*/nullptr,
        _colors_cuda,    // precomputed RGB
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

    // ── Unmap and blit to render target ──────────────────────────────────────
    if (!_interopFailed) {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cudaBuffer));
    } else {
        cudaMemcpy(_fallbackBytes.data(), _fallbackCuda,
                   _fallbackBytes.size(), cudaMemcpyDeviceToHost);
        glNamedBufferSubData(_glBuffer, 0,
                             (GLsizeiptr)_fallbackBytes.size(),
                             _fallbackBytes.data());
    }

    // Store eye position so the next fetchFromPython() sends it to Python.
    // Python overrides camera_center before running the Color MLP so view-
    // dependent shading uses the actual viewer direction, not a dataset camera.
    _lastEyePosition = eye.position();

    // ── Copy shader: GL storage buffer -> IRenderTarget ───────────────────────
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

void sibr::GaussianLiveView::onGUI()
{
    if (ImGui::Begin("GaussianLiveView")) {
        ImGui::Text("Gaussians: %d", _N);
        ImGui::Text("Connected: %s", _connected ? "yes" : "no");
        ImGui::Text("Has data:  %s", _hasData   ? "yes" : "no");
    }
    ImGui::End();
}
