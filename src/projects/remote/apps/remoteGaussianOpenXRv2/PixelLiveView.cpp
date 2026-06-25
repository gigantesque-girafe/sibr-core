/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * PixelLiveView implementation.
 * Receives pre-rendered stereo RGB frames from Python, displays via OpenXR.
 * No CUDA, no rasterizer — pure GL texture blit.
 */

#include "PixelLiveView.hpp"

#include <core/graphics/GUI.hpp>

#include <boost/asio.hpp>
#include <GL/glew.h>

#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <thread>

using boost::asio::ip::tcp;

// ─────────────────────────────────────────────────────────────────────────────
// Embedded blit shaders
// Vertex: identical attribute layout to GaussianLiveView copy shader so that
//         RenderUtility::renderScreenQuad() works without any changes.
// Fragment: samples a 2D texture, flips Y (Python images are top-to-bottom;
//           OpenGL textures stored bottom-to-top).
// ─────────────────────────────────────────────────────────────────────────────

static const char* BLIT_VERT_SRC = R"GLSL(
#version 450
layout(location = 0) in vec4 in_vertex;
layout(location = 1) in vec4 in_texcoord;
out vec2 vUV;
void main() {
    gl_Position = in_vertex;
    vUV = in_texcoord.xy;
}
)GLSL";

static const char* BLIT_FRAG_SRC = R"GLSL(
#version 450
layout(binding = 0) uniform sampler2D uTex;
in vec2 vUV;
out vec4 out_color;
void main() {
    // Python images are top-to-bottom; GL texture origin is bottom-left.
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);
    out_color = vec4(texture(uTex, uv).rgb, 1.0);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
// TCP helper
// ─────────────────────────────────────────────────────────────────────────────

static void tcp_recv_all(tcp::socket& sock, void* buf, size_t n)
{
    boost::asio::read(sock, boost::asio::buffer(buf, n));
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string floatsToJson(const float* v, int n)
{
    std::ostringstream ss;
    ss << std::setprecision(8) << "[";
    for (int i = 0; i < n; i++) {
        if (i) ss << ",";
        ss << v[i];
    }
    ss << "]";
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

sibr::PixelLiveView::PixelLiveView(
    const std::string& ip, int port,
    uint render_w, uint render_h)
    : sibr::ViewBase(render_w, render_h)
    , _ip(ip), _port(port)
{
    recreateTextures(render_w, render_h);
    initShader();

    SIBR_LOG << "[PixelLiveView] Connecting to Python at "
             << ip << ":" << port << std::endl;
    connectTCP();
}

sibr::PixelLiveView::~PixelLiveView()
{
    if (_tex[0]) glDeleteTextures(2, _tex);
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

void sibr::PixelLiveView::initShader()
{
    _blitShader.init("PixelLiveBlit", BLIT_VERT_SRC, BLIT_FRAG_SRC);
}

void sibr::PixelLiveView::recreateTextures(uint w, uint h)
{
    if (_tex[0]) glDeleteTextures(2, _tex);

    glCreateTextures(GL_TEXTURE_2D, 2, _tex);
    for (int i = 0; i < 2; i++) {
        glTextureStorage2D(_tex[i], 1, GL_RGB8, (GLsizei)w, (GLsizei)h);
        glTextureParameteri(_tex[i], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(_tex[i], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(_tex[i], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(_tex[i], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    _pixBuf.resize((size_t)w * h * 3);
}

bool sibr::PixelLiveView::connectTCP()
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
        SIBR_LOG << "[PixelLiveView] Connected to Python." << std::endl;
        return true;
    } catch (const std::exception& e) {
        SIBR_LOG << "[PixelLiveView] TCP connect error: " << e.what() << std::endl;
        _connected = false;
        return false;
    }
}

void sibr::PixelLiveView::setResolution(const sibr::Vector2i& size)
{
    if (size == getResolution()) return;
    ViewBase::setResolution(size);
    recreateTextures((uint)size.x(), (uint)size.y());
}

// ─────────────────────────────────────────────────────────────────────────────
// buildJson — assembles camera JSON to send to Python
// Uses previous frame's eye cameras (one-frame lag, same as GaussianLiveView).
// On the very first frame _hasPrevEye is false; we send identity-ish matrices
// and Python returns a first frame which is usually discarded by the user.
// ─────────────────────────────────────────────────────────────────────────────

std::string sibr::PixelLiveView::buildJson(uint w, uint h) const
{
    std::ostringstream ss;
    ss << std::setprecision(8);
    ss << "{"
       << "\"resolution_x\":" << w << ","
       << "\"resolution_y\":" << h << ","
       << "\"fov_x_left\":"  << _prevFovXL << ","
       << "\"fov_y_left\":"  << _prevFovYL << ","
       << "\"fov_x_right\":" << _prevFovXR << ","
       << "\"fov_y_right\":" << _prevFovYR << ","
       << "\"z_near\":0.01,\"z_far\":100.0,"
       << "\"view_matrix_left\":"             << floatsToJson(_prevViewL.data(), 16) << ","
       << "\"view_projection_matrix_left\":"  << floatsToJson(_prevProjL.data(), 16) << ","
       << "\"view_matrix_right\":"            << floatsToJson(_prevViewR.data(), 16) << ","
       << "\"view_projection_matrix_right\":" << floatsToJson(_prevProjR.data(), 16)
       << "}";
    return ss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// fetchFromPython — called ONCE per frame in the main loop
// ─────────────────────────────────────────────────────────────────────────────

bool sibr::PixelLiveView::fetchFromPython()
{
    if (!_connected) {
        if (!connectTCP()) return false;
    }

    uint w = (uint)_resolution.x();
    uint h = (uint)_resolution.y();
    if (w == 0 || h == 0) return false;

    try {
        // Send camera JSON: uint32_le(len) + bytes
        std::string json = buildJson(w, h);
        uint32_t jlen = (uint32_t)json.size();
        boost::asio::write(*_socket, boost::asio::buffer(&jlen, 4));
        boost::asio::write(*_socket, boost::asio::buffer(json));

        // Receive W*H*3 bytes for each eye, upload to GL texture immediately.
        // Avoids double-buffering: one pixBuf for both receives.
        size_t frameBytes = (size_t)w * h * 3;
        if (_pixBuf.size() < frameBytes)
            _pixBuf.resize(frameBytes);

        // Left eye
        tcp_recv_all(*_socket, _pixBuf.data(), frameBytes);
        glTextureSubImage2D(_tex[0], 0, 0, 0, (GLsizei)w, (GLsizei)h,
                            GL_RGB, GL_UNSIGNED_BYTE, _pixBuf.data());

        // Right eye
        tcp_recv_all(*_socket, _pixBuf.data(), frameBytes);
        glTextureSubImage2D(_tex[1], 0, 0, 0, (GLsizei)w, (GLsizei)h,
                            GL_RGB, GL_UNSIGNED_BYTE, _pixBuf.data());

        _hasData = true;
        _eyeIdx  = 0;   // reset so onRenderIBR[0] = left, [1] = right
        return true;

    } catch (const std::exception& e) {
        SIBR_LOG << "[PixelLiveView] fetch error: " << e.what()
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

void sibr::PixelLiveView::onRenderIBR(sibr::IRenderTarget& dst,
                                       const sibr::Camera& eye)
{
    if (!_hasData) return;

    // Which eye's texture to display: first call = left (0), second = right (1)
    int slot = (_eyeIdx < 2) ? _eyeIdx : 1;

    // Store this eye's camera matrices for the NEXT frame's fetchFromPython.
    // Eigen Matrix4f is column-major, so .data() gives 16 col-major floats —
    // exactly the format render_vr_v2.py expects to receive.
    sibr::Matrix4f vm = eye.view();
    sibr::Matrix4f pm = eye.viewproj();
    float fovy = eye.fovy();   // full vertical FOV in radians
    float fovx = 2.f * std::atan(std::tan(fovy * 0.5f) * eye.aspect());

    if (slot == 0) {
        std::copy(vm.data(), vm.data() + 16, _prevViewL.data());
        std::copy(pm.data(), pm.data() + 16, _prevProjL.data());
        _prevFovXL = fovx;
        _prevFovYL = fovy;
    } else {
        std::copy(vm.data(), vm.data() + 16, _prevViewR.data());
        std::copy(pm.data(), pm.data() + 16, _prevProjR.data());
        _prevFovXR = fovx;
        _prevFovYR = fovy;
        _hasPrevEye = true; // both eyes stored; JSON from next fetch will be valid
    }
    _eyeIdx++;

    // Blit: bind texture to unit 0, render full-screen quad to dst
    glDisable(GL_DEPTH_TEST);
    _blitShader.begin();

    glBindTextureUnit(0, _tex[slot]);

    dst.clear();
    dst.bind();
    sibr::RenderUtility::renderScreenQuad();
    dst.unbind();

    _blitShader.end();

    CHECK_GL_ERROR;
}

// ─────────────────────────────────────────────────────────────────────────────
// onGUI
// ─────────────────────────────────────────────────────────────────────────────

void sibr::PixelLiveView::onGUI()
{
    if (ImGui::Begin("PixelLiveView")) {
        ImGui::Text("Connected: %s", _connected ? "yes" : "no");
        ImGui::Text("Has data:  %s", _hasData   ? "yes" : "no");
        ImGui::Text("Res: %d x %d", _resolution.x(), _resolution.y());
    }
    ImGui::End();
}
