/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * PixelLiveView: receives pre-rendered stereo RGB pixels from Python via TCP,
 * uploads each eye to a GL texture, blits to the OpenXR swapchain per eye.
 *
 * This is the "v2" pixel-streaming mode:
 *   Python runs deformation + color MLP + rasterizer and sends two full
 *   W×H×3 uint8 images per frame.  C++ does nothing except display them.
 *
 * Call once per frame:   view->fetchFromPython();
 * Called per eye by OpenXRRdrMode: view->onRenderIBR(dst, eyeCam);
 *
 * Protocol (matches render_vr_v2.py / render_openxr_baseline.py):
 *   C++ -> Python : uint32_le(json_len) + JSON bytes
 *     JSON fields: resolution_x, resolution_y,
 *                  fov_x_left, fov_y_left, fov_x_right, fov_y_right,
 *                  z_near, z_far,
 *                  view_matrix_left [16 floats col-major],
 *                  view_projection_matrix_left [16 floats col-major],
 *                  view_matrix_right  [16 floats col-major],
 *                  view_projection_matrix_right [16 floats col-major]
 *   Python -> C++ : W*H*3 bytes left eye (RGB uint8, top-to-bottom)
 *                 + W*H*3 bytes right eye (RGB uint8, top-to-bottom)
 *
 * One-frame lag: eye cameras sent to Python are from the PREVIOUS frame
 * (same pattern as GaussianLiveView).  Acceptable for VR pre-renders.
 */
#pragma once

// _WIN32_WINNT before boost/asio to suppress the Windows version warning.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

// boost/asio must come before any Windows header to get winsock2 first.
#include <boost/asio.hpp>

// GL/glew.h before any other GL include.
#include <GL/glew.h>

// WIN32_LEAN_AND_MEAN (from boost/asio) strips COM.  Pulling in <unknwn.h>
// directly gives IUnknown for openxr_platform.h if it ends up transitively
// included via SIBR headers.
#include <unknwn.h>

// Old MSVC gl/gl.h defines 'near' and 'far' as empty macros for 16-bit
// pointer qualifiers.  Undefine before SIBR headers which use them as names.
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

#include <core/view/ViewBase.hpp>
#include <core/graphics/Shader.hpp>

#include <string>
#include <vector>
#include <array>

namespace sibr
{

class PixelLiveView : public sibr::ViewBase
{
    SIBR_CLASS_PTR(PixelLiveView);

public:
    PixelLiveView(const std::string& ip, int port,
                  uint render_w, uint render_h);
    ~PixelLiveView() override;

    // Call ONCE per frame in main loop before onRenderIBR.
    // Sends camera JSON, receives left+right RGB images, uploads to GL textures.
    // Returns false on connection failure (retries next call).
    bool fetchFromPython();

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
    void onGUI() override;
    void onUpdate(Input&) override {}
    void setResolution(const sibr::Vector2i& size) override;

private:
    bool        connectTCP();
    void        recreateTextures(uint w, uint h);
    void        initShader();
    std::string buildJson(uint w, uint h) const;

    // TCP
    boost::asio::io_service                        _io;
    std::unique_ptr<boost::asio::ip::tcp::socket>  _socket;
    std::string _ip;
    int         _port;
    bool        _connected = false;

    // Per-eye GL textures: _tex[0] = left, _tex[1] = right
    GLuint               _tex[2] = {0, 0};
    std::vector<uint8_t> _pixBuf; // CPU staging (one eye at a time)

    // Blit shader: texture sampler → full-screen quad
    sibr::GLShader _blitShader;

    // Per-frame bookkeeping
    bool _hasData  = false;
    int  _eyeIdx   = 0;   // 0 = left, 1 = right; reset by fetchFromPython

    // Previous frame's eye cameras, stored so fetchFromPython can send them.
    // (onRenderIBR is called AFTER fetchFromPython, so we have 1-frame lag.)
    bool _hasPrevEye = false;
    std::array<float, 16> _prevViewL = {}, _prevProjL = {};
    std::array<float, 16> _prevViewR = {}, _prevProjR = {};
    float _prevFovXL = 1.2f, _prevFovYL = 1.2f;
    float _prevFovXR = 1.2f, _prevFovYR = 1.2f;
};

} // namespace sibr
