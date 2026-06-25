/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveView: receives Gaussian attributes from Python via TCP,
 * rasterizes locally with CudaRasterizer for any given eye camera.
 *
 * Call once per frame:   view->fetchFromPython();
 * Called per eye by OpenXRRdrMode: view->onRenderIBR(dst, eyeCam);
 */
#pragma once

// _WIN32_WINNT must be defined before boost/asio to suppress the
// "Please define _WIN32_WINNT" warning. 0x0601 = Windows 7+.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

// boost/asio.hpp must come before any Windows header to get winsock2
// before winsock (classic Windows socket include-order problem).
#include <boost/asio.hpp>

// GL/glew.h must come before cuda_gl_interop.h: cuda_gl_interop.h pulls
// in <GL/gl.h>, and glew.h fatals if gl.h was included first.
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// WIN32_LEAN_AND_MEAN (set by boost/asio) strips COM from windows.h.
// Include unknwn.h directly so IUnknown is available for openxr_platform.h.
#include <unknwn.h>

// Old MSVC gl/gl.h (pulled in by cuda_gl_interop.h on some toolchains)
// defines 'near' and 'far' as empty macros — legacy 16-bit pointer qualifiers.
// SIBR's InputCamera.hpp uses them as struct member names; undefine before
// any SIBR header is processed.
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

#include <core/view/ViewBase.hpp>
#include <core/graphics/Shader.hpp>
#include <core/graphics/Texture.hpp>

#include <string>
#include <vector>
#include <functional>

namespace sibr
{

class GaussianLiveView : public sibr::ViewBase
{
    SIBR_CLASS_PTR(GaussianLiveView);

public:
    GaussianLiveView(const std::string& ip, int port,
                     uint render_w, uint render_h,
                     bool white_bg = false, int device = 0);
    ~GaussianLiveView() override;

    // Call ONCE per frame in main loop before onRender().
    // Sends 1-byte ping to Python, receives N + xyz/colors/opacity/scale/rot.
    // Returns false on connection failure (retries next call).
    bool fetchFromPython();

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
    void onGUI() override;
    void onUpdate(Input&) override {}
    void setResolution(const sibr::Vector2i& size) override;

private:
    bool connectTCP();
    void recreateImageBuffer(uint w, uint h);
    void resizeGaussianBuffers(int N);
    void initShader();

    // TCP
    boost::asio::io_service                       _io;
    std::unique_ptr<boost::asio::ip::tcp::socket> _socket;
    std::string _ip;
    int         _port;
    bool        _connected = false;

    // Gaussian GPU buffers
    float* _pos_cuda     = nullptr;  // [N*3]
    float* _colors_cuda  = nullptr;  // [N*3] precomputed RGB
    float* _opacity_cuda = nullptr;  // [N]
    float* _scale_cuda   = nullptr;  // [N*3]
    float* _rot_cuda     = nullptr;  // [N*4]
    int    _N            = 0;
    int    _allocN       = 0;

    // Per-draw GPU matrices
    float* _view_cuda   = nullptr;
    float* _proj_cuda   = nullptr;
    float* _camPos_cuda = nullptr;
    float* _bg_cuda     = nullptr;

    // Rasterizer resizable scratch
    char*  _geomPtr = nullptr; size_t _allocGeom = 0;
    char*  _binPtr  = nullptr; size_t _allocBin  = 0;
    char*  _imgPtr  = nullptr; size_t _allocImg  = 0;
    std::function<char*(size_t)> _geomBuf, _binBuf, _imgBuf;

    // CUDA-GL interop
    GLuint                 _glBuffer     = 0;
    cudaGraphicsResource_t _cudaBuffer   = nullptr;
    float*                 _fallbackCuda = nullptr;
    std::vector<float>     _fallbackBytes;
    bool _interopFailed = false;
    bool _useInterop    = true;

    // Copy shader (float SSBO -> render target), source embedded in .cpp
    sibr::GLShader        _copyShader;
    sibr::GLuniform<bool> _flipU   { true };
    sibr::GLuniform<int>  _widthU  { 1    };
    sibr::GLuniform<int>  _heightU { 1    };

    bool _white_bg = false;
    int  _device   = 0;
    bool _hasData  = false;

    // Last known eye position, sent to Python with each frame request so the
    // Color MLP can compute the correct view-dependent direction.
    // Updated at the end of every onRenderIBR call (one-frame lag is acceptable).
    sibr::Vector3f _lastEyePosition = sibr::Vector3f(0.f, 0.f, 0.f);
};

} // namespace sibr
