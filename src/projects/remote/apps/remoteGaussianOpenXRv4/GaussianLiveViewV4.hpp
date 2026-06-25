/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveViewV4  (v4.0)
 *
 * Differences from v3 GaussianLiveView:
 *   - Startup handshake: sends b"V400", receives K (feature dim) + TorchScript model.
 *   - Per-frame receive: xyz, features [N×K], R_bwd [N×9], opacity, scale, rot.
 *     No pre-computed colors are sent by Python.
 *   - Per-eye render: runs the TorchScript Color MLP with the actual OpenXR cam center,
 *     then passes the resulting [N×3] colors to CudaRasterizer as colors_precomp.
 *
 * Call once per frame:   view->fetchFromPython();
 * Called per eye by OpenXRRdrMode: view->onRenderIBR(dst, eyeCam);
 */
#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

// boost/asio before any Windows header (winsock2 vs winsock ordering)
#include <boost/asio.hpp>

// GL before cuda_gl_interop (glew must precede gl.h)
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <unknwn.h>

// Legacy gl/gl.h sometimes defines 'near' and 'far' as macros
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif

// LibTorch — must come after cuda_runtime.h
#include <torch/script.h>

#include <core/view/ViewBase.hpp>
#include <core/graphics/Shader.hpp>
#include <core/graphics/Texture.hpp>

#include <string>
#include <vector>
#include <functional>

namespace sibr
{

class GaussianLiveViewV4 : public sibr::ViewBase
{
    SIBR_CLASS_PTR(GaussianLiveViewV4);

public:
    GaussianLiveViewV4(const std::string& ip, int port,
                       uint render_w, uint render_h,
                       bool white_bg = false, int device = 0);
    ~GaussianLiveViewV4() override;

    // Call ONCE per frame before onRender().
    // Sends frame request; receives N + xyz/feat/R_bwd/opacity/scale/rot.
    bool fetchFromPython();

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
    void onGUI() override;
    void onUpdate(Input&) override {}
    void setResolution(const sibr::Vector2i& size) override;

private:
    bool connectTCP();
    bool handshakeWithPython();      // sends V400, receives K + TorchScript model
    void recreateImageBuffer(uint w, uint h);
    void resizeGaussianBuffers(int N);
    void initShader();

    // TCP
    boost::asio::io_service                        _io;
    std::unique_ptr<boost::asio::ip::tcp::socket>  _socket;
    std::string _ip;
    int         _port;
    bool        _connected   = false;
    bool        _handshakeDone = false;

    // LibTorch Color MLP (loaded once at handshake)
    torch::jit::script::Module _colorMLP;
    bool _mlpLoaded = false;

    // Persistent output tensor from the MLP — kept alive until next eye render
    // so the rasterizer kernel (async on the same stream) can finish reading.
    at::Tensor _colors_t;

    // Gaussian GPU buffers
    float* _pos_cuda     = nullptr;  // [N×3]
    float* _feat_cuda    = nullptr;  // [N×K]  view-independent features
    float* _rbwd_cuda    = nullptr;  // [N×9]  backward rotations (flat 3×3)
    float* _opacity_cuda = nullptr;  // [N]
    float* _scale_cuda   = nullptr;  // [N×3]
    float* _rot_cuda     = nullptr;  // [N×4]
    int    _N            = 0;
    int    _allocN       = 0;
    int    _K            = 0;        // view-independent feature dim

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

    // CUDA-GL interop for output image
    GLuint                  _glBuffer     = 0;
    cudaGraphicsResource_t  _cudaBuffer   = nullptr;
    float*                  _fallbackCuda = nullptr;
    std::vector<float>      _fallbackBytes;
    bool _interopFailed = false;
    bool _useInterop    = true;

    // Copy shader (float SSBO -> render target)
    sibr::GLShader        _copyShader;
    sibr::GLuniform<bool> _flipU   { true };
    sibr::GLuniform<int>  _widthU  { 1    };
    sibr::GLuniform<int>  _heightU { 1    };

    bool _white_bg = false;
    int  _device   = 0;
    bool _hasData  = false;

    // Last known eye position (sent to Python as ping context; Python ignores it in v4
    // but keeping it lets us reuse the same 13-byte frame request as v3).
    sibr::Vector3f _lastEyePosition = sibr::Vector3f(0.f, 0.f, 0.f);
};

} // namespace sibr
