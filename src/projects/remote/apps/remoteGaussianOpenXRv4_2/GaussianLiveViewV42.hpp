/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveViewV42  (v4.2)
 *
 * Differences from v4.0 (GaussianLiveViewV4):
 *   - No per-frame TCP bulk transfer. Gaussian attributes live in a shared CUDA IPC buffer
 *     pre-allocated by Python at startup.
 *   - Startup handshake: sends b"V420", receives N_max + K + TorchScript model + IPC handles.
 *   - Per-frame:  C++ sends 1-byte ping → Python packs buffer + records CUDA event + sends uint32 N_live.
 *                 C++ calls cudaEventSynchronize (instant — TCP ordering guarantees event is
 *                 already recorded), then runs Color MLP + CudaRasterizer directly on IPC pointers.
 *   - No cudaMalloc per-Gaussian buffers in C++; all attribute storage is in Python's IPC buffer.
 *
 * IPC buffer layout (float32, base pointer = ipc_base + ipc_offset):
 *   [N_max * 3]      xyz
 *   [N_max * K]      feat    (view-independent features)
 *   [N_max * 9]      R_bwd   (backward rotation, row-major 3×3)
 *   [N_max * 1]      opacity
 *   [N_max * 3]      scale
 *   [N_max * 4]      rot     (unit quaternion)
 *
 * Call once per frame: view->fetchFromPython();
 * Called per eye by OpenXRRdrMode: view->onRenderIBR(dst, eyeCam);
 */
#pragma once

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif

// boost/asio before any Windows header (winsock2 vs winsock ordering)
#include <boost/asio.hpp>

// GL before cuda_gl_interop
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <unknwn.h>

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

class GaussianLiveViewV42 : public sibr::ViewBase
{
    SIBR_CLASS_PTR(GaussianLiveViewV42);

public:
    GaussianLiveViewV42(const std::string& ip, int port,
                        uint render_w, uint render_h,
                        bool white_bg = false, int device = 0);
    ~GaussianLiveViewV42() override;

    // Call ONCE per frame before onRender().
    // Sends 1-byte ping; receives uint32 N_live (after Python records IPC event).
    bool fetchFromPython();

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
    void onGUI() override;
    void onUpdate(Input&) override {}
    void setResolution(const sibr::Vector2i& size) override;

private:
    bool connectTCP();
    bool handshakeWithPython();   // sends V420, receives IPC metadata + model
    void recreateImageBuffer(uint w, uint h);
    void initShader();

    // TCP
    boost::asio::io_service                        _io;
    std::unique_ptr<boost::asio::ip::tcp::socket>  _socket;
    std::string _ip;
    int         _port;
    bool        _connected     = false;
    bool        _handshakeDone = false;

    // LibTorch Color MLP (same TorchScript model as v4.0)
    torch::jit::script::Module _colorMLP;
    bool _mlpLoaded = false;

    // MLP output — kept alive until rasterizer kernel completes (same stream)
    at::Tensor _colors_t;

    // ── CUDA IPC shared buffer ────────────────────────────────────────────────
    void*        _ipc_base  = nullptr;  // base ptr from cudaIpcOpenMemHandle
    float*       _ipc_ptr   = nullptr;  // _ipc_base + ipc_offset (start of attribute data)
    cudaEvent_t  _ipc_event = nullptr;  // from cudaIpcOpenEventHandle

    int _N_max  = 0;   // IPC buffer capacity (Gaussians)
    int _N_live = 0;   // actual Gaussians in current frame (received via TCP)
    int _K      = 0;   // view-independent feature dim

    // Pointer arithmetic helpers (computed once after handshake)
    float* _ipc_xyz_ptr   = nullptr;  // _ipc_ptr + 0
    float* _ipc_feat_ptr  = nullptr;  // _ipc_ptr + N_max*3
    float* _ipc_rbwd_ptr  = nullptr;  // _ipc_ptr + N_max*(3+K)
    float* _ipc_opa_ptr   = nullptr;  // _ipc_ptr + N_max*(3+K+9)
    float* _ipc_scale_ptr = nullptr;  // _ipc_ptr + N_max*(3+K+10)
    float* _ipc_rot_ptr   = nullptr;  // _ipc_ptr + N_max*(3+K+13)

    // Per-draw small GPU buffers
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

    // Copy shader (float SSBO → render target)
    sibr::GLShader        _copyShader;
    sibr::GLuniform<bool> _flipU   { true };
    sibr::GLuniform<int>  _widthU  { 1    };
    sibr::GLuniform<int>  _heightU { 1    };

    bool _white_bg = false;
    int  _device   = 0;
    bool _hasData  = false;
};

} // namespace sibr
