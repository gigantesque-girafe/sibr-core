/*
 * NEW FILE — does not modify any existing SIBR source.
 *
 * GaussianLiveViewV42  (v4.2, double-buffered pipeline)
 *
 * Differences from v4.0 (GaussianLiveViewV4):
 *   - No per-frame TCP bulk transfer. Gaussian attributes live in two shared CUDA IPC
 *     buffers pre-allocated by Python at startup (double-buffered for pipelining).
 *   - Startup handshake: sends b"V42E", receives N_max + K + TorchScript model + IPC
 *     handles for both buffers.
 *   - Per-frame (pipelined, not ping-pong): Python free-runs, alternately writing into
 *     buffer 0 / buffer 1 and announcing each over TCP (frame_id, buf_idx, N_live) without
 *     waiting for a reply. A dedicated network thread receives these announcements and
 *     atomically publishes the latest one. The render thread's beginFrame() snapshots that
 *     atomic state once per frame (never blocks on Python), then onRenderIBR() (called once
 *     per eye) syncs on that buffer's data-ready event (near-instant — already recorded by
 *     the time the announcement arrived) and runs Color MLP + CudaRasterizer directly on
 *     its IPC pointers. After both eyes are done, the render thread records that buffer's
 *     read-complete event so Python knows it's safe to overwrite on a future frame.
 *   - This lets C++ render frame N while Python deforms frame N+1, instead of strict
 *     alternating turns on a single buffer.
 *   - No cudaMalloc per-Gaussian buffers in C++; all attribute storage is in Python's IPC buffers.
 *
 * IPC buffer layout (float32, base pointer = ipc_base + ipc_offset), same for each of the 2 buffers.
 * Wire v2 (handshake magic "V42E"): the tail is a precomputed 3D covariance
 * instead of scale+rot, so the rasterizer runs its cov3D_precomp path and the
 * splats match the Python reference renderer (render.py, compute_cov3D_python=True):
 *   [N_max * 3]      xyz
 *   [N_max * K]      feat    (view-independent features)
 *   [N_max * 9]      R_bwd   (backward rotation, row-major 3×3)
 *   [N_max * 1]      opacity
 *   [N_max * 6]      cov3D   (posed covariance upper triangle [00,01,02,11,12,22])
 *
 * Call once per frame: view->beginFrame();
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

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sibr
{

class GaussianLiveViewV42 : public sibr::ViewBase
{
    SIBR_CLASS_PTR(GaussianLiveViewV42);

public:
    // views_per_frame = how many onRenderIBR() calls make up one Python frame.
    // 2 for OpenXR stereo (left+right eye) — the default, so the v4.2 app is
    // unaffected. 1 for a mono desktop window. The read-complete event that
    // releases the IPC buffer back to Python is recorded after this many calls.
    GaussianLiveViewV42(const std::string& ip, int port,
                        uint render_w, uint render_h,
                        bool white_bg = false, int device = 0,
                        int views_per_frame = 2);
    ~GaussianLiveViewV42() override;

    // Call ONCE per frame before onRender(). Non-blocking: snapshots whichever
    // buffer the network thread most recently marked ready. Never waits on Python.
    bool beginFrame();

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
    void onGUI() override;
    void onUpdate(Input&) override {}
    void setResolution(const sibr::Vector2i& size) override;

    // ── Animation pause / frame step ─────────────────────────────────────────
    // Python owns the animation frame counter, so these only send a request; the
    // body pose freezes on the pose currently on screen while the camera stays
    // live, which is the point — you can orbit a frozen pose to inspect an error.
    // Public so each app's main() can bind them to keys (the GUI panel is not
    // readable inside a headset). See vr_viewer/server.py for the pause loop.
    void togglePause();
    void setPaused(bool paused);
    void stepFrame(int delta);   // +1 / -1 animation frames; implies pause
    bool isPaused() const { return _uiPaused; }

private:
    bool connectTCP();
    bool handshakeWithPython();   // sends V42E, receives IPC metadata + model for both buffers
    // client->server control msg: 4-byte magic + int32 payload. Returns false if
    // not connected or the send threw. "CTL0" identity / "CTL1" pause / "CTL2" step.
    bool sendControl(const char* magic, int payload);
    void sendIdentity(int id);
    void recreateImageBuffer(uint w, uint h);
    void initShader();
    void startNetworkThread();
    void stopNetworkThread();
    void networkThreadFunc();     // runs on _networkThread; only touches atomics, no CUDA/GL

    // TCP
    boost::asio::io_service                        _io;
    std::unique_ptr<boost::asio::ip::tcp::socket>  _socket;
    std::string _ip;
    int         _port;
    bool        _connected     = false;
    bool        _handshakeDone = false;

    // Guards writes to _socket from the GUI/render thread (sendIdentity) against
    // the network thread's concurrent blocking recv. TCP is full-duplex so send
    // and recv don't conflict, but the mutex serialises rapid button clicks.
    std::mutex  _sendMutex;
    int         _uiIdentity    = 0;   // GUI: currently selected identity index
    bool        _uiPaused      = false;   // GUI: mirrors the pause state last sent to Python

    // LibTorch Color MLP (same TorchScript model as v4.0)
    torch::jit::script::Module _colorMLP;
    bool _mlpLoaded = false;

    // MLP output — kept alive until rasterizer kernel completes (same stream)
    at::Tensor _colors_t;

    // ── CUDA IPC double-buffered shared buffers ───────────────────────────────
    struct IpcBuffer
    {
        void*       base            = nullptr;  // base ptr from cudaIpcOpenMemHandle
        float*      ptr             = nullptr;  // base + ipc_offset (start of attribute data)
        cudaEvent_t dataReadyEvt    = nullptr;  // Python -> C++: write complete
        cudaEvent_t readCompleteEvt = nullptr;  // C++ -> Python: both eyes done reading

        float* xyzPtr   = nullptr;
        float* featPtr  = nullptr;
        float* rbwdPtr  = nullptr;
        float* opaPtr   = nullptr;
        float* covPtr   = nullptr;   // cov3D_precomp (6 floats/Gaussian)
    };
    IpcBuffer _buffers[2];

    int _N_max = 0;   // IPC buffer capacity (Gaussians), same for both buffers
    int _K     = 0;   // view-independent feature dim

    // ── Network thread: receives Python's per-frame announcements, decoupled
    // from rendering. Only writes these atomics — never touches CUDA/GL. ─────
    std::thread        _networkThread;
    std::atomic<bool>  _networkRunning{false};
    std::atomic<int>       _latestReadyBuf{-1};
    std::atomic<int>       _latestNLive{0};
    std::atomic<uint64_t>  _latestFrameId{0};

    // Per-frame snapshot, taken once by beginFrame() and used by both eyes'
    // onRenderIBR() calls for this frame (must not change mid-frame).
    int      _currentBufIdx  = -1;
    int      _currentNLive   = 0;
    uint64_t _currentFrameId = 0;

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
    // OFF by default (a previous run saw a black image with interop on: the
    // rasterizer's writes didn't land in the GL buffer). The host-copy fallback
    // (cudaMemcpy D2H -> glNamedBufferSubData) is slower but correct. Set env
    // V42_INTEROP=1 to attempt zero-copy interop; recreateImageBuffer() then logs
    // the GL/CUDA device mapping and the real register error so it can be
    // diagnosed rather than silently falling back.
    bool _useInterop    = false;

    // Copy shader (float SSBO → render target)
    sibr::GLShader        _copyShader;
    sibr::GLuniform<bool> _flipU   { true };
    sibr::GLuniform<int>  _widthU  { 1    };
    sibr::GLuniform<int>  _heightU { 1    };

    bool _white_bg = false;
    int  _device   = 0;
    bool _hasData  = false;

    // onRenderIBR() calls per Python frame: 2 = stereo (OpenXR), 1 = mono (desktop).
    int  _viewsPerFrame = 2;

    // ── Mono/desktop render-resolution downscale ─────────────────────────────
    // Under OpenXR the resolution is not ours to set: OpenXRRdrMode calls
    // setResolution(headsetRes / itsOwnDownscale) every frame and shows its own
    // "Down scale factor" slider, so the GUI below exposes this for the mono
    // desktop app only (_viewsPerFrame == 1) — otherwise the two would fight.
    // _nativeRes is the constructor's full resolution, the divisor's numerator.
    sibr::Vector2i _nativeRes    { 1, 1 };
    int            _uiDownscale  = 1;

    // ── Per-stage timing (logged every kLogIntervalFrames frames) ────────────────
    // beginFrame() runs once per frame and should now be near-instant (it no longer
    // blocks on Python — that's the whole point of pipelining). onRenderIBR() runs
    // twice per frame (once per eye) — Color MLP + rasterizer time are accumulated
    // across both eyes and reported as a per-frame total.
    static constexpr int kLogIntervalFrames = 60;
    double _beginFrameMsThisFrame = 0.0;
    double _mlpMsAccum            = 0.0;
    double _rasterMsAccum         = 0.0;
    int    _eyeCallsThisFrame     = 0;
    int    _frameCount            = 0;

    // Per-stage GPU timing via CUDA events. OFF by default so the hot path
    // never issues a cudaDeviceSynchronize (which would drain the whole context
    // and serialize the two eyes / re-couple with Python's work). Enable by
    // setting env var V42_TIMING=1 to measure mlp/raster ms with events instead.
    bool        _timingEnabled   = false;
    cudaEvent_t _evtMlpStart      = nullptr;
    cudaEvent_t _evtMlpStop       = nullptr;
    cudaEvent_t _evtRasterStart   = nullptr;
    cudaEvent_t _evtRasterStop    = nullptr;
};

} // namespace sibr
