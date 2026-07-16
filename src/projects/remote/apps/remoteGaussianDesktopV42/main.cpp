/*
 * NEW FILE — SIBR_remoteGaussianDesktopV42_app  (v4.2 transport, mono desktop window)
 *
 * Same architecture as remoteGaussianOpenXRv4_2, minus OpenXR:
 *   - Reuses GaussianLiveViewV42 verbatim (compiled from ../remoteGaussianOpenXRv4_2/).
 *   - Reuses the Python server unchanged: same "V42E" handshake, same CUDA IPC
 *     double-buffered transport, same TorchScript Color MLP blob, same "CTL0"
 *     identity control channel. No Python-side change is needed to run this app.
 *
 * The only difference is the display path. OpenXR lived entirely in the v4.2
 * main() (one OpenXRRdrMode object), never in the view: GaussianLiveViewV42 is a
 * plain sibr::ViewBase that renders into whatever IRenderTarget it is handed.
 * Dropping the rendering mode gives MultiViewManager's default mono rendering.
 *
 * Consequently onRenderIBR() is called ONCE per frame here, not twice. That is
 * what views_per_frame=1 tells the view: it gates when the read-complete event
 * is recorded to hand the IPC buffer slot back to Python. Left at the default of
 * 2 the event would never fire and Python would stall after the first frame.
 *
 * Navigation and UI come for free from SIBR:
 *   - InteractiveCameraHandler gives trackball / FPS navigation (see its own GUI
 *     panel for the mode switch and camera save/load).
 *   - The view's onGUI() panel — live Gaussian count, connection state, and the
 *     appearance-identity buttons that send "CTL0" back to Python — is the same
 *     one the VR app shows on its mirror window.
 *
 * Run (Python first — it listens, this app connects):
 *   # Terminal 1 — the unchanged MISTA adapter, same one the VR app uses
 *   python render_vr_v1_modular.py mode=predict dataset.predict_seq=0 \
 *       dataset=zju_377_mono wandb_disable=True \
 *       load_ckpt="./results/zju_377_mono/ckpt50000_MISTA.pth" appearance_identity=0
 *
 *   # Terminal 2
 *   SIBR_remoteGaussianDesktopV42_app_rwdi.exe --ip 127.0.0.1 --port 6012
 */

#include "GaussianLiveViewV42.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/view/InteractiveCameraHandler.hpp>

#include <chrono>

#define PROGRAM_NAME "3DGS-Avatar Desktop Viewer v4.2"
using namespace sibr;

int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6012);
    Arg<uint>        argW   ("width",  1280);
    Arg<uint>        argH   ("height", 960);

    const std::string ip   = argIp.get();
    const int         port = argPort.get();
    const uint        w    = argW.get();
    const uint        h    = argH.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    // views_per_frame = 1: mono, one onRenderIBR() per Python frame.
    auto gaussianView = std::make_shared<GaussianLiveViewV42>(
        ip, port, w, h, /*white_bg=*/false, /*device=*/0, /*views_per_frame=*/1);

    MultiViewManager multiViewManager(window, false);

    // Initial camera. The avatar is recentered to the world origin by
    // render_vr_v4_2.py, so frame it from a few metres out along +Z looking back
    // at the origin. Unlike the VR app there is no OpenXRRdrMode Y/Z flip in the
    // path here, so the v4.2 seatOffset value does not carry over — if the avatar
    // starts off-screen or upside down, just navigate: this is a live trackball,
    // and the camera handler's GUI can save the pose once it is framed.
    const Viewport viewport(0.f, 0.f, (float)w, (float)h);

    sibr::Camera initCam;
    initCam.setLookAt(sibr::Vector3f(0.f, 0.f, 3.f),    // eye
                      sibr::Vector3f(0.f, 0.f, 0.f),    // look at avatar (world origin)
                      sibr::Vector3f(0.f, 1.f, 0.f));   // up
    initCam.aspect((float)w / (float)h);
    initCam.znear(0.01f);
    initCam.zfar(100.f);

    auto camHandler = std::make_shared<InteractiveCameraHandler>();
    camHandler->setup(sibr::InputCamera(initCam, (int)w, (int)h), viewport, nullptr);

    // Drive the view from the interactive camera each frame. (The v4.2 app uses
    // this same hook to pin a fixed seat pose instead, since the headset supplies
    // the real per-eye poses.)
    MultiViewManager::IBRViewUpdateFunc interactiveCam =
        [camHandler](sibr::ViewBase::Ptr&, sibr::Input& input,
                     const sibr::Viewport& vp, const float deltaTime) {
            camHandler->update(input, deltaTime, vp);
            return camHandler->getCamera();
        };

    multiViewManager.addIBRSubView(
        "3DGS Desktop v4.2", gaussianView, interactiveCam,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus
    );
    // NOTE: no multiViewManager.renderingMode(...) call — the default mono
    // rendering mode is exactly what we want. That single omission is the whole
    // difference from the OpenXR app.

    SIBR_LOG << "[Desktop] Connecting to Python at " << ip << ":" << port
             << " — same V42E protocol as the VR app. Mouse to navigate; the "
             << "identity buttons are in the GaussianLiveViewV42 panel." << std::endl;

    while (window.isOpened())
    {
        auto _loop0 = std::chrono::steady_clock::now();

        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

        // Non-blocking: snapshots whichever buffer Python's network thread most
        // recently announced. Color MLP + rasterizer run inside onRenderIBR.
        gaussianView->beginFrame();

        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);

        window.swapBuffer();
        CHECK_GL_ERROR;

        auto _loop1 = std::chrono::steady_clock::now();
        {
            double _loop_ms = std::chrono::duration<double, std::milli>(_loop1 - _loop0).count();
            static int _fc = 0;
            if (++_fc % 60 == 0)
                SIBR_LOG << "[Desktop] main-loop frame " << _fc << "  loop=" << _loop_ms
                         << "ms  (" << (1000.0 / _loop_ms) << " FPS)" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
