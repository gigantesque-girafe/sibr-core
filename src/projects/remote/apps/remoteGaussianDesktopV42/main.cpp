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
 *   - InteractiveCameraHandler, started in TRACKBALL mode (its own default is FPS,
 *     i.e. WASD/IJKL flying — see switchMode() below). Mouse mapping, all from
 *     core/view/TrackBall.cpp, applies to whichever tile the mouse is over:
 *         left-drag  (centre 75%)  orbit around the pivot
 *         left-drag  (outer band)  roll
 *         right-drag (centre 75%)  pan
 *         right-drag (outer band)  dolly in/out
 *         scroll                   zoom (keeps the pivot; no key may be held)
 *         Y                        toggle back to FPS/WASD, Y again to return
 *     The pivot is planted at eye + dir*radius, so the radius MUST be set before
 *     setup() or it defaults to 100 and you orbit a point far behind the avatar.
 *   - The "Camera ..." GUI panel (mode dropdown, FoV, near/far, save/load camera)
 *     appears because the handler is registered via addCameraForView() — that is
 *     also what makes MultiViewManager drive it and draw the trackball gizmo.
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
 *
 * Split-screen A/B comparison (e.g. MIGS vs. MISTA): pass --ip2/--port2 for a
 * second Python server and this app opens a second synchronized subview.
 * Both tiles share ONE InteractiveCameraHandler, so orbiting either tile moves
 * the camera in both — same viewpoint, same frame, one screenshot to compare.
 * FPS is not a concern here (this mode exists for qualitative comparison, not
 * live perf), so also bump --width/--height and pass --label1/--label2:
 *
 *   # Terminal 1 — model A (e.g. MIGS), its own port
 *   python render_vr_v1_modular.py mode=predict dataset=migs migs.type=cp \
 *       appearance_identity=2 load_ckpt="<ckpt_migs>.pth" wandb_disable=True \
 *       gaussians_vr.port=6012
 *
 *   # Terminal 2 — model B (e.g. MISTA), a different port
 *   python render_vr_v1_modular.py mode=predict dataset=zju_377_mono \
 *       load_ckpt="<ckpt_mista>.pth" wandb_disable=True gaussians_vr.port=6013
 *
 *   # Terminal 3 — one window, both tiles
 *   SIBR_remoteGaussianDesktopV42_app_rwdi.exe --port 6012 --port2 6013 \
 *       --width 2560 --height 1080 --label1 MIGS --label2 MISTA
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

    Arg<std::string> argIp    ("ip",     "127.0.0.1");
    Arg<int>         argPort  ("port",   6012);
    Arg<uint>        argW     ("width",  1280);
    Arg<uint>        argH     ("height", 960);
    // Split-screen A/B comparison: leave port2 at 0 to keep today's single-tile
    // behavior unchanged. Set it to open a second synchronized tile fed by a
    // second Python server (see the usage comment above main()).
    Arg<std::string> argIp2    ("ip2",    "127.0.0.1");
    Arg<int>         argPort2  ("port2",  0);
    Arg<std::string> argLabel1 ("label1", "Model A");
    Arg<std::string> argLabel2 ("label2", "Model B");

    const std::string ip          = argIp.get();
    const int         port        = argPort.get();
    const uint        w           = argW.get();
    const uint        h           = argH.get();
    const std::string ip2         = argIp2.get();
    const int         port2       = argPort2.get();
    const bool        splitScreen = (port2 != 0);
    const std::string label1      = argLabel1.get();
    const std::string label2      = argLabel2.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    // views_per_frame = 1: mono, one onRenderIBR() per Python frame. This holds
    // per-view even in split-screen mode: each tile is its own GaussianLiveViewV42
    // fed by its own Python server, so each still gets exactly one onRenderIBR()
    // call per multiViewManager.onRender().
    auto gaussianView = std::make_shared<GaussianLiveViewV42>(
        ip, port, w, h, /*white_bg=*/false, /*device=*/0, /*views_per_frame=*/1);

    std::shared_ptr<GaussianLiveViewV42> gaussianView2;
    if (splitScreen) {
        gaussianView2 = std::make_shared<GaussianLiveViewV42>(
            ip2, port2, w, h, /*white_bg=*/false, /*device=*/0, /*views_per_frame=*/1);
    }

    MultiViewManager multiViewManager(window, false);

    // Initial camera. The avatar is recentered to the world origin by
    // render_vr_v4_2.py, so frame it from a few metres out along +Z looking back
    // at the origin. Unlike the VR app there is no OpenXRRdrMode Y/Z flip in the
    // path here, so the v4.2 seatOffset value does not carry over — if the avatar
    // starts off-screen or upside down, just navigate: this is a live trackball,
    // and the camera handler's GUI can save the pose once it is framed.
    const Viewport viewport(0.f, 0.f, (float)w, (float)h);

    // Distance from the initial eye to the avatar at the world origin. Reused as the
    // trackball radius just below, so the two cannot drift apart.
    constexpr float kEyeDist = 3.f;

    sibr::Camera initCam;
    initCam.setLookAt(sibr::Vector3f(0.f, 0.f, kEyeDist), // eye
                      sibr::Vector3f(0.f, 0.f, 0.f),    // look at avatar (world origin)
                      sibr::Vector3f(0.f, 1.f, 0.f));   // up
    initCam.aspect((float)w / (float)h);
    initCam.znear(0.01f);
    initCam.zfar(100.f);

    auto camHandler = std::make_shared<InteractiveCameraHandler>();
    // Before setup(): setup() -> fromCamera() -> TrackBall::fromCamera(cam, vp, radius)
    // plants the orbit pivot at eye + dir*radius. The handler's default radius is 100
    // and we pass no raycaster (the Gaussians live on the GPU, there is no proxy mesh to
    // intersect), so nothing would correct it — the pivot would land at (0,0,-97) and
    // both orbit and scroll-zoom would be useless. kEyeDist puts it on the avatar.
    camHandler->getRadius() = kEyeDist;
    camHandler->setup(sibr::InputCamera(initCam, (int)w, (int)h), viewport, nullptr);
    // Mouse navigation instead of the FPS default. switchMode() re-syncs the internal
    // cameras through fromCamera(), so it re-reads the radius set above — hence after
    // setup(), not before. Y (or the Camera panel dropdown) toggles back to FPS.
    camHandler->switchMode(sibr::InteractiveCameraHandler::TRACKBALL);

    // Drive the view from the interactive camera each frame. (The v4.2 app uses
    // this same hook to pin a fixed seat pose instead, since the headset supplies
    // the real per-eye poses.)
    //
    // No camHandler->update() here: addCameraForView() below registers the handler with
    // MultiViewManager, which already calls update(subInput, dt, viewport) once per tile
    // per frame. Updating here as well would apply every drag and scroll notch twice.
    MultiViewManager::IBRViewUpdateFunc interactiveCam =
        [camHandler](sibr::ViewBase::Ptr&, sibr::Input&,
                     const sibr::Viewport&, const float) {
            return camHandler->getCamera();
        };

    const std::string view1Name = splitScreen ? label1 : "3DGS Desktop v4.2";

    multiViewManager.addIBRSubView(
        view1Name, gaussianView, interactiveCam,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus
    );
    // Registering the handler is what hands mouse navigation to MultiViewManager: it
    // calls camHandler->update() with this tile's viewport and a mouse position remapped
    // into it, renders the trackball gizmo, and shows the "Camera <name>" panel.
    multiViewManager.addCameraForView(view1Name, camHandler);
    if (splitScreen) {
        // Same camHandler instance as the first tile: InteractiveCameraHandler is
        // shared, not copied (captured by shared_ptr in interactiveCam), so both
        // tiles read/update the exact same camera pose each frame — that is what
        // keeps the two models locked to the same viewpoint while orbiting.
        multiViewManager.addIBRSubView(
            label2, gaussianView2, interactiveCam,
            sibr::Vector2u(w, h),
            ImGuiWindowFlags_NoBringToFrontOnFocus
        );
        // Registered on both tiles so you can drag inside either one. Only the focused
        // tile ever sees real input — MultiViewManager hands an empty Input() to the
        // others and TrackBall::update() early-returns on it — so the two registrations
        // never fight over the shared camera.
        multiViewManager.addCameraForView(label2, camHandler);
    }
    // NOTE: no multiViewManager.renderingMode(...) call — the default mono
    // rendering mode is exactly what we want. That single omission is the whole
    // difference from the OpenXR app.

    if (splitScreen) {
        SIBR_LOG << "[Desktop] Split-screen: \"" << label1 << "\" <- " << ip << ":" << port
                 << "   \"" << label2 << "\" <- " << ip2 << ":" << port2
                 << " — one shared camera, orbit either tile to move both." << std::endl;
    } else {
        SIBR_LOG << "[Desktop] Connecting to Python at " << ip << ":" << port
                 << " — same V42E protocol as the VR app. The identity buttons are in "
                 << "the GaussianLiveViewV42 panel." << std::endl;
    }
    SIBR_LOG << "[Desktop] Camera (trackball): Left-drag=orbit  Right-drag=pan "
             << "(dolly near the border)  Scroll=zoom (no key held)  Y=FPS/WASD mode. "
             << "Hover the rendered image, not the window chrome." << std::endl;
    SIBR_LOG << "[Desktop] Animation: P=pause/resume  Left/Right=step +/-1 frame. "
             << "Only the body pose freezes — the camera stays live, so you can "
             << "orbit a frozen pose." << (splitScreen ? " Applies to both tiles." : "") << std::endl;

    while (window.isOpened())
    {
        auto _loop0 = std::chrono::steady_clock::now();

        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

        // Animation pause / frame step (mirrors the view's ImGui panel). Freezes the
        // body pose only — the trackball stays live, so a frozen pose can be orbited
        // and inspected, which is the point during evaluation.
        //
        // isPressed() is edge-triggered in SIBR (!last && current), so these fire once
        // per press — unlike isActivated(), which is the held/continuous variant.
        //
        // Key choice: the arrows are unused by every SIBR camera handler. P is bound by
        // InteractiveCameraHandler to snapToCamera(-1), but that early-returns on an
        // empty _interpPath and this app passes no camera list (setup(..., nullptr)),
        // so it is inert here. If a camera path is ever loaded, P would also snap the
        // camera and should be rebound.
        // In split-screen mode both models must step in lockstep, otherwise a
        // qualitative comparison is comparing two different poses/frames.
        {
            const auto& keys = sibr::Input::global().key();
            if (keys.isPressed(sibr::Key::P)) {
                gaussianView->togglePause();
                if (splitScreen) gaussianView2->togglePause();
            }
            if (keys.isPressed(sibr::Key::Left)) {
                gaussianView->stepFrame(-1);
                if (splitScreen) gaussianView2->stepFrame(-1);
            }
            if (keys.isPressed(sibr::Key::Right)) {
                gaussianView->stepFrame(+1);
                if (splitScreen) gaussianView2->stepFrame(+1);
            }
        }

        // Non-blocking: snapshots whichever buffer Python's network thread most
        // recently announced. Color MLP + rasterizer run inside onRenderIBR.
        gaussianView->beginFrame();
        if (splitScreen) gaussianView2->beginFrame();

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
