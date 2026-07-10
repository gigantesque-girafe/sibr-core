/*
 * NEW FILE — SIBR_remoteGaussianOpenXRv4_2_app  (v4.2, double-buffered pipeline)
 *
 * Architecture:
 *   Python server (render_vr_v4_2.py) pre-allocates two CUDA IPC buffers and free-runs:
 *   deform -> extract features -> write into buffer[i%2] -> announce (frame_id, buf_idx,
 *   N_live) over TCP, without waiting for a reply. A background network thread in this
 *   process receives those announcements and atomically publishes the latest one, fully
 *   decoupled from rendering — so Python can be deforming frame N+1 while this app is
 *   still rendering frame N.
 *
 * Per-frame flow:
 *   main loop:
 *     gaussianView->beginFrame()      <- non-blocking: snapshots latest announced buffer
 *     multiViewManager.onRender()
 *       └─ OpenXRRdrMode calls onRenderIBR(left eye)
 *            └─ cudaEventSynchronize(dataReadyEvt)   [instant: event already recorded]
 *            └─ _colorMLP.forward({feat, xyz, cam_L, R_bwd}) → colors
 *            └─ CudaRasterizer::forward(xyz, ..., opa, scale, rot)
 *       └─ OpenXRRdrMode calls onRenderIBR(right eye)
 *            [same buffer; after this eye, records readCompleteEvt for Python]
 *
 * Run:
 *   # Terminal 1
 *   python render_vr_v4_2.py mode=predict ... load_ckpt=...
 *
 *   # Terminal 2
 *   SIBR_remoteGaussianOpenXRv4_2_app.exe --ip 127.0.0.1 --port 6012
 */

#include "GaussianLiveViewV42.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>

#include <chrono>
#include <vector>
#include <iomanip>
#include <ctime>

#define PROGRAM_NAME "3DGS-Avatar OpenXR Viewer v4.2"
using namespace sibr;

int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6012);
    Arg<uint>        argW   ("width",  960);
    Arg<uint>        argH   ("height", 1080);

    std::string ip   = argIp.get();
    int         port = argPort.get();
    uint        w    = argW.get();
    uint        h    = argH.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    auto gaussianView = std::make_shared<GaussianLiveViewV42>(ip, port, w, h);

    MultiViewManager multiViewManager(window, false);

    auto openxrMode = std::make_shared<OpenXRRdrMode>(window);
    // OpenXRRdrMode's "seated" mode (default) anchors the headset's tracked
    // pose by translating it by this camera's position each frame. Under the
    // Meta/Oculus runtime the reference space is STAGE (floor origin) and, after
    // the renderer's Y/Z flip, the eye camera looks toward +Z — so the avatar
    // (recentered to the world origin in render_vr_v4_2.py) must sit at NEGATIVE
    // Z to be in front of the viewer. The old +2.5 put it behind the camera.
    //
    // seatOffset is now adjustable live from the keyboard so it can be dialed in
    // by watching the desktop mirror window (no headset needed):
    //   A/D : move viewer -X / +X        W/S : move viewer -Z / +Z (closer/farther)
    //   Q/E : move viewer -Y / +Y (down/up)   R : reset to default
    // The current value is printed to the console whenever it changes.
    // Under STAGE (floor origin) the eye sits at Y ~= +1.05 (head height); the
    // renderer flips Y, so seatOffset.y must be ~ +1.05 to bring the camera back
    // to the avatar's mid-height. Z is negative so the avatar (world origin) is
    // in front. Fine-tune live with the keys below.
    static sibr::Vector3f seatOffset(0.f, 1.05f, -2.5f);
    static const sibr::Vector3f kSeatDefault = seatOffset;
    static const float kSeatStep = 0.2f;

    MultiViewManager::IBRViewUpdateFunc fixedCam =
        [](sibr::ViewBase::Ptr&, sibr::Input&, const sibr::Viewport&, const float) {
            sibr::InputCamera cam;
            cam.position(seatOffset);
            return cam;
        };
    multiViewManager.addIBRSubView(
        "3DGS OpenXR v4.2", gaussianView, fixedCam,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus
    );
    multiViewManager.renderingMode(openxrMode);

    SIBR_LOG << "[V42] Seat offset keyboard control: A/D=X  W/S=Z  Q/E=Y  R=reset. "
             << "Start = (" << seatOffset.x() << ", " << seatOffset.y() << ", " << seatOffset.z()
             << "). Watch the desktop mirror to dial the avatar into view." << std::endl;

    while (window.isOpened())
    {
        // Full render-loop wall time — this is what the on-screen FPS counter
        // reports. It equals waitNextFrame(throttle) + submitFrame(work) +
        // beginFrame + swapBuffer + input/UI. Compare against the [OpenXR] and
        // [V42] render_total lines to see where the frame time actually goes.
        auto _loop0 = std::chrono::steady_clock::now();

        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

        // Live seat-offset tuning (discrete step per key press). Watch the
        // desktop mirror window: it shows exactly what each eye sees, so the
        // avatar can be framed without wearing the headset.
        {
            auto& kb = sibr::Input::global().key();
            sibr::Vector3f before = seatOffset;
            if (kb.isPressed(sibr::Key::D)) seatOffset.x() += kSeatStep;
            if (kb.isPressed(sibr::Key::A)) seatOffset.x() -= kSeatStep;
            if (kb.isPressed(sibr::Key::E)) seatOffset.y() += kSeatStep;
            if (kb.isPressed(sibr::Key::Q)) seatOffset.y() -= kSeatStep;
            if (kb.isPressed(sibr::Key::S)) seatOffset.z() += kSeatStep;
            if (kb.isPressed(sibr::Key::W)) seatOffset.z() -= kSeatStep;
            if (kb.isPressed(sibr::Key::R)) seatOffset = kSeatDefault;
            if (seatOffset != before)
            {
                SIBR_LOG << "[V42] seatOffset = (" << seatOffset.x() << ", "
                         << seatOffset.y() << ", " << seatOffset.z() << ")" << std::endl;
            }
        }

        // Non-blocking: snapshots whichever buffer Python's network thread most
        // recently announced. Per-eye Color MLP + rasterizer run inside onRenderIBR.
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
                SIBR_LOG << "[V42] main-loop frame " << _fc << "  loop=" << _loop_ms
                         << "ms  (== on-screen frame time; " << (1000.0 / _loop_ms)
                         << " FPS)" << std::endl;
        }
    }

    return EXIT_SUCCESS;
}
