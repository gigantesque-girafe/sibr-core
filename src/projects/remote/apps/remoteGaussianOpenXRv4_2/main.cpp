/*
 * NEW FILE — SIBR_remoteGaussianOpenXRv4_2_app  (v4.2)
 *
 * Architecture:
 *   Python server (render_vr_v4_2.py) pre-allocates a CUDA IPC buffer, runs deformation
 *   and view-independent feature extraction, and writes Gaussian attributes directly into
 *   shared GPU memory — no host-device copy, no TCP bulk transfer.
 *   This C++ client opens the IPC buffer, loads the TorchScript Color MLP once at startup,
 *   and each frame: waits on a CUDA IPC event, runs the MLP per-eye, and rasterizes.
 *
 * Per-frame flow:
 *   main loop:
 *     gaussianView->fetchFromPython()      <- sends 1-byte ping; receives uint32 N_live
 *     multiViewManager.onRender()
 *       └─ OpenXRRdrMode calls onRenderIBR(left eye)
 *            └─ cudaEventSynchronize(ipc_event)        [instant: event already recorded]
 *            └─ _colorMLP.forward({feat, xyz, cam_L, R_bwd}) → colors
 *            └─ CudaRasterizer::forward(ipc_xyz, ..., ipc_opa, ipc_scale, ipc_rot)
 *       └─ OpenXRRdrMode calls onRenderIBR(right eye)
 *            [same, cudaEventSynchronize still complete for this frame]
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
    // pose by translating it (world-space add, see Transform3::translate) by
    // this camera's position each frame. The right height depends on the
    // scene's coordinate convention and avatar placement, which we can't see
    // from here — manual keyboard tuning requires alternating between the
    // headset and the PC, and taking the Quest off puts it to sleep and kills
    // the Steam Link session. So instead: auto-sweep through a range of
    // heights, holding each for a few seconds, no PC interaction needed.
    // Avatar's measured canonical-mesh Y extent is ~1.44 (see bbox probe),
    // so sweep well past that on both sides to be safe.
    static const float kHoldSeconds = 4.f;
    static const float kYMin = -2.0f, kYMax = 2.0f, kYStep = 0.2f;
    static std::vector<float> ySteps = [] {
        std::vector<float> v;
        for (float y = kYMin; y <= kYMax + 1e-4f; y += kYStep) v.push_back(y);
        return v;
    }();
    static sibr::Vector3f seatOffset(0.f, kYMin, 2.5f);
    static auto sweepStart = std::chrono::steady_clock::now();
    static int lastStepIdx = -1;

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

    SIBR_LOG << "[V42] Auto-sweeping seat height Y from " << kYMin << " to " << kYMax
             << " (" << kHoldSeconds << "s per step, " << ySteps.size() << " steps, loops forever)" << std::endl;

    while (window.isOpened())
    {
        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

        // Auto-sweep seat height: no PC/keyboard interaction needed while wearing the headset.
        {
            float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - sweepStart).count();
            int idx = static_cast<int>(elapsed / kHoldSeconds) % static_cast<int>(ySteps.size());
            if (idx != lastStepIdx)
            {
                lastStepIdx = idx;
                seatOffset.y() = ySteps[idx];
                auto now = std::chrono::system_clock::now();
                auto t = std::chrono::system_clock::to_time_t(now);
                SIBR_LOG << "[V42] " << std::put_time(std::localtime(&t), "%H:%M:%S")
                         << " seatOffset.y = " << seatOffset.y() << std::endl;
            }
        }

        // Fetch once per frame; ping triggers Python to pack IPC buffer.
        // Per-eye Color MLP + rasterizer run inside onRenderIBR.
        gaussianView->fetchFromPython();

        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);

        window.swapBuffer();
        CHECK_GL_ERROR;
    }

    return EXIT_SUCCESS;
}
