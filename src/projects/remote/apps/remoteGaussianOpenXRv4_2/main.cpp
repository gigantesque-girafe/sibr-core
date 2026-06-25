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
    multiViewManager.addIBRSubView(
        "3DGS OpenXR v4.2", gaussianView,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus
    );
    multiViewManager.renderingMode(openxrMode);

    while (window.isOpened())
    {
        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

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
