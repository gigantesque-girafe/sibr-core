/*
 * NEW FILE — SIBR_remoteGaussianOpenXRv4_app  (v4.0)
 *
 * Architecture:
 *   Python server (render_vr_v4.py) runs deformation + view-independent feature
 *   extraction, then sends Gaussian geometry + features over TCP.
 *   This C++ client loads the TorchScript Color MLP at startup, runs it per-eye
 *   with the actual OpenXR head pose, and rasterizes with CudaRasterizer.
 *
 * Per-frame flow:
 *   main loop:
 *     gaussianView->fetchFromPython()    <- TCP: ~1 MB (no pixel data)
 *     multiViewManager.onRender()
 *       └─ OpenXRRdrMode calls onRenderIBR(left eye)
 *            └─ _colorMLP.forward({feat, xyz, cam_L, R_bwd}) → colors
 *            └─ CudaRasterizer::forward() → left framebuffer
 *       └─ OpenXRRdrMode calls onRenderIBR(right eye)
 *            └─ _colorMLP.forward({feat, xyz, cam_R, R_bwd}) → colors
 *            └─ CudaRasterizer::forward() → right framebuffer
 *
 * Run:
 *   # Terminal 1
 *   python render_vr_v4.py mode=predict ... load_ckpt=...
 *
 *   # Terminal 2
 *   SIBR_remoteGaussianOpenXRv4_app.exe --ip 127.0.0.1 --port 6011
 */

// GaussianLiveViewV4.hpp first: pulls in boost/asio → winsock2 before windows.h
#include "GaussianLiveViewV4.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>

#define PROGRAM_NAME "3DGS-Avatar OpenXR Viewer v4.0"
using namespace sibr;

int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6011);
    Arg<uint>        argW   ("width",  960);
    Arg<uint>        argH   ("height", 1080);

    std::string ip   = argIp.get();
    int         port = argPort.get();
    uint        w    = argW.get();
    uint        h    = argH.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    auto gaussianView = std::make_shared<GaussianLiveViewV4>(ip, port, w, h);

    MultiViewManager multiViewManager(window, false);

    auto openxrMode = std::make_shared<OpenXRRdrMode>(window);
    multiViewManager.addIBRSubView(
        "3DGS OpenXR v4", gaussianView,
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

        // Fetch once per frame; per-eye Color MLP runs inside onRenderIBR.
        gaussianView->fetchFromPython();

        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);

        window.swapBuffer();
        CHECK_GL_ERROR;
    }

    return EXIT_SUCCESS;
}
