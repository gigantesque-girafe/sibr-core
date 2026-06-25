/*
 * NEW FILE — SIBR_remoteGaussianOpenXRv2_app
 *
 * OpenXR stereo VR viewer for 3DGS-Avatar — pixel-streaming mode (v2).
 * Python runs the full deformation + rasterization pipeline and sends two
 * pre-rendered W×H×3 uint8 images per frame.  C++ just displays them.
 *
 * Per-frame flow:
 *   main loop: pixelView->fetchFromPython()   <- sends JSON, receives 2 images
 *              multiViewManager.onRender()
 *                └─ OpenXRRdrMode calls onRenderIBR for left eye  (blit _tex[0])
 *                └─ OpenXRRdrMode calls onRenderIBR for right eye (blit _tex[1])
 *
 * Run:
 *   # Terminal 1 — start Python server (matches viewer_openxr_app.exe protocol)
 *   python render_vr_v2.py mode=predict ...
 *    --or--
 *   python render_openxr_baseline.py mode=predict ...
 *
 *   # Terminal 2 — launch SIBR viewer (put Quest on first)
 *   SIBR_remoteGaussianOpenXRv2_app.exe --ip 127.0.0.1 --port 6009
 */

// PixelLiveView.hpp first: includes boost/asio which pulls in winsock2.
// Any SIBR header that arrives before it drags in winsock.h and Boost fatals.
#include "PixelLiveView.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>

#define PROGRAM_NAME "3DGS-Avatar OpenXR Pixel Viewer (v2)"
using namespace sibr;

int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6009);
    Arg<uint>        argW   ("width",  960);
    Arg<uint>        argH   ("height", 1080);

    std::string ip   = argIp.get();
    int         port = argPort.get();
    uint        w    = argW.get();
    uint        h    = argH.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    auto pixelView = std::make_shared<PixelLiveView>(ip, port, w, h);

    MultiViewManager multiViewManager(window, false);

    auto openxrMode = std::make_shared<OpenXRRdrMode>(window);
    multiViewManager.addIBRSubView(
        "Pixel Stream OpenXR", pixelView,
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

        // Fetch both eye images ONCE per frame before the two onRenderIBR calls
        pixelView->fetchFromPython();

        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);

        window.swapBuffer();
        CHECK_GL_ERROR;
    }

    return EXIT_SUCCESS;
}
