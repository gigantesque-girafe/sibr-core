/*
 * NEW FILE — SIBR_remoteGaussianOpenXR_app
 *
 * OpenXR stereo VR viewer for 3DGS-Avatar.
 * Python (render_gaussians_vr.py) runs deformation + color MLP and sends
 * Gaussian attributes once per frame.  C++ rasterizes both eyes locally.
 *
 * Per-frame flow:
 *   main loop: gaussianView->fetchFromPython()   <- 1 TCP round-trip, ~1 MB
 *              multiViewManager.onRender()
 *                └─ OpenXRRdrMode calls onRenderIBR for left eye
 *                └─ OpenXRRdrMode calls onRenderIBR for right eye
 *                      each: CudaRasterizer::forward() + blit to swapchain
 *
 * Run:
 *   # Terminal 1
 *   python render_gaussians_vr.py mode=predict ...
 *
 *   # Terminal 2 (put on Quest first)
 *   SIBR_remoteGaussianOpenXR_app.exe --ip 127.0.0.1 --port 6010
 */

// GaussianLiveView.hpp must come first: it includes <boost/asio.hpp> which
// pulls in <winsock2.h>.  If any SIBR header comes first it drags in
// <windows.h> → <winsock.h>, and Boost.Asio then fatals with
// "WinSock.h has already been included".
#include "GaussianLiveView.hpp"

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>

#define PROGRAM_NAME "3DGS-Avatar OpenXR Viewer"
using namespace sibr;

int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    // Custom args: --ip, --port, --width, --height
    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6010);
    Arg<uint>        argW   ("width",  960);
    Arg<uint>        argH   ("height", 1080);

    std::string ip   = argIp.get();
    int         port = argPort.get();
    uint        w    = argW.get();
    uint        h    = argH.get();

    // ── SIBR window (hidden is fine — headset is the real display) ────────────
    // Window(uint w, uint h, const std::string& title, const WindowArgs& args = {})
    sibr::Window window(w, h, PROGRAM_NAME);

    // ── GaussianLiveView: TCP + CudaRasterizer ────────────────────────────────
    auto gaussianView = std::make_shared<GaussianLiveView>(ip, port, w, h);

    // ── MultiViewManager + OpenXR rendering mode ──────────────────────────────
    MultiViewManager multiViewManager(window, false);

    auto openxrMode = std::make_shared<OpenXRRdrMode>(window);
    multiViewManager.addIBRSubView(
        "3DGS OpenXR", gaussianView,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus
    );
    multiViewManager.renderingMode(openxrMode);

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (window.isOpened())
    {
        sibr::Input::poll();
        window.makeContextCurrent();

        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();

        // Fetch Gaussian attributes ONCE per frame (before both-eye render)
        gaussianView->fetchFromPython();

        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);

        window.swapBuffer();
        CHECK_GL_ERROR;
    }

    return EXIT_SUCCESS;
}
