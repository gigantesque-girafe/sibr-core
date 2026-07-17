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

// ac: number of command line arguments
// av: pointer of arguments
int main(int ac, char** av)
{
    CommandLineArgs::parseMainArgs(ac, av);

    Arg<std::string> argIp  ("ip",     "127.0.0.1");
    Arg<int>         argPort("port",   6012);
    Arg<uint>        argW   ("width",  1824);
    Arg<uint>        argH   ("height", 1968);

    std::string ip   = argIp.get();
    int         port = argPort.get();
    uint        w    = argW.get();
    uint        h    = argH.get();

    sibr::Window window(w, h, PROGRAM_NAME);

    // create the GS viewer (see file)
    auto gaussianView = std::make_shared<GaussianLiveViewV42>(ip, port, w, h);

    // Creates the SIBR manager that has many subview (fps counter, different options)
    MultiViewManager multiViewManager(window, false);
    
    //Creates the VR renderer
    auto openxrMode = std::make_shared<OpenXRRdrMode>(window);

    //init seat offset
    static sibr::Vector3f seatOffset(0.f, -1.0f, -2.5f);
    static const sibr::Vector3f kSeatDefault = seatOffset;
    static const float kSeatStep = 0.2f;

    // creates a camera and put at current seating position
    MultiViewManager::IBRViewUpdateFunc fixedCam =
        [](sibr::ViewBase::Ptr&, sibr::Input&, const sibr::Viewport&, const float) {
            sibr::InputCamera cam;
            cam.position(seatOffset);
            return cam;
        };
    
    // register the view in one sub-view window, choose rendering mode
    multiViewManager.addIBRSubView(
        "3DGS OpenXR v4.2", gaussianView, fixedCam,
        sibr::Vector2u(w, h),
        ImGuiWindowFlags_NoBringToFrontOnFocus  // disable bringin the window to front when clickin on it (since it is bigger thqn others)
    );
    multiViewManager.renderingMode(openxrMode);


    SIBR_LOG << "[V42] Seat offset keyboard control: A/D=X  W/S=Z  Q/E=Y  R=reset. "
             << "Start = (" << seatOffset.x() << ", " << seatOffset.y() << ", " << seatOffset.z()
             << "). Watch the desktop mirror to dial the avatar into view." << std::endl;

    // THE MAIN LOOP
    while (window.isOpened())
    {
        sibr::Input::poll();
        window.makeContextCurrent();

        //press esc = close windows
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
    }

    return EXIT_SUCCESS;
}
