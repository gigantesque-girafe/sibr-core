# ─────────────────────────────────────────────────────────────────────────────
# NEW FILE — remoteGaussianOpenXRv4/CMakeLists.txt
#
# Builds SIBR_remoteGaussianOpenXRv4_app.
#
# Differences from v3 (remoteGaussianOpenXR):
#   - Links against LibTorch so the C++ side can run the Color MLP inference.
#   - Color MLP weights are received from Python at startup as a TorchScript blob.
#   - Per-frame payload is Gaussian geometry + view-independent features (no pre-computed RGB).
#   - Per-eye view direction is computed from the actual OpenXR tracking pose.
#
# Prerequisites
#   LibTorch (CUDA build) must be findable. Point CMake to it at configure time:
#     cmake ... -DCMAKE_PREFIX_PATH="C:/libtorch"   (or wherever you extracted it)
#   Download: https://pytorch.org/  →  LibTorch  →  CUDA 11.8  →  Release  →  Windows
#
# Build alongside v3 (from the repo root build dir):
#   cmake -S submodules/sibr_core -B build_sibr ^
#       -G "NMake Makefiles" ^
#       -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
#       -DCMAKE_PREFIX_PATH="C:/libtorch;..." ^
#       ...other flags same as v3 build...
#   cmake --build build_sibr --target SIBR_remoteGaussianOpenXRv4_app
# ─────────────────────────────────────────────────────────────────────────────

project(SIBR_remoteGaussianOpenXRv4_app)

# ── LibTorch ──────────────────────────────────────────────────────────────────
# find_package searches CMAKE_PREFIX_PATH for TorchConfig.cmake.
# If not found we print a clear message and skip the target rather than
# breaking the whole build (v3 must still compile without LibTorch).
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    message(STATUS
        "[v4] LibTorch not found — skipping ${PROJECT_NAME}.\n"
        "       Set -DCMAKE_PREFIX_PATH=<path/to/libtorch> to enable.")
    return()
endif()

message(STATUS "[v4] LibTorch found: ${TORCH_INCLUDE_DIRS}")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})

# ── Include dirs ──────────────────────────────────────────────────────────────
# TORCH_INCLUDE_DIRS contains both torch/include and torch/include/torch/csrc/api/include
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    # ── Platform / OpenGL ──
    ${Boost_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX

    # ── SIBR ──────────────────────────────────────────────────────────────────
    sibr_openxr          # OpenXRHMD + OpenXRRdrMode
    sibr_view
    sibr_assets
    sibr_renderer
    # sibr_basic excluded: BUILD_IBR_BASIC=OFF (same reason as v3)

    # ── Gaussian rasterizer (geometry pass) ───────────────────────────────────
    CudaRasterizer       # rasterizer.h + libCudaRasterizer
    CUDA::cudart         # cudaMalloc, cudaMemcpy, cudaGraphics*

    # ── LibTorch (color MLP inference) ────────────────────────────────────────
    ${TORCH_LIBRARIES}
)

# ── Compiler flags ────────────────────────────────────────────────────────────
# LibTorch requires /EHsc on MSVC (C++ exceptions must be enabled).
# It also ships with its own CUDA runtime — avoid duplicate symbol warnings with /NODEFAULTLIB.
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /EHsc           # exception handling (required by LibTorch headers)
        /wd4251         # 'T' needs to have dll-interface (torch exports everything)
        /wd4275         # non dll-interface class used as base
    )
    # LibTorch links against a specific MSVC runtime. Match it.
    # torch ships with the DLLs — copy them to the output dir automatically.
    set_target_properties(${PROJECT_NAME} PROPERTIES
        VS_DEBUGGER_ENVIRONMENT "PATH=${TORCH_INSTALL_PREFIX}/lib;%PATH%"
    )
endif()

# ── CUDA toolkit path ─────────────────────────────────────────────────────────
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
    FOLDER "projects/remote/apps"
)

# ── Post-build: copy LibTorch DLLs next to the exe ───────────────────────────
# Without this the exe fails at runtime ("torch_cuda.dll not found").
# The file(GLOB ...) runs at *configure* time so it catches all DLLs present in the
# LibTorch lib/ directory. If you update LibTorch, re-run cmake to refresh the list.
if(WIN32 AND Torch_FOUND)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(dll ${TORCH_DLLS})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying ${dll} to output dir"
        )
    endforeach()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB
    RSC_FOLDER   "remote"
    STANDALONE   ${INSTALL_STANDALONE}
    COMPONENT    ${PROJECT_NAME}_install
)
