# ─────────────────────────────────────────────────────────────────────────────
# NEW FILE — remoteGaussianOpenXRv4_2/CMakeLists.txt
#
# Builds SIBR_remoteGaussianOpenXRv4_2_app.
#
# Differences from v4.0 (remoteGaussianOpenXRv4):
#   - Per-frame Gaussian attributes arrive via CUDA IPC (zero-copy) instead of TCP.
#   - fetchFromPython() sends 1-byte ping; Python packs the IPC buffer and records
#     a CUDA event before sending back uint32 N_live.
#   - No per-Gaussian cudaMalloc in C++; positions, features, and rasterizer inputs
#     are read directly from the IPC-mapped CUDA pointer.
#
# Prerequisites:
#   Same as v4.0. LibTorch (CUDA build) must be findable:
#     cmake ... -DCMAKE_PREFIX_PATH="C:/libtorch"
#
#   CUDA IPC requires Windows 10 v2004+ (build 19041) or Windows 11,
#   NVIDIA driver 452.39+, and CUDA 11.2+.
# ─────────────────────────────────────────────────────────────────────────────

project(SIBR_remoteGaussianOpenXRv4_2_app)

# ── LibTorch ──────────────────────────────────────────────────────────────────
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    message(STATUS
        "[v4.2] LibTorch not found — skipping ${PROJECT_NAME}.\n"
        "       Set -DCMAKE_PREFIX_PATH=<path/to/libtorch> to enable.")
    return()
endif()

message(STATUS "[v4.2] LibTorch found: ${TORCH_INCLUDE_DIRS}")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})

# ── Include dirs ──────────────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX

    sibr_openxr
    sibr_view
    sibr_assets
    sibr_renderer

    CudaRasterizer
    CUDA::cudart          # cudaIpcGetMemHandle / cudaIpcOpenMemHandle etc.

    ${TORCH_LIBRARIES}
)

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /EHsc
        /wd4251
        /wd4275
    )
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
