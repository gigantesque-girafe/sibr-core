# ─────────────────────────────────────────────────────────────────────────────
# NEW FILE — remoteGaussianDesktopV42/CMakeLists.txt
#
# Builds SIBR_remoteGaussianDesktopV42_app: the v4.2 CUDA IPC pipeline rendered
# into a normal mono desktop window instead of an OpenXR headset.
#
# Deliberately does NOT copy GaussianLiveViewV42.{hpp,cpp} into this folder — it
# compiles the originals in place from ../remoteGaussianOpenXRv4_2/ so the two
# apps can never drift apart. The v4.2 app's own CMakeLists is untouched (it
# globs only its own directory, and this target is a separate executable, so
# compiling the class twice is harmless).
#
# Same prerequisites as v4.2: LibTorch (CUDA build) via
#   cmake ... -DCMAKE_PREFIX_PATH="C:/libtorch"
# and Windows 10 v2004+ / Windows 11 for CUDA IPC.
#
# Note: sibr_openxr is NOT linked here — GaussianLiveViewV42 never referenced it.
# ─────────────────────────────────────────────────────────────────────────────

project(SIBR_remoteGaussianDesktopV42_app)

set(V42_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../remoteGaussianOpenXRv4_2")

if(NOT EXISTS "${V42_SHARED_DIR}/GaussianLiveViewV42.cpp")
    message(STATUS
        "[desktop-v4.2] ${V42_SHARED_DIR}/GaussianLiveViewV42.cpp not found — "
        "skipping ${PROJECT_NAME}.")
    return()
endif()

# ── LibTorch ──────────────────────────────────────────────────────────────────
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    message(STATUS
        "[desktop-v4.2] LibTorch not found — skipping ${PROJECT_NAME}.\n"
        "       Set -DCMAKE_PREFIX_PATH=<path/to/libtorch> to enable.")
    return()
endif()

message(STATUS "[desktop-v4.2] LibTorch found: ${TORCH_INCLUDE_DIRS}")

# This LibTorch build injects /d2implyavx512upperregs- via INTERFACE_COMPILE_OPTIONS
# on its imported targets. That MSVC-internal flag is only understood by the VS2022
# toolset Torch was built with; our VS2019 cl.exe rejects it with C1007. Strip it.
# (Same workaround as v4.2 — the properties are global, but this target may be
# configured first, so it must not depend on v4.2 having already run.)
foreach(_torch_tgt c10 c10_cuda torch_cpu torch_cuda)
    if(TARGET ${_torch_tgt})
        get_target_property(_torch_opts ${_torch_tgt} INTERFACE_COMPILE_OPTIONS)
        if(_torch_opts)
            list(FILTER _torch_opts EXCLUDE REGEX "implyavx512upperregs")
            set_target_properties(${_torch_tgt} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_torch_opts}")
        endif()
    endif()
endforeach()

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Sources: this app's main.cpp + the shared view from the v4.2 app ──────────
set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${V42_SHARED_DIR}/GaussianLiveViewV42.cpp"
    "${V42_SHARED_DIR}/GaussianLiveViewV42.hpp"
)
source_group("Source Files" FILES ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})

# ── Include dirs ──────────────────────────────────────────────────────────────
# V42_SHARED_DIR so `#include "GaussianLiveViewV42.hpp"` resolves from main.cpp.
target_include_directories(${PROJECT_NAME} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${V42_SHARED_DIR}
)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX

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
    CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
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
