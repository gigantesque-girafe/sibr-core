# NEW FILE — remoteGaussianOpenXR/CMakeLists.txt
#
# Builds SIBR_remoteGaussianOpenXR_app.
# Links: sibr_openxr (OpenXRRdrMode) + sibr_gaussian (CudaRasterizer, CUDA).
# GaussianLiveView.cpp is compiled directly into the exe (not a separate lib).

project(SIBR_remoteGaussianOpenXR_app)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})

# GaussianLiveView.cpp includes <rasterizer.h> which comes from CudaRasterizer.
# The CudaRasterizer target is registered by gaussianviewer/renderer/CMakeLists.txt
# via sibr_gitlibrary — available globally within this CMake build.
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX
    sibr_openxr          # OpenXRHMD + OpenXRRdrMode
    # sibr_gaussian removed: we call CudaRasterizer directly; sibr_gaussian
    # pulls in sibr_video which has an OpenCV 4.5.0 ABI issue.
    sibr_view
    sibr_assets
    sibr_renderer
    # sibr_basic removed: BUILD_IBR_BASIC=OFF so it is never built
    CudaRasterizer       # rasterizer.h + libCudaRasterizer (direct)
    CUDA::cudart         # cudaMalloc, cudaMemcpy, cudaGraphics*
)

# CUDA 11.8 toolkit — set explicitly so CMake finds nvcc on VS2022
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
    FOLDER "projects/remote/apps"
)

include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB
    RSC_FOLDER   "remote"
    STANDALONE   ${INSTALL_STANDALONE}
    COMPONENT    ${PROJECT_NAME}_install
)
