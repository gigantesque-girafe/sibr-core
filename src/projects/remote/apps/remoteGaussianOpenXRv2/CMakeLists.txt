# NEW FILE — remoteGaussianOpenXRv2/CMakeLists.txt
#
# Builds SIBR_remoteGaussianOpenXRv2_app.
# Pixel-streaming mode: no CUDA, no CudaRasterizer.
# Only needs boost (TCP), OpenGL/GLEW (texture blit), and sibr_openxr.

project(SIBR_remoteGaussianOpenXRv2_app)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${Boost_LIBRARIES}
    ${ASSIMP_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX
    sibr_openxr          # OpenXRHMD + OpenXRRdrMode
    sibr_view
    sibr_assets
    sibr_renderer
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    FOLDER "projects/remote/apps"
)

include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB
    RSC_FOLDER   "remote"
    STANDALONE   ${INSTALL_STANDALONE}
    COMPONENT    ${PROJECT_NAME}_install
)
