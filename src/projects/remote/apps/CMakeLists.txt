# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr



project(SIBR_remote_apps)

add_subdirectory(remoteGaussianUI/)
# NEW (v2): OpenXR stereo VR viewer — pixel streaming, Python rasterizes and sends RGB frames
add_subdirectory(remoteGaussianOpenXRv2/)
# NEW (v3): OpenXR stereo VR viewer for 3DGS-Avatar — attribute streaming, color MLP in Python
add_subdirectory(remoteGaussianOpenXR/)
# NEW (v4): OpenXR stereo VR viewer — color MLP moved to C++ (LibTorch JIT), per-eye view dirs
# Requires LibTorch: set -DCMAKE_PREFIX_PATH="<path/to/libtorch>" at cmake configure time.
# If LibTorch is not found the target is silently skipped.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/remoteGaussianOpenXRv4/CMakeLists.txt")
    add_subdirectory(remoteGaussianOpenXRv4/)
endif()
# NEW (v4.2): OpenXR stereo VR viewer — zero-copy CUDA IPC transport (no TCP bulk transfer).
# Same LibTorch requirement as v4. Also requires Windows 10 v2004+ / Windows 11 for IPC support.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/remoteGaussianOpenXRv4_2/CMakeLists.txt")
    add_subdirectory(remoteGaussianOpenXRv4_2/)
endif()
# NEW (v4.2 desktop): mono desktop-window viewer — same v4.2 CUDA IPC transport and
# the same GaussianLiveViewV42 (compiled from remoteGaussianOpenXRv4_2/), no OpenXR.
# Runs against the unchanged Python server. Same LibTorch / Windows IPC requirements.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/remoteGaussianDesktopV42/CMakeLists.txt")
    add_subdirectory(remoteGaussianDesktopV42/)
endif()