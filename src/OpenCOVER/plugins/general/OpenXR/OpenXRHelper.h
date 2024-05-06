// Copyright 2023, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OpenXR Tutorial for Khronos Group

#pragma once
// Define any XR_USE_PLATFORM_... / XR_USE_GRAPHICS_API_... before this header file.

// OpenXR Headers
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

// For DEBUG_BREAK
#include <HelperFunctions.h>

inline void OpenXRDebugBreak() {
    std::cerr << "Breakpoint here to debug." << std::endl;
    DEBUG_BREAK;
}

inline const char* GetXRErrorString(XrInstance xrInstance, XrResult result) {
    static char string[XR_MAX_RESULT_STRING_SIZE];
    xrResultToString(xrInstance, result, string);
    return string;
}

#define OPENXR_CHECK(x, y)                                                                                                                                  \
    {                                                                                                                                                       \
        XrResult result = (x);                                                                                                                              \
        if (!XR_SUCCEEDED(result)) {                                                                                                                        \
            std::cerr << "ERROR: OPENXR: " << int(result) << "(" << (m_xrInstance ? GetXRErrorString(m_xrInstance, result) : "") << ") " << y << std::endl; \
            OpenXRDebugBreak();                                                                                                                             \
        }                                                                                                                                                   \
    }
