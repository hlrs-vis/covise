// Copyright 2023, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OpenXR Tutorial for Khronos Group

#include <GraphicsAPI_OpenGL.h>

#if defined(XR_USE_GRAPHICS_API_OPENGL)

#if defined(OS_WINDOWS)
PROC GetExtension(const char *functionName) { return wglGetProcAddress(functionName); }
#elif defined(OS_APPLE)
void (*GetExtension(const char *functionName))() { return NULL; }
#elif defined(OS_LINUX_XCB) || defined(OS_LINUX_XLIB) || defined(OS_LINUX_XCB_GLX)
void (*GetExtension(const char *functionName))() { return glXGetProcAddress((const GLubyte *)functionName); }
#elif defined(OS_ANDROID) || defined(OS_LINUX_WAYLAND)
void (*GetExtension(const char *functionName))() { return eglGetProcAddress(functionName); }
#endif

#pragma region PiplineHelpers

GLenum GetGLTextureTarget(const GraphicsAPI::ImageCreateInfo &imageCI) {
    GLenum target = 0;
    if (imageCI.dimension == 1) {
        if (imageCI.arrayLayers > 1) {
            target = GL_TEXTURE_1D_ARRAY;
        } else {
            target = GL_TEXTURE_1D;
        }
    } else if (imageCI.dimension == 2) {
        if (imageCI.cubemap) {
            if (imageCI.arrayLayers > 6) {
                target = GL_TEXTURE_CUBE_MAP_ARRAY;
            } else {
                target = GL_TEXTURE_CUBE_MAP;
            }
        } else {
            if (imageCI.sampleCount > 1) {
                if (imageCI.arrayLayers > 1) {
                    target = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
                } else {
                    target = GL_TEXTURE_2D_MULTISAMPLE;
                }
            } else {
                if (imageCI.arrayLayers > 1) {
                    target = GL_TEXTURE_2D_ARRAY;
                } else {
                    target = GL_TEXTURE_2D;
                }
            }
        }
    } else if (imageCI.dimension == 3) {
        target = GL_TEXTURE_3D;
    } else {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Unknown Dimension for GetGLTextureTarget(): " << imageCI.dimension << std::endl;
    }
    return target;
}

GLint ToGLFilter(GraphicsAPI::SamplerCreateInfo::Filter filter) {
    switch (filter) {
    case GraphicsAPI::SamplerCreateInfo::Filter::NEAREST:
        return GL_NEAREST;
    case GraphicsAPI::SamplerCreateInfo::Filter::LINEAR:
        return GL_LINEAR;
    default:
        return 0;
    }
};
GLint ToGLFilterMipmap(GraphicsAPI::SamplerCreateInfo::Filter filter, GraphicsAPI::SamplerCreateInfo::MipmapMode mipmapMode) {
    switch (filter) {
    case GraphicsAPI::SamplerCreateInfo::Filter::NEAREST: {
        if (mipmapMode == GraphicsAPI::SamplerCreateInfo::MipmapMode::NEAREST)
            return GL_NEAREST_MIPMAP_LINEAR;
        else if (mipmapMode == GraphicsAPI::SamplerCreateInfo::MipmapMode::LINEAR)
            return GL_NEAREST_MIPMAP_NEAREST;
        else
            return GL_NEAREST;
    }
    case GraphicsAPI::SamplerCreateInfo::Filter::LINEAR: {
        if (mipmapMode == GraphicsAPI::SamplerCreateInfo::MipmapMode::NEAREST)
            return GL_LINEAR_MIPMAP_LINEAR;
        else if (mipmapMode == GraphicsAPI::SamplerCreateInfo::MipmapMode::LINEAR)
            return GL_LINEAR_MIPMAP_NEAREST;
        else
            return GL_LINEAR;
    }
    default:
        return 0;
    }
};
GLint ToGLAddressMode(GraphicsAPI::SamplerCreateInfo::AddressMode mode) {
    switch (mode) {
    case GraphicsAPI::SamplerCreateInfo::AddressMode::REPEAT:
        return GL_REPEAT;
    case GraphicsAPI::SamplerCreateInfo::AddressMode::MIRRORED_REPEAT:
        return GL_MIRRORED_REPEAT;
    case GraphicsAPI::SamplerCreateInfo::AddressMode::CLAMP_TO_EDGE:
        return GL_CLAMP_TO_EDGE;
    case GraphicsAPI::SamplerCreateInfo::AddressMode::CLAMP_TO_BORDER:
        return GL_CLAMP_TO_BORDER;
    case GraphicsAPI::SamplerCreateInfo::AddressMode::MIRROR_CLAMP_TO_EDGE:
        return GL_MIRROR_CLAMP_TO_EDGE;
    default:
        return 0;
    }
};

inline GLenum ToGLTopology(GraphicsAPI::PrimitiveTopology topology) {
    switch (topology) {
    case GraphicsAPI::PrimitiveTopology::POINT_LIST:
        return GL_POINTS;
    case GraphicsAPI::PrimitiveTopology::LINE_LIST:
        return GL_LINES;
    case GraphicsAPI::PrimitiveTopology::LINE_STRIP:
        return GL_LINE_STRIP;
    case GraphicsAPI::PrimitiveTopology::TRIANGLE_LIST:
        return GL_TRIANGLES;
    case GraphicsAPI::PrimitiveTopology::TRIANGLE_STRIP:
        return GL_TRIANGLE_STRIP;
    case GraphicsAPI::PrimitiveTopology::TRIANGLE_FAN:
        return GL_TRIANGLE_FAN;
    default:
        return 0;
    }
};
inline GLenum ToGLPolygonMode(GraphicsAPI::PolygonMode polygonMode) {
    switch (polygonMode) {
    case GraphicsAPI::PolygonMode::FILL:
        return GL_FILL;
    case GraphicsAPI::PolygonMode::LINE:
        return GL_LINE;
    case GraphicsAPI::PolygonMode::POINT:
        return GL_POINT;
    default:
        return 0;
    }
};
inline GLenum ToGLCullMode(GraphicsAPI::CullMode cullMode) {
    switch (cullMode) {
    case GraphicsAPI::CullMode::NONE:
        return GL_BACK;
    case GraphicsAPI::CullMode::FRONT:
        return GL_FRONT;
    case GraphicsAPI::CullMode::BACK:
        return GL_BACK;
    case GraphicsAPI::CullMode::FRONT_AND_BACK:
        return GL_FRONT_AND_BACK;
    default:
        return 0;
    }
}
inline GLenum ToGLCompareOp(GraphicsAPI::CompareOp op) {
    switch (op) {
    case GraphicsAPI::CompareOp::NEVER:
        return GL_NEVER;
    case GraphicsAPI::CompareOp::LESS:
        return GL_LESS;
    case GraphicsAPI::CompareOp::EQUAL:
        return GL_EQUAL;
    case GraphicsAPI::CompareOp::LESS_OR_EQUAL:
        return GL_LEQUAL;
    case GraphicsAPI::CompareOp::GREATER:
        return GL_GREATER;
    case GraphicsAPI::CompareOp::NOT_EQUAL:
        return GL_NOTEQUAL;
    case GraphicsAPI::CompareOp::GREATER_OR_EQUAL:
        return GL_GEQUAL;
    case GraphicsAPI::CompareOp::ALWAYS:
        return GL_ALWAYS;
    default:
        return 0;
    }
};
inline GLenum ToGLStencilCompareOp(GraphicsAPI::StencilOp op) {
    switch (op) {
    case GraphicsAPI::StencilOp::KEEP:
        return GL_KEEP;
    case GraphicsAPI::StencilOp::ZERO:
        return GL_ZERO;
    case GraphicsAPI::StencilOp::REPLACE:
        return GL_REPLACE;
    case GraphicsAPI::StencilOp::INCREMENT_AND_CLAMP:
        return GL_INCR;
    case GraphicsAPI::StencilOp::DECREMENT_AND_CLAMP:
        return GL_DECR;
    case GraphicsAPI::StencilOp::INVERT:
        return GL_INVERT;
    case GraphicsAPI::StencilOp::INCREMENT_AND_WRAP:
        return GL_INCR_WRAP;
    case GraphicsAPI::StencilOp::DECREMENT_AND_WRAP:
        return GL_DECR_WRAP;
    default:
        return 0;
    }
};
inline GLenum ToGLBlendFactor(GraphicsAPI::BlendFactor factor) {
    switch (factor) {
    case GraphicsAPI::BlendFactor::ZERO:
        return GL_ZERO;
    case GraphicsAPI::BlendFactor::ONE:
        return GL_ONE;
    case GraphicsAPI::BlendFactor::SRC_COLOR:
        return GL_SRC_COLOR;
    case GraphicsAPI::BlendFactor::ONE_MINUS_SRC_COLOR:
        return GL_ONE_MINUS_SRC_COLOR;
    case GraphicsAPI::BlendFactor::DST_COLOR:
        return GL_DST_COLOR;
    case GraphicsAPI::BlendFactor::ONE_MINUS_DST_COLOR:
        return GL_ONE_MINUS_DST_COLOR;
    case GraphicsAPI::BlendFactor::SRC_ALPHA:
        return GL_SRC_ALPHA;
    case GraphicsAPI::BlendFactor::ONE_MINUS_SRC_ALPHA:
        return GL_ONE_MINUS_SRC_ALPHA;
    case GraphicsAPI::BlendFactor::DST_ALPHA:
        return GL_DST_ALPHA;
    case GraphicsAPI::BlendFactor::ONE_MINUS_DST_ALPHA:
        return GL_ONE_MINUS_DST_ALPHA;
    default:
        return 0;
    }
};
inline GLenum ToGLBlendOp(GraphicsAPI::BlendOp op) {
    switch (op) {
    case GraphicsAPI::BlendOp::ADD:
        return GL_FUNC_ADD;
    case GraphicsAPI::BlendOp::SUBTRACT:
        return GL_FUNC_SUBTRACT;
    case GraphicsAPI::BlendOp::REVERSE_SUBTRACT:
        return GL_FUNC_REVERSE_SUBTRACT;
    case GraphicsAPI::BlendOp::MIN:
        return GL_MIN;
    case GraphicsAPI::BlendOp::MAX:
        return GL_MAX;
    default:
        return 0;
    }
};
inline GLenum ToGLLogicOp(GraphicsAPI::LogicOp op) {
    switch (op) {
    case GraphicsAPI::LogicOp::CLEAR:
        return GL_CLEAR;
    case GraphicsAPI::LogicOp::AND:
        return GL_AND;
    case GraphicsAPI::LogicOp::AND_REVERSE:
        return GL_AND_REVERSE;
    case GraphicsAPI::LogicOp::COPY:
        return GL_COPY;
    case GraphicsAPI::LogicOp::AND_INVERTED:
        return GL_AND_INVERTED;
    case GraphicsAPI::LogicOp::NO_OP:
        return GL_NOOP;
    case GraphicsAPI::LogicOp::XOR:
        return GL_XOR;
    case GraphicsAPI::LogicOp::OR:
        return GL_OR;
    case GraphicsAPI::LogicOp::NOR:
        return GL_NOR;
    case GraphicsAPI::LogicOp::EQUIVALENT:
        return GL_EQUIV;
    case GraphicsAPI::LogicOp::INVERT:
        return GL_INVERT;
    case GraphicsAPI::LogicOp::OR_REVERSE:
        return GL_OR_REVERSE;
    case GraphicsAPI::LogicOp::COPY_INVERTED:
        return GL_COPY_INVERTED;
    case GraphicsAPI::LogicOp::OR_INVERTED:
        return GL_OR_INVERTED;
    case GraphicsAPI::LogicOp::NAND:
        return GL_NAND;
    case GraphicsAPI::LogicOp::SET:
        return GL_SET;
    default:
        return 0;
    }
};
#pragma endregion

void GLDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
    std::cout << "OpenGL Debug message (" << id << "): " << message << std::endl;

    switch (source) {
    case GL_DEBUG_SOURCE_API:
        std::cout << "Source: API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        std::cout << "Source: Window System";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        std::cout << "Source: Shader Compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        std::cout << "Source: Third Party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        std::cout << "Source: Application";
        break;
    case GL_DEBUG_SOURCE_OTHER:
        std::cout << "Source: Other";
        break;
    }
    std::cout << std::endl;

    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        std::cout << "Type: Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        std::cout << "Type: Deprecated Behaviour";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        std::cout << "Type: Undefined Behaviour";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        std::cout << "Type: Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        std::cout << "Type: Performance";
        break;
    case GL_DEBUG_TYPE_MARKER:
        std::cout << "Type: Marker";
        break;
    case GL_DEBUG_TYPE_PUSH_GROUP:
        std::cout << "Type: Push Group";
        break;
    case GL_DEBUG_TYPE_POP_GROUP:
        std::cout << "Type: Pop Group";
        break;
    case GL_DEBUG_TYPE_OTHER:
        std::cout << "Type: Other";
        break;
    }
    std::cout << std::endl;

    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        std::cout << "Severity: high";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        std::cout << "Severity: medium";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        std::cout << "Severity: low";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        std::cout << "Severity: notification";
        break;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    if (type == GL_DEBUG_TYPE_ERROR)
        DEBUG_BREAK;
}

GraphicsAPI_OpenGL::GraphicsAPI_OpenGL() {
    // https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/f122f9f1fc729e2dc82e12c3ce73efa875182854/src/tests/hello_xr/graphicsplugin_opengl.cpp#L103-L121
    // Initialize the gl extensions. Note we have to open a window.
    ksDriverInstance driverInstance{};
    ksGpuQueueInfo queueInfo{};
    ksGpuSurfaceColorFormat colorFormat{KS_GPU_SURFACE_COLOR_FORMAT_B8G8R8A8};
    ksGpuSurfaceDepthFormat depthFormat{KS_GPU_SURFACE_DEPTH_FORMAT_D24};
    ksGpuSampleCount sampleCount{KS_GPU_SAMPLE_COUNT_1};
    if (!ksGpuWindow_Create(&window, &driverInstance, &queueInfo, 0, colorFormat, depthFormat, sampleCount, 640, 480, false)) {
        std::cerr << "ERROR: OPENGL: Failed to create Context." << std::endl;
    }

    GLint glMajorVersion = 0;
    GLint glMinorVersion = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &glMajorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &glMinorVersion);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GLDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
}

GraphicsAPI_OpenGL::GraphicsAPI_OpenGL(XrInstance m_xrInstance, XrSystemId systemId) {
    OPENXR_CHECK(xrGetInstanceProcAddr(m_xrInstance, "xrGetOpenGLGraphicsRequirementsKHR", (PFN_xrVoidFunction *)&xrGetOpenGLGraphicsRequirementsKHR), "Failed to get InstanceProcAddr for xrGetOpenGLGraphicsRequirementsKHR.");
    XrGraphicsRequirementsOpenGLKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR};
    OPENXR_CHECK(xrGetOpenGLGraphicsRequirementsKHR(m_xrInstance, systemId, &graphicsRequirements), "Failed to get Graphics Requirements for OpenGL.");

    // https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/f122f9f1fc729e2dc82e12c3ce73efa875182854/src/tests/hello_xr/graphicsplugin_opengl.cpp#L103-L121
    // Initialize the gl extensions. Note we have to open a window.
    ksDriverInstance driverInstance{};
    ksGpuQueueInfo queueInfo{};
    ksGpuSurfaceColorFormat colorFormat{KS_GPU_SURFACE_COLOR_FORMAT_B8G8R8A8};
    ksGpuSurfaceDepthFormat depthFormat{KS_GPU_SURFACE_DEPTH_FORMAT_D24};
    ksGpuSampleCount sampleCount{KS_GPU_SAMPLE_COUNT_1};
    if (!ksGpuWindow_Create(&window, &driverInstance, &queueInfo, 0, colorFormat, depthFormat, sampleCount, 640, 480, false)) {
        std::cerr << "ERROR: OPENGL: Failed to create Context." << std::endl;
    }

    GLint glMajorVersion = 0;
    GLint glMinorVersion = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &glMajorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &glMinorVersion);

    const XrVersion glApiVersion = XR_MAKE_VERSION(glMajorVersion, glMinorVersion, 0);
    if (graphicsRequirements.minApiVersionSupported > glApiVersion) {
        int requiredMajorVersion = XR_VERSION_MAJOR(graphicsRequirements.minApiVersionSupported);
        int requiredMinorVersion = XR_VERSION_MINOR(graphicsRequirements.minApiVersionSupported);
        std::cerr << "ERROR: OPENGL: The created OpenGL version " << glMajorVersion << "." << glMinorVersion << " doesn't meet the minimum required API version " << requiredMajorVersion << "." << requiredMinorVersion << " for OpenXR." << std::endl;
    }

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GLDebugCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
    glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
}

GraphicsAPI_OpenGL::~GraphicsAPI_OpenGL() {
    ksGpuWindow_Destroy(&window);
}

void *GraphicsAPI_OpenGL::CreateDesktopSwapchain(const SwapchainCreateInfo &swapchainCI) { return nullptr; }
void GraphicsAPI_OpenGL::DestroyDesktopSwapchain(void *&swapchain) {}
void *GraphicsAPI_OpenGL::GetDesktopSwapchainImage(void *swapchain, uint32_t index) { return nullptr; }
void GraphicsAPI_OpenGL::AcquireDesktopSwapchanImage(void *swapchain, uint32_t &index) {}
void GraphicsAPI_OpenGL::PresentDesktopSwapchainImage(void *swapchain, uint32_t index) {
#if defined(XR_USE_PLATFORM_WIN32)
    SwapBuffers(window.hDC);
#elif defined(XR_USE_PLATFORM_XLIB) || defined(XR_USE_PLATFORM_XCB)
	glXSwapBuffers(window.context.xDisplay, window.context.glxDrawable);
#endif
}

void *GraphicsAPI_OpenGL::GetGraphicsBinding() {
    // https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/f122f9f1fc729e2dc82e12c3ce73efa875182854/src/tests/hello_xr/graphicsplugin_opengl.cpp#L123-L144
#if defined(XR_USE_PLATFORM_WIN32)
    graphicsBinding = {XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR};
    graphicsBinding.hDC = window.context.hDC;
    graphicsBinding.hGLRC = window.context.hGLRC;
#elif defined(XR_USE_PLATFORM_XLIB)
    graphicsBinding = {XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR};
    graphicsBinding.xDisplay = window.context.xDisplay;
    graphicsBinding.visualid = window.context.visualid;
    graphicsBinding.glxFBConfig = window.context.glxFBConfig;
    graphicsBinding.glxDrawable = window.context.glxDrawable;
    graphicsBinding.glxContext = window.context.glxContext;
#elif defined(XR_USE_PLATFORM_XCB)
    graphicsBinding = {XR_TYPE_GRAPHICS_BINDING_OPENGL_XCB_KHR};
    // TODO: Still missing the platform adapter, and some items to make this usable.
    graphicsBinding.connection = window.connection;
    // m_graphicsBinding.screenNumber = window.context.screenNumber;
    // m_graphicsBinding.fbconfigid = window.context.fbconfigid;
    graphicsBinding.visualid = window.context.visualid;
    graphicsBinding.glxDrawable = window.context.glxDrawable;
    // m_graphicsBinding.glxContext = window.context.glxContext;
#elif defined(XR_USE_PLATFORM_WAYLAND)
    // TODO: Just need something other than NULL here for now (for validation).  Eventually need
    //       to correctly put in a valid pointer to an wl_display
    graphicsBinding = {XR_TYPE_GRAPHICS_BINDING_OPENGL_WAYLAND};
    graphicsBinding.display = reinterpret_cast<wl_display *>(0xFFFFFFFF);
#endif
    return &graphicsBinding;
}

XrSwapchainImageBaseHeader *GraphicsAPI_OpenGL::AllocateSwapchainImageData(XrSwapchain swapchain, SwapchainType type, uint32_t count) {
    swapchainImagesMap[swapchain].first = type;
    swapchainImagesMap[swapchain].second.resize(count, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR});
    return reinterpret_cast<XrSwapchainImageBaseHeader *>(swapchainImagesMap[swapchain].second.data());
}

void *GraphicsAPI_OpenGL::CreateImage(const ImageCreateInfo &imageCI) {
    GLuint texture = 0;
    glGenTextures(1, &texture);

    GLenum target = GetGLTextureTarget(imageCI);
    glBindTexture(target, texture);

    if (target == GL_TEXTURE_1D) {
        // glTexStorage1D() is not available - Poor work around.
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexStorage2D(GL_TEXTURE_2D, imageCI.mipLevels, imageCI.format, imageCI.width, 1);
        glBindTexture(GL_TEXTURE_2D, 0);
    } else if (target == GL_TEXTURE_2D) {
        glTexStorage2D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.height);
    } else if (target == GL_TEXTURE_2D_MULTISAMPLE) {
        glTexStorage2DMultisample(target, imageCI.sampleCount, imageCI.format, imageCI.width, imageCI.height, GL_TRUE);
    } else if (target == GL_TEXTURE_3D) {
        glTexStorage3D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.height, imageCI.depth);
    } else if (target == GL_TEXTURE_CUBE_MAP) {
        glTexStorage2D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.height);
    } else if (target == GL_TEXTURE_1D_ARRAY) {
        glTexStorage2D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.arrayLayers);
    } else if (target == GL_TEXTURE_2D_ARRAY) {
        glTexStorage3D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.height, imageCI.arrayLayers);
    } else if (target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
        glTexStorage3DMultisample(target, imageCI.sampleCount, imageCI.format, imageCI.width, imageCI.height, imageCI.arrayLayers, GL_TRUE);
    } else if (target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        glTexStorage3D(target, imageCI.mipLevels, imageCI.format, imageCI.width, imageCI.height, imageCI.arrayLayers);
    }

    glBindTexture(target, 0);

    images[texture] = imageCI;
    return (void *)(uint64_t)texture;
}

void GraphicsAPI_OpenGL::DestroyImage(void *&image) {
    GLuint texture = (GLuint)(uint64_t)image;
    images.erase(texture);
    glDeleteTextures(1, &texture);
    image = nullptr;
}

void *GraphicsAPI_OpenGL::CreateImageView(const ImageViewCreateInfo &imageViewCI) {
    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);

    GLenum attachment = imageViewCI.aspect == ImageViewCreateInfo::Aspect::COLOR_BIT ? GL_COLOR_ATTACHMENT0 : GL_DEPTH_ATTACHMENT;

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D_ARRAY) {
        glFramebufferTextureMultiviewOVR(GL_DRAW_FRAMEBUFFER, attachment, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel, imageViewCI.baseArrayLayer, imageViewCI.layerCount);
    } else if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D) {
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, GL_TEXTURE_2D, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel);
    } else {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Unknown ImageView View type." << std::endl;
    }

    GLenum result = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (result != GL_FRAMEBUFFER_COMPLETE) {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Framebuffer is not complete." << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    imageViews[framebuffer] = imageViewCI;
    return (void *)(uint64_t)framebuffer;
}

void GraphicsAPI_OpenGL::DestroyImageView(void *&imageView) {
    GLuint framebuffer = (GLuint)(uint64_t)imageView;
    imageViews.erase(framebuffer);
    glDeleteFramebuffers(1, &framebuffer);
    imageView = nullptr;
}

void *GraphicsAPI_OpenGL::CreateSampler(const SamplerCreateInfo &samplerCI) {
    GLuint sampler = 0;
    PFNGLGENSAMPLERSPROC glGenSamplers = (PFNGLGENSAMPLERSPROC)GetExtension("glGenSamplers");  // 3.2+
    glGenSamplers(1, &sampler);

    PFNGLSAMPLERPARAMETERIPROC glSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC)GetExtension("glSamplerParameteri");      // 3.2+
    PFNGLSAMPLERPARAMETERFPROC glSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC)GetExtension("glSamplerParameterf");      // 3.2+
    PFNGLSAMPLERPARAMETERFVPROC glSamplerParameterfv = (PFNGLSAMPLERPARAMETERFVPROC)GetExtension("glSamplerParameterfv");  // 3.2+

    // Filter
    glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, ToGLFilter(samplerCI.magFilter));
    glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, ToGLFilterMipmap(samplerCI.minFilter, samplerCI.mipmapMode));

    // AddressMode

    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, ToGLAddressMode(samplerCI.addressModeS));
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, ToGLAddressMode(samplerCI.addressModeT));
    glSamplerParameteri(sampler, GL_TEXTURE_WRAP_R, ToGLAddressMode(samplerCI.addressModeR));

    // Lod Bias
    glSamplerParameterf(sampler, GL_TEXTURE_LOD_BIAS, samplerCI.mipLodBias);

    // Compare
    glSamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, samplerCI.compareEnable ? GL_COMPARE_REF_TO_TEXTURE : GL_NONE);
    glSamplerParameteri(sampler, GL_TEXTURE_COMPARE_FUNC, ToGLCompareOp(samplerCI.compareOp));

    // Lod
    glSamplerParameterf(sampler, GL_TEXTURE_MIN_LOD, samplerCI.minLod);
    glSamplerParameterf(sampler, GL_TEXTURE_MAX_LOD, samplerCI.maxLod);

    // BorderColor
    glSamplerParameterfv(sampler, GL_TEXTURE_BORDER_COLOR, samplerCI.borderColor);

    return (void *)(uint64_t)sampler;
}

void GraphicsAPI_OpenGL::DestroySampler(void *&sampler) {
    GLuint glsampler = (GLuint)(uint64_t)sampler;
    PFNGLDELETESAMPLERSPROC glDeleteSamplers = (PFNGLDELETESAMPLERSPROC)GetExtension("glDeleteSamplers");  // 3.2+
    glDeleteSamplers(1, &glsampler);
    sampler = nullptr;
}

void *GraphicsAPI_OpenGL::CreateBuffer(const BufferCreateInfo &bufferCI) {
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);

    GLenum target = 0;
    if (bufferCI.type == BufferCreateInfo::Type::VERTEX) {
        target = GL_ARRAY_BUFFER;
    } else if (bufferCI.type == BufferCreateInfo::Type::INDEX) {
        target = GL_ELEMENT_ARRAY_BUFFER;
    } else if (bufferCI.type == BufferCreateInfo::Type::UNIFORM) {
        target = GL_UNIFORM_BUFFER;
    } else {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Unknown Buffer Type." << std::endl;
    }

    glBindBuffer(target, buffer);
    glBufferData(target, (GLsizeiptr)bufferCI.size, bufferCI.data, GL_STATIC_DRAW);
    glBindBuffer(target, 0);

    buffers[buffer] = bufferCI;
    return (void *)(uint64_t)buffer;
}

void GraphicsAPI_OpenGL::DestroyBuffer(void *&buffer) {
    GLuint glBuffer = (GLuint)(uint64_t)buffer;
    buffers.erase(glBuffer);
    glDeleteBuffers(1, &glBuffer);
    buffer = nullptr;
}

void *GraphicsAPI_OpenGL::CreateShader(const ShaderCreateInfo &shaderCI) {
    GLenum type = 0;
    switch (shaderCI.type) {
    case ShaderCreateInfo::Type::VERTEX: {
        type = GL_VERTEX_SHADER;
        break;
    }
    case ShaderCreateInfo::Type::TESSELLATION_CONTROL: {
        type = GL_TESS_CONTROL_SHADER;
        break;
    }
    case ShaderCreateInfo::Type::TESSELLATION_EVALUATION: {
        type = GL_TESS_EVALUATION_SHADER;
        break;
    }
    case ShaderCreateInfo::Type::GEOMETRY: {
        type = GL_GEOMETRY_SHADER;
        break;
    }
    case ShaderCreateInfo::Type::FRAGMENT: {
        type = GL_FRAGMENT_SHADER;
        break;
    }
    case ShaderCreateInfo::Type::COMPUTE: {
        type = GL_COMPUTE_SHADER;
        break;
    }
    default:
        std::cout << "ERROR: OPENGL: Unknown Shader Type." << std::endl;
    }
    GLuint shader = glCreateShader(type);

    glShaderSource(shader, 1, &shaderCI.sourceData, nullptr);
    glCompileShader(shader);

    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled);
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &infoLog[0]);
        std::cout << infoLog.data() << std::endl;
        DEBUG_BREAK;

        glDeleteShader(shader);
        shader = 0;
    }

    return (void *)(uint64_t)shader;
}

void GraphicsAPI_OpenGL::DestroyShader(void *&shader) {
    GLuint glShader = (GLuint)(uint64_t)shader;
    glDeleteShader(glShader);
    shader = nullptr;
}

void *GraphicsAPI_OpenGL::CreatePipeline(const PipelineCreateInfo &pipelineCI) {
    GLuint program = glCreateProgram();

    for (const void *const &shader : pipelineCI.shaders)
        glAttachShader(program, (GLuint)(uint64_t)shader);

    glLinkProgram(program);

    PFNGLVALIDATEPROGRAMPROC glValidateProgram = (PFNGLVALIDATEPROGRAMPROC)GetExtension("glValidateProgram");  // 2.0+
    glValidateProgram(program);

    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);
    if (isLinked == GL_FALSE) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> infoLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);

        glDeleteProgram(program);
    }

    PFNGLDETACHSHADERPROC glDetachShader = (PFNGLDETACHSHADERPROC)GetExtension("glDetachShader");  // 2.0+
    for (const void *const &shader : pipelineCI.shaders)
        glDetachShader(program, (GLuint)(uint64_t)shader);

    pipelines[program] = pipelineCI;

    return (void *)(uint64_t)program;
}

void GraphicsAPI_OpenGL::DestroyPipeline(void *&pipeline) {
    GLint program = (GLuint)(uint64_t)pipeline;
    pipelines.erase(program);
    glDeleteProgram(program);
    pipeline = nullptr;
}

void GraphicsAPI_OpenGL::BeginRendering() {
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    glGenFramebuffers(1, &setFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, setFramebuffer);
}

void GraphicsAPI_OpenGL::EndRendering() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &setFramebuffer);
    setFramebuffer = 0;

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vertexArray);
    vertexArray = 0;
}

void GraphicsAPI_OpenGL::SetBufferData(void *buffer, size_t offset, size_t size, void *data) {
    GLuint glBuffer = (GLuint)(uint64_t)buffer;
    const BufferCreateInfo &bufferCI = buffers[glBuffer];

    GLenum target = 0;
    if (bufferCI.type == BufferCreateInfo::Type::VERTEX) {
        target = GL_ARRAY_BUFFER;
    } else if (bufferCI.type == BufferCreateInfo::Type::INDEX) {
        target = GL_ELEMENT_ARRAY_BUFFER;
    } else if (bufferCI.type == BufferCreateInfo::Type::UNIFORM) {
        target = GL_UNIFORM_BUFFER;
    } else {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Unknown Buffer Type." << std::endl;
    }

    if (data) {
        glBindBuffer(target, glBuffer);
        glBufferSubData(target, (GLintptr)offset, (GLsizeiptr)size, data);
        glBindBuffer(target, 0);
    }
}

void GraphicsAPI_OpenGL::ClearColor(void *imageView, float r, float g, float b, float a) {
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)(uint64_t)imageView);
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GraphicsAPI_OpenGL::ClearDepth(void *imageView, float d) {
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)(uint64_t)imageView);
    glClearDepth(d);
    glClear(GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GraphicsAPI_OpenGL::SetRenderAttachments(void **colorViews, size_t colorViewCount, void *depthStencilView, uint32_t width, uint32_t height, void *pipeline) {
    // Reset Framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &setFramebuffer);
    setFramebuffer = 0;

    glGenFramebuffers(1, &setFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, setFramebuffer);

    // Color
    for (size_t i = 0; i < colorViewCount; i++) {
        GLenum attachment = GL_COLOR_ATTACHMENT0;

        GLuint glColorView = (GLuint)(uint64_t)colorViews[i];
        const ImageViewCreateInfo &imageViewCI = imageViews[glColorView];

        if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D_ARRAY) {
            glFramebufferTextureMultiviewOVR(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel, imageViewCI.baseArrayLayer, imageViewCI.layerCount);
        } else if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D) {
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel);
        } else {
            DEBUG_BREAK;
            std::cout << "ERROR: OPENGL: Unknown ImageView View type." << std::endl;
        }
    }
    // DepthStencil
    if (depthStencilView) {
        GLuint glDepthView = (GLuint)(uint64_t)depthStencilView;
        const ImageViewCreateInfo &imageViewCI = imageViews[glDepthView];

        if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D_ARRAY) {
            glFramebufferTextureMultiviewOVR(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel, imageViewCI.baseArrayLayer, imageViewCI.layerCount);
        } else if (imageViewCI.view == ImageViewCreateInfo::View::TYPE_2D) {
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, (GLuint)(uint64_t)imageViewCI.image, imageViewCI.baseMipLevel);
        } else {
            DEBUG_BREAK;
            std::cout << "ERROR: OPENGL: Unknown ImageView View type." << std::endl;
        }
    }

    GLenum result = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER);
    if (result != GL_FRAMEBUFFER_COMPLETE) {
        DEBUG_BREAK;
        std::cout << "ERROR: OPENGL: Framebuffer is not complete." << std::endl;
    }
}

void GraphicsAPI_OpenGL::SetViewports(Viewport *viewports, size_t count) {
    PFNGLVIEWPORTINDEXEDFPROC glViewportIndexedf = (PFNGLVIEWPORTINDEXEDFPROC)GetExtension("glViewportIndexedf");      // 4.1+
    PFNGLDEPTHRANGEINDEXEDPROC glDepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC)GetExtension("glDepthRangeIndexed");  // 4.1+

    for (size_t i = 0; i < count; i++) {
        Viewport viewport = viewports[i];
        glViewportIndexedf((GLuint)i, viewport.x, viewport.y, viewport.width, viewport.height);
        glDepthRangeIndexed((GLuint)i, (GLdouble)viewport.minDepth, (GLdouble)viewport.maxDepth);
    }
}

void GraphicsAPI_OpenGL::SetScissors(Rect2D *scissors, size_t count) {
    PFNGLSCISSORINDEXEDPROC glScissorIndexed = (PFNGLSCISSORINDEXEDPROC)GetExtension("glScissorIndexed");  // 4.1+

    for (size_t i = 0; i < count; i++) {
        Rect2D scissor = scissors[i];
        glScissorIndexed((GLuint)i, (GLint)scissor.offset.x, (GLint)scissor.offset.y, (GLsizei)scissor.extent.width, (GLsizei)scissor.extent.height);
    }
}

void GraphicsAPI_OpenGL::SetPipeline(void *pipeline) {
    GLuint program = (GLuint)(uint64_t)pipeline;
    glUseProgram(program);
    setPipeline = program;

    const PipelineCreateInfo &pipelineCI = pipelines[program];

    // InputAssemblyState
    const InputAssemblyState &IAS = pipelineCI.inputAssemblyState;
    if (IAS.primitiveRestartEnable) {
        glEnable(GL_PRIMITIVE_RESTART);
    } else {
        glDisable(GL_PRIMITIVE_RESTART);
    }

    // RasterisationState
    const RasterisationState &RS = pipelineCI.rasterisationState;

    if (RS.depthClampEnable) {
        glEnable(GL_DEPTH_CLAMP);
    } else {
        glDisable(GL_DEPTH_CLAMP);
    }

    if (RS.rasteriserDiscardEnable) {
        glEnable(GL_RASTERIZER_DISCARD);
    } else {
        glDisable(GL_RASTERIZER_DISCARD);
    }

    if (RS.cullMode == CullMode::FRONT_AND_BACK) {
        glPolygonMode(GL_FRONT_AND_BACK, ToGLPolygonMode(RS.polygonMode));
    }

    if (RS.cullMode > CullMode::NONE) {
        glEnable(GL_CULL_FACE);
        glCullFace(ToGLCullMode(RS.cullMode));
    } else {
        glDisable(GL_CULL_FACE);
    }

    glFrontFace(RS.frontFace == FrontFace::COUNTER_CLOCKWISE ? GL_CCW : GL_CW);

    GLenum polygonOffsetMode = 0;
    switch (RS.polygonMode) {
    default:
    case PolygonMode::FILL: {
        polygonOffsetMode = GL_POLYGON_OFFSET_FILL;
        break;
    }
    case PolygonMode::LINE: {
        polygonOffsetMode = GL_POLYGON_OFFSET_LINE;
        break;
    }
    case PolygonMode::POINT: {
        polygonOffsetMode = GL_POLYGON_OFFSET_POINT;
        break;
    }
    }
    if (RS.depthBiasEnable) {
        glEnable(polygonOffsetMode);
        // glPolygonOffsetClamp
        glPolygonOffset(RS.depthBiasSlopeFactor, RS.depthBiasConstantFactor);
    } else {
        glDisable(polygonOffsetMode);
    }

    glLineWidth(RS.lineWidth);

    // MultisampleState
    const MultisampleState &MS = pipelineCI.multisampleState;

    if (MS.rasterisationSamples > 1) {
        glEnable(GL_MULTISAMPLE);
    } else {
        glDisable(GL_MULTISAMPLE);
    }

    if (MS.sampleShadingEnable) {
        glEnable(GL_SAMPLE_SHADING);
        PFNGLMINSAMPLESHADINGPROC glMinSampleShading = (PFNGLMINSAMPLESHADINGPROC)GetExtension("glMinSampleShading");  // 4.0+
        glMinSampleShading(MS.minSampleShading);
    } else {
        glDisable(GL_SAMPLE_SHADING);
    }

    if (MS.sampleMask > 0) {
        glEnable(GL_SAMPLE_MASK);
        PFNGLSAMPLEMASKIPROC glSampleMaski = (PFNGLSAMPLEMASKIPROC)GetExtension("glSampleMaski");  // 3.2+
        glSampleMaski(0, MS.sampleMask);
    } else {
        glDisable(GL_SAMPLE_MASK);
    }

    if (MS.alphaToCoverageEnable) {
        glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    } else {
        glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    }

    if (MS.alphaToOneEnable) {
        glEnable(GL_SAMPLE_ALPHA_TO_ONE);
    } else {
        glDisable(GL_SAMPLE_ALPHA_TO_ONE);
    }

    // DepthStencilState
    const DepthStencilState &DSS = pipelineCI.depthStencilState;

    if (DSS.depthTestEnable) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }

    glDepthMask(DSS.depthWriteEnable ? GL_TRUE : GL_FALSE);

    glDepthFunc(ToGLCompareOp(DSS.depthCompareOp));

    PFNGLDEPTHBOUNDSEXTPROC glDepthBoundsEXT = (PFNGLDEPTHBOUNDSEXTPROC)GetExtension("glDepthBoundsEXT");  // EXT
    if (glDepthBoundsEXT) {
        if (DSS.depthBoundsTestEnable) {
            glEnable(GL_DEPTH_BOUNDS_TEST_EXT);
            glDepthBoundsEXT(DSS.minDepthBounds, DSS.maxDepthBounds);
        } else {
            glDisable(GL_DEPTH_BOUNDS_TEST_EXT);
        }
    }

    if (DSS.stencilTestEnable) {
        glEnable(GL_STENCIL_TEST);
    } else {
        glDisable(GL_STENCIL_TEST);
    }

    PFNGLSTENCILOPSEPARATEPROC glStencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC)GetExtension("glStencilOpSeparate");          // 2.0+
    PFNGLSTENCILFUNCSEPARATEPROC glStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC)GetExtension("glStencilFuncSeparate");  // 2.0+
    PFNGLSTENCILMASKSEPARATEPROC glStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC)GetExtension("glStencilMaskSeparate");  // 2.0+

    glStencilOpSeparate(GL_FRONT,
                        ToGLStencilCompareOp(DSS.front.failOp),
                        ToGLStencilCompareOp(DSS.front.depthFailOp),
                        ToGLStencilCompareOp(DSS.front.passOp));
    glStencilFuncSeparate(GL_FRONT,
                          ToGLCompareOp(DSS.front.compareOp),
                          DSS.front.reference,
                          DSS.front.compareMask);
    glStencilMaskSeparate(GL_FRONT, DSS.front.writeMask);

    glStencilOpSeparate(GL_BACK,
                        ToGLStencilCompareOp(DSS.back.failOp),
                        ToGLStencilCompareOp(DSS.back.depthFailOp),
                        ToGLStencilCompareOp(DSS.back.passOp));
    glStencilFuncSeparate(GL_BACK,
                          ToGLCompareOp(DSS.back.compareOp),
                          DSS.back.reference,
                          DSS.back.compareMask);
    glStencilMaskSeparate(GL_BACK, DSS.back.writeMask);

    // ColorBlendState
    const ColorBlendState &CBS = pipelineCI.colorBlendState;

    if (CBS.logicOpEnable) {
        glEnable(GL_COLOR_LOGIC_OP);
        glLogicOp(ToGLLogicOp(CBS.logicOp));
    } else {
        glDisable(GL_COLOR_LOGIC_OP);
    }

    for (int i = 0; i < (int)CBS.attachments.size(); i++) {
        const ColorBlendAttachmentState &CBA = CBS.attachments[i];

        PFNGLENABLEIPROC glEnablei = (PFNGLENABLEIPROC)GetExtension("glEnablei");                                                              // 3.0+
        PFNGLDISABLEIPROC glDisablei = (PFNGLDISABLEIPROC)GetExtension("glDisablei");                                                          // 3.0+
        PFNGLBLENDEQUATIONSEPARATEIPROC glBlendEquationSeparatei = (PFNGLBLENDEQUATIONSEPARATEIPROC)GetExtension("glBlendEquationSeparatei");  // 4.0+
        PFNGLBLENDFUNCSEPARATEIPROC glBlendFuncSeparatei = (PFNGLBLENDFUNCSEPARATEIPROC)GetExtension("glBlendFuncSeparatei");                  // 4.0+
        PFNGLCOLORMASKIPROC glColorMaski = (PFNGLCOLORMASKIPROC)GetExtension("glColorMaski");                                                  // 3.0+

        if (CBA.blendEnable) {
            glEnablei(GL_BLEND, i);
        } else {
            glDisablei(GL_BLEND, i);
        }

        glBlendEquationSeparatei(i, ToGLBlendOp(CBA.colorBlendOp), ToGLBlendOp(CBA.alphaBlendOp));

        glBlendFuncSeparatei(i,
                             ToGLBlendFactor(CBA.srcColorBlendFactor),
                             ToGLBlendFactor(CBA.dstColorBlendFactor),
                             ToGLBlendFactor(CBA.srcAlphaBlendFactor),
                             ToGLBlendFactor(CBA.dstAlphaBlendFactor));

        glColorMaski(i,
                     (((uint32_t)CBA.colorWriteMask & (uint32_t)ColorComponentBit::R_BIT) == (uint32_t)ColorComponentBit::R_BIT),
                     (((uint32_t)CBA.colorWriteMask & (uint32_t)ColorComponentBit::G_BIT) == (uint32_t)ColorComponentBit::G_BIT),
                     (((uint32_t)CBA.colorWriteMask & (uint32_t)ColorComponentBit::B_BIT) == (uint32_t)ColorComponentBit::B_BIT),
                     (((uint32_t)CBA.colorWriteMask & (uint32_t)ColorComponentBit::A_BIT) == (uint32_t)ColorComponentBit::A_BIT));
    }
    glBlendColor(CBS.blendConstants[0], CBS.blendConstants[1], CBS.blendConstants[2], CBS.blendConstants[3]);
}

void GraphicsAPI_OpenGL::SetDescriptor(const DescriptorInfo &descriptorInfo) {
    GLuint glResource = (GLuint)(uint64_t)descriptorInfo.resource;
    const GLuint &bindingIndex = descriptorInfo.bindingIndex;
    if (descriptorInfo.type == DescriptorInfo::Type::BUFFER) {
        PFNGLBINDBUFFERRANGEPROC glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)GetExtension("glBindBufferRange");  // 3.0+
        glBindBufferRange(GL_UNIFORM_BUFFER, bindingIndex, glResource, (GLintptr)descriptorInfo.bufferOffset, (GLsizeiptr)descriptorInfo.bufferSize);
    } else if (descriptorInfo.type == DescriptorInfo::Type::IMAGE) {
        glActiveTexture(GL_TEXTURE0 + bindingIndex);
        glBindTexture(GetGLTextureTarget(images[glResource]), glResource);
    } else if (descriptorInfo.type == DescriptorInfo::Type::SAMPLER) {
        PFNGLBINDSAMPLERPROC glBindSampler = (PFNGLBINDSAMPLERPROC)GetExtension("glBindSampler");  // 3.0+
        glBindSampler(bindingIndex, glResource);
    } else {
        std::cout << "ERROR: OPENGL: Unknown Descriptor Type." << std::endl;
    }
}

void GraphicsAPI_OpenGL::UpdateDescriptors() {
}

void GraphicsAPI_OpenGL::SetVertexBuffers(void **vertexBuffers, size_t count) {
    const VertexInputState &vertexInputState = pipelines[setPipeline].vertexInputState;
    for (size_t i = 0; i < count; i++) {
        GLuint glVertexBufferID = (GLuint)(uint64_t)vertexBuffers[i];
        if (buffers[glVertexBufferID].type != BufferCreateInfo::Type::VERTEX) {
            std::cout << "ERROR: OpenGL: Provided buffer is not type: VERTEX." << std::endl;
        }

        glBindBuffer(GL_ARRAY_BUFFER, (GLuint)(uint64_t)vertexBuffers[i]);

        // https://i.redd.it/fyxp5ah06a661.png
        for (const VertexInputBinding &vertexBinding : vertexInputState.bindings) {
            if (vertexBinding.bindingIndex == (uint32_t)i) {
                for (const VertexInputAttribute &vertexAttribute : vertexInputState.attributes) {
                    if (vertexAttribute.bindingIndex == (uint32_t)i) {
                        GLuint attribIndex = vertexAttribute.attribIndex;
                        GLint size = ((GLint)vertexAttribute.vertexType % 4) + 1;
                        GLenum type = (GLenum)vertexAttribute.vertexType >= (GLenum)VertexType::UINT ? GL_UNSIGNED_INT : (GLenum)vertexAttribute.vertexType >= (GLenum)VertexType::INT ? GL_INT
                                                                                                                                                                                       : GL_FLOAT;
                        GLsizei stride = vertexBinding.stride;
                        const void *offset = (const void *)vertexAttribute.offset;
                        glEnableVertexAttribArray(attribIndex);
                        glVertexAttribPointer(attribIndex, size, type, false, stride, offset);
                    }
                }
            }
        }
    }
}

void GraphicsAPI_OpenGL::SetIndexBuffer(void *indexBuffer) {
    GLuint glIndexBufferID = (GLuint)(uint64_t)indexBuffer;
    if (buffers[glIndexBufferID].type != BufferCreateInfo::Type::INDEX) {
        std::cout << "ERROR: OpenGL: Provided buffer is not type: INDEX." << std::endl;
    }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBufferID);
    setIndexBuffer = glIndexBufferID;
}

void GraphicsAPI_OpenGL::DrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC glDrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC)GetExtension("glDrawElementsInstancedBaseVertexBaseInstance");  // 4.2+
    GLenum indexType = buffers[setIndexBuffer].stride == 4 ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
    glDrawElementsInstancedBaseVertexBaseInstance(ToGLTopology(pipelines[setPipeline].inputAssemblyState.topology), indexCount, indexType, nullptr, instanceCount, vertexOffset, firstInstance);
}

void GraphicsAPI_OpenGL::Draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
    PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC glDrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC)GetExtension("glDrawArraysInstancedBaseInstance");  // 4.2+
    glDrawArraysInstancedBaseInstance(ToGLTopology(pipelines[setPipeline].inputAssemblyState.topology), firstVertex, vertexCount, instanceCount, firstInstance);
}

const std::vector<int64_t> GraphicsAPI_OpenGL::GetSupportedColorSwapchainFormats() {
    // https://github.com/KhronosGroup/OpenXR-SDK-Source/blob/f122f9f1fc729e2dc82e12c3ce73efa875182854/src/tests/hello_xr/graphicsplugin_opengl.cpp#L229-L236
    return {
        GL_RGB10_A2,
        GL_RGBA16F,
        // The two below should only be used as a fallback, as they are linear color formats without enough bits for color
        // depth, thus leading to banding.
        GL_RGBA8,
        GL_RGBA8_SNORM,
    };
}
const std::vector<int64_t> GraphicsAPI_OpenGL::GetSupportedDepthSwapchainFormats() {
    return {
        GL_DEPTH_COMPONENT32F,
        GL_DEPTH_COMPONENT32,
        GL_DEPTH_COMPONENT24,
        GL_DEPTH_COMPONENT16};
}
#endif