// Copyright 2023, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OpenXR Tutorial for Khronos Group

#pragma once
#include <HelperFunctions.h>

// Platform headers
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <unknwn.h>
#define XR_USE_GRAPHICS_API_OPENGL
#include <osg/Geode>
#include <sysdep/khronos-glext.h>

#define XR_USE_PLATFORM_WIN32

#if defined(XR_TUTORIAL_USE_D3D11)
#define XR_USE_GRAPHICS_API_D3D11
#endif
#if defined(XR_TUTORIAL_USE_D3D12)
#define XR_USE_GRAPHICS_API_D3D12
#endif
#if defined(XR_TUTORIAL_USE_OPENGL)
#define XR_USE_GRAPHICS_API_OPENGL
#endif
#if defined(XR_TUTORIAL_USE_VULKAN)
#define XR_USE_GRAPHICS_API_VULKAN
#endif
#endif  // _WIN32

#if defined(__linux__) && !defined(__ANDROID__)
#if defined(XR_TUTORIAL_USE_LINUX_XLIB)
#include <X11/Xlib.h>
#define XR_USE_PLATFORM_XLIB
#endif
#if defined(XR_TUTORIAL_USE_LINUX_XCB)
#include <xcb/xcb.h>
#define XR_USE_PLATFORM_XCB
#endif
#if defined(XR_TUTORIAL_USE_LINUX_WAYLAND)
#include <wayland-client.h>
#define XR_USE_PLATFORM_WAYLAND
#endif

#if defined(XR_TUTORIAL_USE_OPENGL)
#define XR_USE_GRAPHICS_API_OPENGL
#endif
#if defined(XR_TUTORIAL_USE_VULKAN)
#define XR_USE_GRAPHICS_API_VULKAN
#endif
#endif  // __linux__

#if defined(__ANDROID__)
#include <android_native_app_glue.h>
#define XR_USE_PLATFORM_ANDROID

#if defined(XR_TUTORIAL_USE_OPENGL_ES)
#define XR_USE_GRAPHICS_API_OPENGL_ES
#endif
#if defined(XR_TUTORIAL_USE_VULKAN)
#define XR_USE_GRAPHICS_API_VULKAN
#endif
#endif  // __ANDROID__

// Graphic APIs headers
#if defined(XR_USE_GRAPHICS_API_D3D11)
#include <d3d11_1.h>
#include <dxgi1_6.h>
#endif

#if defined(XR_USE_GRAPHICS_API_D3D12)
#include <d3d12.h>
#include <dxgi1_6.h>
#endif

#if defined(XR_USE_GRAPHICS_API_OPENGL)
#if defined(XR_USE_PLATFORM_XLIB)
#define OS_LINUX_XLIB 1
#endif
#if defined(XR_USE_PLATFORM_XCB)
#define OS_LINUX_XCB 1
#endif
#if defined(XR_USE_PLATFORM_WAYLAND)
#define OS_LINUX_WAYLAND 1
#endif

// gfxwrapper will redefine these macros
#undef XR_USE_PLATFORM_WIN32
#undef XR_USE_PLATFORM_XLIB
#undef XR_USE_PLATFORM_XCB
#undef XR_USE_PLATFORM_WAYLAND
//#include <gfxwrapper_opengl.h>
#endif

#if defined(XR_USE_GRAPHICS_API_OPENGL_ES)
#include <gfxwrapper_opengl.h>
#endif

#if defined(XR_USE_GRAPHICS_API_VULKAN)
#include <vulkan/vulkan.h>
#endif

// OpenXR Helper
#include <OpenXRHelper.h>

enum GraphicsAPI_Type : uint8_t {
    UNKNOWN,
    D3D11,
    D3D12,
    OPENGL,
    OPENGL_ES,
    VULKAN
};

bool CheckGraphicsAPI_TypeIsValidForPlatform(GraphicsAPI_Type type);

const char* GetGraphicsAPIInstanceExtensionString(GraphicsAPI_Type type);

class GraphicsAPI {
public:
// Pipeline Helpers
#pragma region Pipeline Helpers
    enum class SwapchainType : uint8_t {
        COLOR,
        DEPTH
    };
    enum class VertexType : uint8_t {
        FLOAT,
        VEC2,
        VEC3,
        VEC4,
        INT,
        IVEC2,
        IVEC3,
        IVEC4,
        UINT,
        UVEC2,
        UVEC3,
        UVEC4
    };
    enum class PrimitiveTopology : uint8_t {
        POINT_LIST = 0,
        LINE_LIST = 1,
        LINE_STRIP = 2,
        TRIANGLE_LIST = 3,
        TRIANGLE_STRIP = 4,
        TRIANGLE_FAN = 5,
    };
    enum class PolygonMode : uint8_t {
        FILL = 0,
        LINE = 1,
        POINT = 2,
    };
    enum class CullMode : uint8_t {
        NONE = 0,
        FRONT = 1,
        BACK = 2,
        FRONT_AND_BACK = 3
    };
    enum class FrontFace : uint8_t {
        COUNTER_CLOCKWISE = 0,
        CLOCKWISE = 1,
    };
    enum class CompareOp : uint8_t {
        NEVER = 0,
        LESS = 1,
        EQUAL = 2,
        LESS_OR_EQUAL = 3,
        GREATER = 4,
        NOT_EQUAL = 5,
        GREATER_OR_EQUAL = 6,
        ALWAYS = 7,
    };
    enum class StencilOp : uint8_t {
        KEEP = 0,
        ZERO = 1,
        REPLACE = 2,
        INCREMENT_AND_CLAMP = 3,
        DECREMENT_AND_CLAMP = 4,
        INVERT = 5,
        INCREMENT_AND_WRAP = 6,
        DECREMENT_AND_WRAP = 7
    };
    struct StencilOpState {
        StencilOp failOp;
        StencilOp passOp;
        StencilOp depthFailOp;
        CompareOp compareOp;
        uint32_t compareMask;
        uint32_t writeMask;
        uint32_t reference;
    };
    enum class BlendFactor : uint8_t {
        ZERO = 0,
        ONE = 1,
        SRC_COLOR = 2,
        ONE_MINUS_SRC_COLOR = 3,
        DST_COLOR = 4,
        ONE_MINUS_DST_COLOR = 5,
        SRC_ALPHA = 6,
        ONE_MINUS_SRC_ALPHA = 7,
        DST_ALPHA = 8,
        ONE_MINUS_DST_ALPHA = 9,
    };
    enum class BlendOp : uint8_t {
        ADD = 0,
        SUBTRACT = 1,
        REVERSE_SUBTRACT = 2,
        MIN = 3,
        MAX = 4,
    };
    enum class ColorComponentBit : uint8_t {
        R_BIT = 0x00000001,
        G_BIT = 0x00000002,
        B_BIT = 0x00000004,
        A_BIT = 0x00000008,
    };
    struct ColorBlendAttachmentState {
        bool blendEnable;
        BlendFactor srcColorBlendFactor;
        BlendFactor dstColorBlendFactor;
        BlendOp colorBlendOp;
        BlendFactor srcAlphaBlendFactor;
        BlendFactor dstAlphaBlendFactor;
        BlendOp alphaBlendOp;
        ColorComponentBit colorWriteMask;
    };
    enum class LogicOp : uint8_t {
        CLEAR = 0,
        AND = 1,
        AND_REVERSE = 2,
        COPY = 3,
        AND_INVERTED = 4,
        NO_OP = 5,
        XOR = 6,
        OR = 7,
        NOR = 8,
        EQUIVALENT = 9,
        INVERT = 10,
        OR_REVERSE = 11,
        COPY_INVERTED = 12,
        OR_INVERTED = 13,
        NAND = 14,
        SET = 15
    };
#pragma endregion

    struct ShaderCreateInfo {
        enum class Type : uint8_t {
            VERTEX,
            TESSELLATION_CONTROL,
            TESSELLATION_EVALUATION,
            GEOMETRY,
            FRAGMENT,
            COMPUTE
        } type;
        const char* sourceData;
        size_t sourceSize;
    };
    struct VertexInputAttribute {
        uint32_t attribIndex;   // layout(location = X)
        uint32_t bindingIndex;  // Which buffer to use when bound for draws.
        VertexType vertexType;
        size_t offset;
        const char* semanticName;
    };
    typedef std::vector<VertexInputAttribute> VertexInputAttributes;
    struct VertexInputBinding {
        uint32_t bindingIndex;  // Which buffer to use when bound for draws.
        size_t offset;
        size_t stride;
    };
    typedef std::vector<VertexInputBinding> VertexInputBindings;
    struct VertexInputState {
        VertexInputAttributes attributes;
        VertexInputBindings bindings;
    };
    struct InputAssemblyState {
        PrimitiveTopology topology;
        bool primitiveRestartEnable;
    };
    struct RasterisationState {
        bool depthClampEnable;
        bool rasteriserDiscardEnable;
        PolygonMode polygonMode;
        CullMode cullMode;
        FrontFace frontFace;
        bool depthBiasEnable;
        float depthBiasConstantFactor;
        float depthBiasClamp;
        float depthBiasSlopeFactor;
        float lineWidth;
    };
    struct MultisampleState {
        uint32_t rasterisationSamples;
        bool sampleShadingEnable;
        float minSampleShading;
        uint32_t sampleMask;
        bool alphaToCoverageEnable;
        bool alphaToOneEnable;
    };
    struct DepthStencilState {
        bool depthTestEnable;
        bool depthWriteEnable;
        CompareOp depthCompareOp;
        bool depthBoundsTestEnable;
        bool stencilTestEnable;
        StencilOpState front;
        StencilOpState back;
        float minDepthBounds;
        float maxDepthBounds;
    };
    struct ColorBlendState {
        bool logicOpEnable;
        LogicOp logicOp;
        std::vector<ColorBlendAttachmentState> attachments;
        float blendConstants[4];
    };

    struct DescriptorInfo {
        uint32_t bindingIndex;
        void* resource;
        enum class Type : uint8_t {
            BUFFER,
            IMAGE,
            SAMPLER
        } type;
        enum class Stage : uint8_t {
            VERTEX,
            TESSELLATION_CONTROL,
            TESSELLATION_EVALUATION,
            GEOMETRY,
            FRAGMENT,
            COMPUTE
        } stage;
        bool readWrite;
        size_t bufferOffset;
        size_t bufferSize;
    };
    struct PipelineCreateInfo {
        std::vector<void*> shaders;
        VertexInputState vertexInputState;
        InputAssemblyState inputAssemblyState;
        RasterisationState rasterisationState;
        MultisampleState multisampleState;
        DepthStencilState depthStencilState;
        ColorBlendState colorBlendState;
        std::vector<int64_t> colorFormats;
        int64_t depthFormat;
        std::vector<DescriptorInfo> layout;
    };

    struct SwapchainCreateInfo {
        uint32_t width;
        uint32_t height;
        uint32_t count;
        void* windowHandle;
        int64_t format;
        bool vsync;
    };

    struct BufferCreateInfo {
        enum class Type : uint8_t {
            VERTEX,
            INDEX,
            UNIFORM,
        } type;
        size_t stride;
        size_t size;
        void* data;
    };

    struct ImageCreateInfo {
        uint32_t dimension;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
        uint32_t mipLevels;
        uint32_t arrayLayers;
        uint32_t sampleCount;
        int64_t format;
        bool cubemap;
        bool colorAttachment;
        bool depthAttachment;
        bool sampled;
    };

    struct ImageViewCreateInfo {
        void* image;
        enum class Type : uint8_t {
            RTV,
            DSV,
            SRV,
            UAV
        } type;
        enum class View : uint8_t {
            TYPE_1D,
            TYPE_2D,
            TYPE_3D,
            TYPE_CUBE,
            TYPE_1D_ARRAY,
            TYPE_2D_ARRAY,
            TYPE_CUBE_ARRAY,
        } view;
        int64_t format;
        enum class Aspect : uint8_t {
            COLOR_BIT = 0x01,
            DEPTH_BIT = 0x02,
            STENCIL_BIT = 0x04
        } aspect;
        uint32_t baseMipLevel;
        uint32_t levelCount;
        uint32_t baseArrayLayer;
        uint32_t layerCount;
    };

    struct SamplerCreateInfo {
        enum class Filter : uint8_t {
            NEAREST,
            LINEAR
        } magFilter,
            minFilter;
        enum class MipmapMode : uint8_t {
            NEAREST,
            LINEAR,
            NOOP
        } mipmapMode;
        enum class AddressMode : uint8_t {
            REPEAT,
            MIRRORED_REPEAT,
            CLAMP_TO_EDGE,
            CLAMP_TO_BORDER,
            MIRROR_CLAMP_TO_EDGE
        } addressModeS,
            addressModeT, addressModeR;
        float mipLodBias;
        bool compareEnable;
        CompareOp compareOp;
        float minLod;
        float maxLod;
        float borderColor[4];
    };

    struct Viewport {
        float x;
        float y;
        float width;
        float height;
        float minDepth;
        float maxDepth;
    };
    struct Offset2D {
        int32_t x;
        int32_t y;
    };
    struct Extent2D {
        uint32_t width;
        uint32_t height;
    };
    struct Rect2D {
        Offset2D offset;
        Extent2D extent;
    };

public:
    virtual ~GraphicsAPI() = default;

    int64_t SelectColorSwapchainFormat(const std::vector<int64_t>& formats);
    int64_t SelectDepthSwapchainFormat(const std::vector<int64_t>& formats);

    virtual void* CreateDesktopSwapchain(const SwapchainCreateInfo& swapchainCI) = 0;
    virtual void DestroyDesktopSwapchain(void*& swapchain) = 0;
    virtual void* GetDesktopSwapchainImage(void* swapchain, uint32_t index) = 0;
    virtual void AcquireDesktopSwapchanImage(void* swapchain, uint32_t& index) = 0;
    virtual void PresentDesktopSwapchainImage(void* swapchain, uint32_t index) = 0;

    virtual int64_t GetDepthFormat() = 0;

    virtual void* GetGraphicsBinding() = 0;
    virtual XrSwapchainImageBaseHeader* AllocateSwapchainImageData(XrSwapchain swapchain, SwapchainType type, uint32_t count) = 0;
    virtual void FreeSwapchainImageData(XrSwapchain swapchain) = 0;
    virtual XrSwapchainImageBaseHeader* GetSwapchainImageData(XrSwapchain swapchain, uint32_t index) = 0;
    virtual void* GetSwapchainImage(XrSwapchain swapchain, uint32_t index) = 0;

    virtual void* CreateImage(const ImageCreateInfo& imageCI) = 0;
    virtual void DestroyImage(void*& image) = 0;

    virtual void* CreateImageView(const ImageViewCreateInfo& imageViewCI) = 0;
    virtual void DestroyImageView(void*& imageView) = 0;

    virtual void* CreateSampler(const SamplerCreateInfo& samplerCI) = 0;
    virtual void DestroySampler(void*& sampler) = 0;

    virtual void* CreateBuffer(const BufferCreateInfo& bufferCI) = 0;
    virtual void DestroyBuffer(void*& buffer) {}

    virtual void* CreateShader(const ShaderCreateInfo& shaderCI) = 0;
    virtual void DestroyShader(void*& shader) = 0;

    virtual void* CreatePipeline(const PipelineCreateInfo& pipelineCI) = 0;
    virtual void DestroyPipeline(void*& pipeline) = 0;

    virtual void BeginRendering() = 0;
    virtual void EndRendering() = 0;

    virtual void SetBufferData(void* buffer, size_t offset, size_t size, void* data) = 0;

    virtual void ClearColor(void* imageView, float r, float g, float b, float a) = 0;
    virtual void ClearDepth(void* imageView, float d) = 0;

    virtual void SetRenderAttachments(void** colorViews, size_t colorViewCount, void* depthStencilView, uint32_t width, uint32_t height, void* pipeline) = 0;
    virtual void SetViewports(Viewport* viewports, size_t count) = 0;
    virtual void SetScissors(Rect2D* scissors, size_t count) = 0;

    virtual void SetPipeline(void* pipeline) = 0;
    virtual void SetDescriptor(const DescriptorInfo& descriptorInfo) = 0;
    virtual void UpdateDescriptors() = 0;
    virtual void SetVertexBuffers(void** vertexBuffers, size_t count) = 0;
    virtual void SetIndexBuffer(void* indexBuffer) = 0;
    virtual void DrawIndexed(uint32_t indexCount, uint32_t instanceCount = 1, uint32_t firstIndex = 0, int32_t vertexOffset = 0, uint32_t firstInstance = 0) = 0;
    virtual void Draw(uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t firstVertex = 0, uint32_t firstInstance = 0) = 0;

protected:
    virtual const std::vector<int64_t> GetSupportedColorSwapchainFormats() = 0;
    virtual const std::vector<int64_t> GetSupportedDepthSwapchainFormats() = 0;
    bool debugAPI = false;
};
