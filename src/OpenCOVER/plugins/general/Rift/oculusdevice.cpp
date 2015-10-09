/*
 * oculusdevice.cpp
 *
 *  Created on: Jul 03, 2013
 *      Author: Bjorn Blissing
 *
 *  MSAA support added: Sep 17, 2015
 *      Author: Chris Denham
 */

#include "oculusdevice.h"

#ifdef _WIN32
	#include <Windows.h>
#endif

#include <osg/Geometry>
#include <osgViewer/Renderer>
#include <osgViewer/GraphicsWindow>
#include <cover/input/input.h>
#include <cover/input/dev/rift/RIFTDriver.h>

#ifndef GL_TEXTURE_MAX_LEVEL
	#define GL_TEXTURE_MAX_LEVEL 0x813D
#endif

static const OSG_GLExtensions* getGLExtensions(const osg::State& state)
{
#if(OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
	return state.get<osg::GLExtensions>();
#else
	return osg::FBOExtensions::instance(state.getContextID(), true);
#endif
}

static const OSG_Texture_Extensions* getTextureExtensions(const osg::State& state)
{
#if(OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
	return state.get<osg::GLExtensions>();
#else
	return osg::Texture::getExtensions(state.getContextID(), true);
#endif
}

static osg::FrameBufferObject* getFrameBufferObject(osg::RenderInfo& renderInfo)
{
	osg::Camera* camera = renderInfo.getCurrentCamera();
	osgViewer::Renderer* camRenderer = (dynamic_cast<osgViewer::Renderer*>(camera->getRenderer()));

	if (camRenderer != nullptr)
	{
		osgUtil::SceneView* sceneView = camRenderer->getSceneView(0);

		if (sceneView != nullptr)
		{
			osgUtil::RenderStage* renderStage = sceneView->getRenderStage();

			if (renderStage != nullptr)
			{
				return renderStage->getFrameBufferObject();
			}
		}
	}

	return nullptr;
}

void OculusPreDrawCallback::operator()(osg::RenderInfo& renderInfo) const
{
	m_textureBuffer->onPreRender(renderInfo);
}

void OculusPostDrawCallback::operator()(osg::RenderInfo& renderInfo) const
{
	m_textureBuffer->onPostRender(renderInfo);
}

/* Public functions */
OculusTextureBuffer::OculusTextureBuffer(const ovrHmd& hmd, osg::ref_ptr<osg::State> state, const ovrSizei& size, int samples) : m_hmdDevice(hmd),
	m_textureSet(nullptr),
	m_colorBuffer(nullptr),
	m_depthBuffer(nullptr),
	m_textureSize(osg::Vec2i(size.w, size.h)),
	m_Oculus_FBO(0),
	m_samples(samples)
{
    
	m_MSAA_FBO = 0; // framebuffer for MSAA texture
	m_MSAA_ColorTex = 0; // color texture for MSAA
	m_MSAA_DepthTex = 0; // depth texture for MSAA
	if (samples == 0)
	{
		setup(*state);
	}
	else
	{
		setupMSAA(*state);
	}

}

void OculusTextureBuffer::setup(osg::State& state)
{
	if (ovr_CreateSwapTextureSetGL(m_hmdDevice, GL_SRGB8_ALPHA8, textureWidth(), textureHeight(), &m_textureSet) == ovrSuccess)
	{
		// Assign textures to OSG textures
		for (int i = 0; i < m_textureSet->TextureCount; ++i)
		{
			ovrGLTexture* tex = reinterpret_cast<ovrGLTexture*>(&m_textureSet->Textures[i]);

			GLuint handle = tex->OGL.TexId;

			osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();

			osg::ref_ptr<osg::Texture::TextureObject> textureObject = new osg::Texture::TextureObject(texture.get(), handle, GL_TEXTURE_2D);

			textureObject->setAllocated(true);

			texture->setTextureObject(state.getContextID(), textureObject.get());

			texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
			texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
			texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
			texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);

			texture->setTextureSize(tex->Texture.Header.TextureSize.w, tex->Texture.Header.TextureSize.h);
			texture->setSourceFormat(GL_SRGB8_ALPHA8);

			// Set the current texture to point to the texture with the index (which will be advanced before drawing)
			if (i == (m_textureSet->CurrentIndex + 1))
			{
				m_colorBuffer = texture;
			}
		}

		osg::notify(osg::DEBUG_INFO) << "Successfully created the swap texture set!" << std::endl;
	}
	else
	{
		osg::notify(osg::WARN) << "Warning: Unable to create swap texture set! " << std::endl;
		return;
	}

	m_depthBuffer = new osg::Texture2D();
	m_depthBuffer->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
	m_depthBuffer->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
	m_depthBuffer->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
	m_depthBuffer->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
	m_depthBuffer->setSourceFormat(GL_DEPTH_COMPONENT);
	m_depthBuffer->setSourceType(GL_UNSIGNED_INT);
	m_depthBuffer->setInternalFormat(GL_DEPTH_COMPONENT24);
	m_depthBuffer->setTextureWidth(textureWidth());
	m_depthBuffer->setTextureHeight(textureHeight());
	m_depthBuffer->setBorderWidth(0);

}


void OculusTextureBuffer::setupMSAA(osg::State& state)
{
	const OSG_GLExtensions* fbo_ext = getGLExtensions(state);

	if (ovr_CreateSwapTextureSetGL(m_hmdDevice, GL_SRGB8_ALPHA8, m_textureSize.x(), m_textureSize.y(), &m_textureSet) == ovrSuccess)
	{
		for (int i = 0; i < m_textureSet->TextureCount; ++i)
		{
			ovrGLTexture* tex = reinterpret_cast<ovrGLTexture*>(&m_textureSet->Textures[i]);
			glBindTexture(GL_TEXTURE_2D, tex->OGL.TexId);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		}
	}

	// Create an FBO for output to Oculus swap texture set.
	fbo_ext->glGenFramebuffers(1, &m_Oculus_FBO);

	// Create an FBO for primary render target.
	fbo_ext->glGenFramebuffers(1, &m_MSAA_FBO);

	// We don't want to support MIPMAP so, ensure only level 0 is allowed.
	const int maxTextureLevel = 0;

	const OSG_Texture_Extensions* extensions = getTextureExtensions(state);

	// Create MSAA colour buffer
	glGenTextures(1, &m_MSAA_ColorTex);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_MSAA_ColorTex);
	extensions->glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, m_samples, GL_RGBA, m_textureSize.x(), m_textureSize.y(), false);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER_ARB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MAX_LEVEL, maxTextureLevel);

	// Create MSAA depth buffer
	glGenTextures(1, &m_MSAA_DepthTex);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, m_MSAA_DepthTex);
	extensions->glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, m_samples, GL_DEPTH_COMPONENT, m_textureSize.x(), m_textureSize.y(), false);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_MAX_LEVEL, maxTextureLevel);
}

void OculusTextureBuffer::onPreRender(osg::RenderInfo& renderInfo)
{
	advanceIndex();

	osg::State& state = *renderInfo.getState();
	const OSG_GLExtensions* fbo_ext = getGLExtensions(state);

	if (m_samples == 0)
	{
		osg::FrameBufferObject* fbo = getFrameBufferObject(renderInfo);

		if (fbo == nullptr)
		{
			return;
		}

		const unsigned int ctx = state.getContextID();
		ovrGLTexture* tex = reinterpret_cast<ovrGLTexture*>(&m_textureSet->Textures[m_textureSet->CurrentIndex]);
		fbo_ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, fbo->getHandle(ctx));
		fbo_ext->glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex->OGL.TexId, 0);
		fbo_ext->glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_depthBuffer->getTextureObject(state.getContextID())->id(), 0);
	}
	else
	{
		fbo_ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, m_MSAA_FBO);
		fbo_ext->glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D_MULTISAMPLE, m_MSAA_ColorTex, 0);
		fbo_ext->glFramebufferTexture2D(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D_MULTISAMPLE, m_MSAA_DepthTex, 0);
	}

}
void OculusTextureBuffer::onPostRender(osg::RenderInfo& renderInfo)
{
	if (m_samples == 0)
	{
		// Nothing to do here if MSAA not being used.
		return;
	}

	osg::State& state = *renderInfo.getState();
	const OSG_GLExtensions* fbo_ext = getGLExtensions(state);

	fbo_ext->glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, m_MSAA_FBO);
	fbo_ext->glFramebufferTexture2D(GL_READ_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D_MULTISAMPLE, m_MSAA_ColorTex, 0);
	fbo_ext->glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);

	ovrGLTexture* tex = reinterpret_cast<ovrGLTexture*>(&m_textureSet->Textures[m_textureSet->CurrentIndex]);
	fbo_ext->glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, m_Oculus_FBO);
	fbo_ext->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex->OGL.TexId, 0);
	fbo_ext->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);

	int w = m_textureSize.x();
	int h = m_textureSize.y();
	fbo_ext->glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	fbo_ext->glBindFramebuffer(GL_FRAMEBUFFER_EXT, 0);

}

void OculusTextureBuffer::destroy()
{
	ovr_DestroySwapTextureSet(m_hmdDevice, m_textureSet);
}

OculusMirrorTexture::OculusMirrorTexture(const ovrHmd& hmd, osg::ref_ptr<osg::State> state, int width, int height) : m_hmdDevice(hmd), m_texture(nullptr)
{
	const OSG_GLExtensions* fbo_ext = getGLExtensions(*state);
	if (ovr_CreateMirrorTextureGL(m_hmdDevice, GL_SRGB8_ALPHA8, width, height, reinterpret_cast<ovrTexture**>(&m_texture)) == ovrSuccess)
    {
	// Configure the mirror read buffer
	fbo_ext->glGenFramebuffers(1, &m_mirrorFBO);
	fbo_ext->glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, m_mirrorFBO);
	fbo_ext->glFramebufferTexture2D(GL_READ_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_texture->OGL.TexId, 0);
	fbo_ext->glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, 0);
	fbo_ext->glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, 0);
    }
    else
    {
        ovrErrorInfo ei;
        ovr_GetLastErrorInfo(&ei);
        fprintf(stderr,"%s\n",ei.ErrorString);
    }
}

void OculusMirrorTexture::blitTexture(osg::GraphicsContext* gc)
{
	const OSG_GLExtensions* fbo_ext = getGLExtensions(*(gc->getState()));
	// Blit mirror texture to back buffer
	fbo_ext->glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, m_mirrorFBO);
	fbo_ext->glBindFramebuffer(GL_DRAW_FRAMEBUFFER_EXT, 0);
	GLint w = m_texture->OGL.Header.TextureSize.w;
	GLint h = m_texture->OGL.Header.TextureSize.h;
	fbo_ext->glBlitFramebuffer(0, h, w, 0,
							   0, 0, w, h,
							   GL_COLOR_BUFFER_BIT, GL_NEAREST);
	fbo_ext->glBindFramebuffer(GL_READ_FRAMEBUFFER_EXT, 0);
}

void OculusMirrorTexture::destroy(const OSG_GLExtensions* fbo_ext)
{
	if (fbo_ext)
	{
		fbo_ext->glDeleteFramebuffers(1, &m_mirrorFBO);
	}

	ovr_DestroyMirrorTexture(m_hmdDevice, reinterpret_cast<ovrTexture*>(m_texture));
}

/* Public functions */
OculusDevice::OculusDevice(float nearClip, float farClip, const float pixelsPerDisplayPixel, const float worldUnitsPerMetre, const int samples) :
	m_hmdDevice(nullptr),
	m_hmdDesc(),
	m_pixelsPerDisplayPixel(pixelsPerDisplayPixel),
	m_worldUnitsPerMetre(worldUnitsPerMetre),
	m_mirrorTexture(nullptr),
	m_position(osg::Vec3(0.0f, 0.0f, 0.0f)),
	m_orientation(osg::Quat(0.0f, 0.0f, 0.0f, 1.0f)),
	m_nearClip(nearClip), m_farClip(farClip),
	m_samples(samples),
    m_initialized(false)
{
	for (int i = 0; i < 2; i++)
	{
		m_textureBuffer[i] = nullptr;
	}

	ovrGraphicsLuid luid;

	trySetProcessAsHighPriority();
    opencover::InputDevice *riftInput = opencover::Input::instance()->getDevice("RiftDevice");
    if(riftInput != NULL)
    {
        trackingDriver = (RIFTDriver *)riftInput;
    }

	if (ovr_Initialize(nullptr) != ovrSuccess)
	{
		osg::notify(osg::WARN) << "Warning: Unable to initialize the Oculus library!" << std::endl;
		return;
	}

	// Get first available HMD
	ovrResult result = ovr_Create(&m_hmdDevice, &luid);

	if (result != ovrSuccess)
	{
		osg::notify(osg::WARN) << "Warning: No device could be found." << std::endl;
		return;
	}

	// Get HMD description
	m_hmdDesc = ovr_GetHmdDesc(m_hmdDevice);

	// Print information about device
	printHMDDebugInfo();
}

void OculusDevice::createRenderBuffers(osg::ref_ptr<osg::State> state)
{
	// Compute recommended render texture size
	if (m_pixelsPerDisplayPixel > 1.0f)
	{
		osg::notify(osg::WARN) << "Warning: Pixel per display pixel is set to a value higher than 1.0." << std::endl;
	}

	for (int i = 0; i < 2; i++)
	{
		ovrSizei recommenedTextureSize = ovr_GetFovTextureSize(m_hmdDevice, (ovrEyeType)i, m_hmdDesc.DefaultEyeFov[i], m_pixelsPerDisplayPixel);
		m_textureBuffer[i] = new OculusTextureBuffer(m_hmdDevice, state, recommenedTextureSize, m_samples);
	}

	int width = screenResolutionWidth() / 2;
	int height = screenResolutionHeight() / 2;
	m_mirrorTexture = new OculusMirrorTexture(m_hmdDevice, state, width, height);
}

void OculusDevice::init()
{
    if(!m_initialized)
    {
        m_initialized=true;
        initializeEyeRenderDesc();

        calculateEyeAdjustment();

        calculateProjectionMatrices();

        unsigned int hmdCaps = 0;
        // Set render capabilities
        ovr_SetEnabledCaps(m_hmdDevice, hmdCaps);

        // Start the sensor which provides the Rift's pose and motion.
        ovr_ConfigureTracking(m_hmdDevice, ovrTrackingCap_Orientation |
            ovrTrackingCap_MagYawCorrection |
            ovrTrackingCap_Position, 0);

        // Setup layers
        setupLayers();

        // Reset perf hud
        ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_Off);
    }
}

unsigned int OculusDevice::screenResolutionWidth() const
{
	return  m_hmdDesc.Resolution.w;
}

unsigned int OculusDevice::screenResolutionHeight() const
{
	return  m_hmdDesc.Resolution.h;
}

osg::Matrix OculusDevice::projectionMatrixCenter() const
{
	osg::Matrix projectionMatrixCenter;
	projectionMatrixCenter = m_leftEyeProjectionMatrix.operator*(0.5) + m_rightEyeProjectionMatrix.operator*(0.5);
	return projectionMatrixCenter;
}

osg::Matrix OculusDevice::projectionMatrixLeft() const
{
	return m_leftEyeProjectionMatrix;
}

osg::Matrix OculusDevice::projectionMatrixRight() const
{
	return m_rightEyeProjectionMatrix;
}

osg::Matrix OculusDevice::projectionOffsetMatrixLeft() const
{
	osg::Matrix projectionOffsetMatrix;
	float offset = m_leftEyeProjectionMatrix(2, 0);
	projectionOffsetMatrix.makeTranslate(osg::Vec3(-offset, 0.0, 0.0));
	return projectionOffsetMatrix;
}

osg::Matrix OculusDevice::projectionOffsetMatrixRight() const
{
	osg::Matrix projectionOffsetMatrix;
	float offset = m_rightEyeProjectionMatrix(2, 0);
	projectionOffsetMatrix.makeTranslate(osg::Vec3(-offset, 0.0, 0.0));
	return projectionOffsetMatrix;
}

osg::Matrix OculusDevice::viewMatrixLeft() const
{
	osg::Matrix viewMatrix;
	viewMatrix.makeTranslate(-m_leftEyeAdjust);
	return viewMatrix;
}

osg::Matrix OculusDevice::viewMatrixRight() const
{
	osg::Matrix viewMatrix;
	viewMatrix.makeTranslate(-m_rightEyeAdjust);
	return viewMatrix;
}

void OculusDevice::resetSensorOrientation() const
{
	ovr_RecenterPose(m_hmdDevice);
}

void OculusDevice::updatePose(unsigned int frameIndex)
{
	// Ask the API for the times when this frame is expected to be displayed.
	m_frameTiming = ovr_GetFrameTiming(m_hmdDevice, frameIndex);

	m_viewOffset[0] = m_eyeRenderDesc[0].HmdToEyeViewOffset;
	m_viewOffset[1] = m_eyeRenderDesc[1].HmdToEyeViewOffset;

	// Query the HMD for the current tracking state.
	ovrTrackingState ts = ovr_GetTrackingState(m_hmdDevice, m_frameTiming.DisplayMidpointSeconds);
	ovr_CalcEyePoses(ts.HeadPose.ThePose, m_viewOffset, m_eyeRenderPose);
	ovrPoseStatef headpose = ts.HeadPose;
	ovrPosef pose = headpose.ThePose;
	m_position.set(-pose.Position.x, -pose.Position.y, -pose.Position.z);
	m_position *= m_worldUnitsPerMetre;
	m_orientation.set(pose.Orientation.x, pose.Orientation.y, pose.Orientation.z, -pose.Orientation.w);
}

osg::Camera* OculusDevice::createRTTCamera(OculusDevice::Eye eye, osg::Transform::ReferenceFrame referenceFrame, const osg::Vec4& clearColor, osg::GraphicsContext* gc) const
{
	OculusTextureBuffer* buffer = m_textureBuffer[eye];

	osg::Camera::RenderTargetImplementation target = osg::Camera::FRAME_BUFFER_OBJECT;

	if (m_samples != 0)
	{
		// If we are using MSAA, we don't want OSG doing anything regarding FBO
		// setup and selection because this is handled completely by 'SetupMSAA'
		// and by pre and post render callbacks. So setting target to FRAME_BUFFER
		// here to prevent OSG from undoing the MSAA buffer configuration.
		// Note that we also implicity avoid the camera buffer attachments below
		// when MSAA is enabled because we don't want OSG to affect the texture
		// bindings handled by the pre and post render callbacks.
		target = osg::Camera::FRAME_BUFFER;
	}

	osg::ref_ptr<osg::Camera> camera = new osg::Camera();
	camera->setClearColor(clearColor);
	camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	camera->setRenderTargetImplementation(target);
	camera->setRenderOrder(osg::Camera::PRE_RENDER, eye);
	camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
	camera->setAllowEventFocus(false);
	camera->setReferenceFrame(referenceFrame);
	camera->setViewport(0, 0, buffer->textureWidth(), buffer->textureHeight());
	camera->setGraphicsContext(gc);

	if (buffer->colorBuffer())
	{
		camera->attach(osg::Camera::COLOR_BUFFER, buffer->colorBuffer());
	}

	if (buffer->depthBuffer())
	{
		camera->attach(osg::Camera::DEPTH_BUFFER, buffer->depthBuffer());
	}

	camera->setPreDrawCallback(new OculusPreDrawCallback(camera.get(), buffer));
	camera->setFinalDrawCallback(new OculusPostDrawCallback(camera.get(), buffer));

	return camera.release();
}

bool OculusDevice::submitFrame(unsigned int frameIndex)
{
	m_layerEyeFov.ColorTexture[0] = m_textureBuffer[0]->textureSet();
	m_layerEyeFov.ColorTexture[1] = m_textureBuffer[1]->textureSet();

	// Set render pose
	m_layerEyeFov.RenderPose[0] = m_eyeRenderPose[0];
	m_layerEyeFov.RenderPose[1] = m_eyeRenderPose[1];

	ovrLayerHeader* layers = &m_layerEyeFov.Header;
	ovrViewScaleDesc viewScale;
	viewScale.HmdToEyeViewOffset[0] = m_viewOffset[0];
	viewScale.HmdToEyeViewOffset[1] = m_viewOffset[1];
	viewScale.HmdSpaceToWorldScaleInMeters = m_worldUnitsPerMetre;
	ovrResult result = ovr_SubmitFrame(m_hmdDevice, frameIndex, &viewScale, &layers, 1);
	return result == ovrSuccess;
}

void OculusDevice::blitMirrorTexture(osg::GraphicsContext* gc)
{
	m_mirrorTexture->blitTexture(gc);
}

void OculusDevice::setPerfHudMode(int mode)
{
	if (mode == 0) { ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_Off); }

	if (mode == 1) { ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_LatencyTiming); }

	if (mode == 2) { ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_RenderTiming); }

	if (mode == 3) { ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_PerfHeadroom); }

	if (mode == 4) { ovr_SetInt(m_hmdDevice, "PerfHudMode", (int)ovrPerfHud_VersionInfo); }
}

void OculusDevice::setPositionalTrackingState(bool state)
{
	unsigned sensorCaps = 0;
	sensorCaps = ovrTrackingCap_Orientation | ovrTrackingCap_MagYawCorrection;

	if (state)
	{
		sensorCaps |= ovrTrackingCap_Position;
	}

	ovr_ConfigureTracking(m_hmdDevice, sensorCaps, 0);
}

osg::GraphicsContext::Traits* OculusDevice::graphicsContextTraits() const
{
	// Create screen with match the Oculus Rift resolution
	osg::GraphicsContext::WindowingSystemInterface* wsi = osg::GraphicsContext::getWindowingSystemInterface();

	if (!wsi)
	{
		osg::notify(osg::NOTICE) << "Error, no WindowSystemInterface available, cannot create windows." << std::endl;
		return 0;
	}

	// Get the screen identifiers set in environment variable DISPLAY
	osg::GraphicsContext::ScreenIdentifier si;
	si.readDISPLAY();

	// If displayNum has not been set, reset it to 0.
	if (si.displayNum < 0)
	{
		si.displayNum = 0;
		osg::notify(osg::INFO) << "Couldn't get display number, setting to 0" << std::endl;
	}

	// If screenNum has not been set, reset it to 0.
	if (si.screenNum < 0)
	{
		si.screenNum = 0;
		osg::notify(osg::INFO) << "Couldn't get screen number, setting to 0" << std::endl;
	}

	unsigned int width, height;
	wsi->getScreenResolution(si, width, height);

	osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
	traits->hostName = si.hostName;
	traits->screenNum = si.screenNum;
	traits->displayNum = si.displayNum;
	traits->windowDecoration = true;
	traits->x = 50;
	traits->y = 50;
	traits->width = screenResolutionWidth() / 2;
	traits->height = screenResolutionHeight() / 2;
	traits->doubleBuffer = true;
	traits->sharedContext = nullptr;
	traits->vsync = false; // VSync should always be disabled for Oculus Rift applications, the SDK compositor handles the swap

	return traits.release();
}

/* Protected functions */
OculusDevice::~OculusDevice()
{
	// Delete mirror texture
	m_mirrorTexture->destroy();

	// Delete texture and depth buffers
	for (int i = 0; i < 2; i++)
	{
		m_textureBuffer[i]->destroy();
	}

	ovr_Destroy(m_hmdDevice);
	ovr_Shutdown();
}

void OculusDevice::printHMDDebugInfo()
{
	osg::notify(osg::ALWAYS) << "Product:         " << m_hmdDesc.ProductName << std::endl;
	osg::notify(osg::ALWAYS) << "Manufacturer:    " << m_hmdDesc.Manufacturer << std::endl;
	osg::notify(osg::ALWAYS) << "VendorId:        " << m_hmdDesc.VendorId << std::endl;
	osg::notify(osg::ALWAYS) << "ProductId:       " << m_hmdDesc.ProductId << std::endl;
	osg::notify(osg::ALWAYS) << "SerialNumber:    " << m_hmdDesc.SerialNumber << std::endl;
	osg::notify(osg::ALWAYS) << "FirmwareVersion: " << m_hmdDesc.FirmwareMajor << "." << m_hmdDesc.FirmwareMinor << std::endl;
}

void OculusDevice::initializeEyeRenderDesc()
{
	m_eyeRenderDesc[0] = ovr_GetRenderDesc(m_hmdDevice, ovrEye_Left, m_hmdDesc.DefaultEyeFov[0]);
	m_eyeRenderDesc[1] = ovr_GetRenderDesc(m_hmdDevice, ovrEye_Right, m_hmdDesc.DefaultEyeFov[1]);
}

void OculusDevice::calculateEyeAdjustment()
{
	ovrVector3f leftEyeAdjust = m_eyeRenderDesc[0].HmdToEyeViewOffset;
	ovrVector3f rightEyeAdjust = m_eyeRenderDesc[1].HmdToEyeViewOffset;

	m_leftEyeAdjust.set(leftEyeAdjust.x, leftEyeAdjust.y, leftEyeAdjust.z);
	m_rightEyeAdjust.set(rightEyeAdjust.x, rightEyeAdjust.y, rightEyeAdjust.z);

	// Display IPD
	float ipd = (m_leftEyeAdjust - m_rightEyeAdjust).length();
	osg::notify(osg::ALWAYS) << "Interpupillary Distance (IPD): " << ipd * 1000.0f << " mm" << std::endl;

	// Scale to world units
	m_leftEyeAdjust *= m_worldUnitsPerMetre;
	m_rightEyeAdjust *= m_worldUnitsPerMetre;
}

void OculusDevice::calculateProjectionMatrices()
{
	unsigned int projectionModifier = ovrProjection_RightHanded;
	projectionModifier |= ovrProjection_ClipRangeOpenGL;

	ovrMatrix4f leftEyeProjectionMatrix = ovrMatrix4f_Projection(m_eyeRenderDesc[0].Fov, m_nearClip, m_farClip, projectionModifier);
	// Transpose matrix
	m_leftEyeProjectionMatrix.set(leftEyeProjectionMatrix.M[0][0], leftEyeProjectionMatrix.M[1][0], leftEyeProjectionMatrix.M[2][0], leftEyeProjectionMatrix.M[3][0],
								  leftEyeProjectionMatrix.M[0][1], leftEyeProjectionMatrix.M[1][1], leftEyeProjectionMatrix.M[2][1], leftEyeProjectionMatrix.M[3][1],
								  leftEyeProjectionMatrix.M[0][2], leftEyeProjectionMatrix.M[1][2], leftEyeProjectionMatrix.M[2][2], leftEyeProjectionMatrix.M[3][2],
								  leftEyeProjectionMatrix.M[0][3], leftEyeProjectionMatrix.M[1][3], leftEyeProjectionMatrix.M[2][3], leftEyeProjectionMatrix.M[3][3]);

	ovrMatrix4f rightEyeProjectionMatrix = ovrMatrix4f_Projection(m_eyeRenderDesc[1].Fov, m_nearClip, m_farClip, projectionModifier);
	// Transpose matrix
	m_rightEyeProjectionMatrix.set(rightEyeProjectionMatrix.M[0][0], rightEyeProjectionMatrix.M[1][0], rightEyeProjectionMatrix.M[2][0], rightEyeProjectionMatrix.M[3][0],
								   rightEyeProjectionMatrix.M[0][1], rightEyeProjectionMatrix.M[1][1], rightEyeProjectionMatrix.M[2][1], rightEyeProjectionMatrix.M[3][1],
								   rightEyeProjectionMatrix.M[0][2], rightEyeProjectionMatrix.M[1][2], rightEyeProjectionMatrix.M[2][2], rightEyeProjectionMatrix.M[3][2],
								   rightEyeProjectionMatrix.M[0][3], rightEyeProjectionMatrix.M[1][3], rightEyeProjectionMatrix.M[2][3], rightEyeProjectionMatrix.M[3][3]);
}

void OculusDevice::setupLayers()
{
	m_layerEyeFov.Header.Type = ovrLayerType_EyeFov;
	m_layerEyeFov.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;   // Because OpenGL.

	ovrRecti viewPort[2];
	viewPort[0].Pos.x = 0;
	viewPort[0].Pos.y = 0;
	viewPort[0].Size.w = m_textureBuffer[0]->textureWidth();
	viewPort[0].Size.h = m_textureBuffer[0]->textureHeight();

	viewPort[1].Pos.x = 0;
	viewPort[1].Pos.y = 0;
	viewPort[1].Size.w = m_textureBuffer[1]->textureWidth();
	viewPort[1].Size.h = m_textureBuffer[1]->textureHeight();

	m_layerEyeFov.Viewport[0] = viewPort[0];
	m_layerEyeFov.Viewport[1] = viewPort[1];
	m_layerEyeFov.Fov[0] = m_eyeRenderDesc[0].Fov;
	m_layerEyeFov.Fov[1] = m_eyeRenderDesc[1].Fov;
}

void OculusDevice::trySetProcessAsHighPriority() const
{
	// Require at least 4 processors, otherwise the process could occupy the machine.
	if (OpenThreads::GetNumberOfProcessors() >= 4)
	{
#ifdef _WIN32
		SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
	}
}

void OculusRealizeOperation::operator() (osg::GraphicsContext* gc)
{
	if (!m_realized)
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
		gc->makeCurrent();

		if (osgViewer::GraphicsWindow* window = dynamic_cast<osgViewer::GraphicsWindow*>(gc))
		{
			// Run wglSwapIntervalEXT(0) to force VSync Off
			window->setSyncToVBlank(false);
		}

		osg::ref_ptr<osg::State> state = gc->getState();
		m_device->createRenderBuffers(state);
		// Init the oculus system
		m_device->init();
	}

	m_realized = true;
}

void OculusSwapCallback::swapBuffersImplementation(osg::GraphicsContext* gc)
{
	// Submit rendered frame to compositor
	m_device->submitFrame();

	// Blit mirror texture to backbuffer
	m_device->blitMirrorTexture(gc);

	// Run the default system swapBufferImplementation
	gc->swapBuffersImplementation();
}


