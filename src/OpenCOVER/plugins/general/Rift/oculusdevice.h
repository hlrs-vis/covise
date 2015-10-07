/*
 * oculusdevice.h
 *
 *  Created on: Jul 03, 2013
 *      Author: Bjorn Blissing
 */

#ifndef _OSG_OCULUSDEVICE_H_
#define _OSG_OCULUSDEVICE_H_

// Include the OculusVR SDK
#include <OVR_CAPI_GL.h>

#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/Version>
#include <osg/FrameBufferObject>

class RIFTDriver;

#if(OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
	typedef osg::GLExtensions OSG_GLExtensions;
	typedef osg::GLExtensions OSG_Texture_Extensions;
#else
	typedef osg::FBOExtensions OSG_GLExtensions;
	typedef osg::Texture::Extensions OSG_Texture_Extensions;
#endif

class OculusTextureBuffer : public osg::Referenced
{
public:
	OculusTextureBuffer(const ovrHmd& hmd, osg::ref_ptr<osg::State> state, const ovrSizei& size, int msaaSamples);
	void destroy();
	int textureWidth() const { return m_textureSize.x(); }
	int textureHeight() const { return m_textureSize.y(); }
	int samples() const { return m_samples; }
	ovrSwapTextureSet* textureSet() const { return m_textureSet; }
	osg::ref_ptr<osg::Texture2D> colorBuffer() const { return m_colorBuffer; }
	osg::ref_ptr<osg::Texture2D> depthBuffer() const { return m_depthBuffer; }
	void advanceIndex() { m_textureSet->CurrentIndex = (m_textureSet->CurrentIndex + 1) % m_textureSet->TextureCount; }
	void onPreRender(osg::RenderInfo& renderInfo);
	void onPostRender(osg::RenderInfo& renderInfo);

protected:
	~OculusTextureBuffer() {}

	const ovrHmd m_hmdDevice;
	ovrSwapTextureSet* m_textureSet;
	osg::ref_ptr<osg::Texture2D> m_colorBuffer;
	osg::ref_ptr<osg::Texture2D> m_depthBuffer;
	osg::Vec2i m_textureSize;

	void setup(osg::State& state);
	void setupMSAA(osg::State& state);

	GLuint m_Oculus_FBO; // MSAA FBO is copied to this FBO after render.
	GLuint m_MSAA_FBO; // framebuffer for MSAA texture
	GLuint m_MSAA_ColorTex; // color texture for MSAA
	GLuint m_MSAA_DepthTex; // depth texture for MSAA
	int m_samples;  // sample width for MSAA

};

class OculusMirrorTexture : public osg::Referenced
{
public:
	OculusMirrorTexture(const ovrHmd& hmd, osg::ref_ptr<osg::State> state, int width, int height);
	void destroy(const OSG_GLExtensions* fbo_ext = 0);
	GLuint id() const { return m_texture->OGL.TexId; }
	GLint width() const { return m_texture->OGL.Header.TextureSize.w; }
	GLint height() const { return m_texture->OGL.Header.TextureSize.h; }
	void blitTexture(osg::GraphicsContext* gc);
protected:
	~OculusMirrorTexture() {}

	const ovrHmd m_hmdDevice;
	ovrGLTexture* m_texture;
	GLuint m_mirrorFBO;
};


class OculusPreDrawCallback : public osg::Camera::DrawCallback
{
public:
	OculusPreDrawCallback(osg::Camera* camera, OculusTextureBuffer* textureBuffer)
		: m_camera(camera)
		, m_textureBuffer(textureBuffer)
	{
	}

	virtual void operator()(osg::RenderInfo& renderInfo) const;
protected:
	osg::Camera* m_camera;
	OculusTextureBuffer* m_textureBuffer;

};

class OculusPostDrawCallback : public osg::Camera::DrawCallback
{
public:
	OculusPostDrawCallback(osg::Camera* camera, OculusTextureBuffer* textureBuffer)
		: m_camera(camera)
		, m_textureBuffer(textureBuffer)
	{
	}

	virtual void operator()(osg::RenderInfo& renderInfo) const;
protected:
	osg::Camera* m_camera;
	OculusTextureBuffer* m_textureBuffer;

};


class OculusDevice : public osg::Referenced
{

public:
	typedef enum Eye_
	{
		LEFT = 0,
		RIGHT = 1,
		COUNT = 2
	} Eye;
	OculusDevice(float nearClip, float farClip, const float pixelsPerDisplayPixel = 1.0f, const float worldUnitsPerMetre = 1.0f, const int samples = 0);
	void createRenderBuffers(osg::ref_ptr<osg::State> state);
	void init();

	unsigned int screenResolutionWidth() const;
	unsigned int screenResolutionHeight() const;

	osg::Matrix projectionMatrixCenter() const;
	osg::Matrix projectionMatrixLeft() const;
	osg::Matrix projectionMatrixRight() const;

	osg::Matrix projectionOffsetMatrixLeft() const;
	osg::Matrix projectionOffsetMatrixRight() const;

	osg::Matrix viewMatrixLeft() const;
	osg::Matrix viewMatrixRight() const;

	float nearClip() const { return m_nearClip;	}
	float farClip() const { return m_farClip; }

	void resetSensorOrientation() const;
	void updatePose(unsigned int frameIndex = 0);

	osg::Vec3 position() const { return m_position; }
	osg::Quat orientation() const { return m_orientation;  }

	osg::Camera* createRTTCamera(OculusDevice::Eye eye, osg::Transform::ReferenceFrame referenceFrame, const osg::Vec4& clearColor, osg::GraphicsContext* gc = 0) const;

	bool submitFrame(unsigned int frameIndex = 0);
	void blitMirrorTexture(osg::GraphicsContext* gc);

	void setPerfHudMode(int mode);
	void setPositionalTrackingState(bool state);

	osg::GraphicsContext::Traits* graphicsContextTraits() const;
    
    ovrHmd getHMD()
    {
        return m_hmdDevice;
    };
    RIFTDriver *trackingDriver;
protected:
	~OculusDevice(); // Since we inherit from osg::Referenced we must make destructor protected

	void printHMDDebugInfo();

	void initializeEyeRenderDesc();
	// Note: this function requires you to run the previous function first.
	void calculateEyeAdjustment();
	// Note: this function requires you to run the previous function first.
	void calculateProjectionMatrices();

	void setupLayers();

	void trySetProcessAsHighPriority() const;

	ovrHmd m_hmdDevice;
	ovrHmdDesc m_hmdDesc;

	const float m_pixelsPerDisplayPixel;
	const float m_worldUnitsPerMetre;

	osg::ref_ptr<OculusTextureBuffer> m_textureBuffer[2];
	osg::ref_ptr<OculusMirrorTexture> m_mirrorTexture;

	ovrEyeRenderDesc m_eyeRenderDesc[2];
	ovrVector2f m_UVScaleOffset[2][2];
	ovrFrameTiming m_frameTiming;
	ovrPosef m_headPose[2];
	ovrPosef m_eyeRenderPose[2];
	ovrLayerEyeFov m_layerEyeFov;
	ovrVector3f m_viewOffset[2];
	osg::Matrixf m_leftEyeProjectionMatrix;
	osg::Matrixf m_rightEyeProjectionMatrix;
	osg::Vec3f m_leftEyeAdjust;
	osg::Vec3f m_rightEyeAdjust;

	osg::Vec3 m_position;
	osg::Quat m_orientation;

	float m_nearClip;
	float m_farClip;
	int m_samples;
    bool m_initialized;
private:
	OculusDevice(const OculusDevice&); // Do not allow copy
	OculusDevice& operator=(const OculusDevice&); // Do not allow assignment operator.
};


class OculusRealizeOperation : public osg::GraphicsOperation
{
public:
	explicit OculusRealizeOperation(osg::ref_ptr<OculusDevice> device) :
		osg::GraphicsOperation("OculusRealizeOperation", false), m_device(device), m_realized(false) {}
	virtual void operator () (osg::GraphicsContext* gc);
	bool realized() const { return m_realized; }
protected:
	OpenThreads::Mutex  _mutex;
	osg::observer_ptr<OculusDevice> m_device;
	bool m_realized;
};


class OculusSwapCallback : public osg::GraphicsContext::SwapCallback
{
public:
	explicit OculusSwapCallback(osg::ref_ptr<OculusDevice> device) : m_device(device), m_frameIndex(0) {}
	void swapBuffersImplementation(osg::GraphicsContext* gc);
	int frameIndex() const { return m_frameIndex; }
private:
	osg::observer_ptr<OculusDevice> m_device;
	int m_frameIndex;
};


#endif /* _OSG_OCULUSDEVICE_H_ */
