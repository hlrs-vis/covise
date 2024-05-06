/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OpenXR_H
#define OpenXR_H

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: OpenXR Plugin				                                 **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner		                                             **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#define XR_USE_GRAPHICS_API_OPENGL
#include <cover/coVRPlugin.h>

#include <openxr/openxr.h>
#include <DebugOutput.h>
#include <GraphicsAPI_OpenGL.h>
#include <OpenXRDebugUtils.h>

#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/Version>
#include <osg/FrameBufferObject>
#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>
#include <cover/input/inputdevice.h>

#include <xr_linear_algebra.h>


#if(OSG_VERSION_GREATER_OR_EQUAL(3, 4, 0))
typedef osg::GLExtensions OSG_GLExtensions;
typedef osg::GLExtensions OSG_Texture_Extensions;
#else
typedef osg::FBOExtensions OSG_GLExtensions;
typedef osg::Texture::Extensions OSG_Texture_Extensions;
#endif
struct serialInfo
{
	int ID;
	int controllerID;
};

class OpenXRTextureBuffer : public osg::Referenced
{
public:
	OpenXRTextureBuffer(osg::ref_ptr<osg::State> state, int width, int height, int msaaSamples);
	void destroy(osg::GraphicsContext* gc);
	GLuint getTexture() { return m_Resolve_ColorTex; }
	int textureWidth() const { return m_width; }
	int textureHeight() const { return m_height; }
	int samples() const { return m_samples; }
	void onPreRender(osg::RenderInfo& renderInfo);
	void onPostRender(osg::RenderInfo& renderInfo);

protected:
	~OpenXRTextureBuffer() {}

	friend class OpenXRMirrorTexture;
	GLuint m_Resolve_FBO; // MSAA FBO is copied to this FBO after render.
	GLuint m_Resolve_ColorTex; // color texture for above FBO.
	GLuint m_MSAA_FBO; // framebuffer for MSAA RTT
	GLuint m_MSAA_ColorTex; // color texture for MSAA RTT 
	GLuint m_MSAA_DepthTex; // depth texture for MSAA RTT
	GLint m_width; // width of texture in pixels
	GLint m_height; // height of texture in pixels
	int m_samples;  // sample width for MSAA

};

class OpenXRMirrorTexture : public osg::Referenced
{
public:
	OpenXRMirrorTexture(osg::ref_ptr<osg::State> state, GLint width, GLint height);
	void destroy(osg::GraphicsContext* gc);
	void blitTexture(osg::GraphicsContext* gc, OpenXRTextureBuffer* leftEye, OpenXRTextureBuffer* rightEye);
protected:
	~OpenXRMirrorTexture() {}

	GLuint m_mirrorFBO;
	GLuint m_mirrorTex;
	GLint m_width;
	GLint m_height;
};




class OpenXR : public opencover::coVRPlugin, public opencover::InputDevice
{
public:
    OpenXR();
    ~OpenXR();
	void preFrame();
	void postFrame();
	//! this function is called from the draw thread before swapbuffers
	virtual void preSwapBuffers(int /*windowNumber*/);
	bool init();
	virtual bool needsThread() const; //< we don't needan extra thread
    bool trackingOnly = false;

private:

	void CreateDebugMessenger();
	void DestroyDebugMessenger();
	void GetInstanceProperties();
	void GetSystemID();
	void PollSystemEvents() {
	}

	XrApplicationInfo AI;
	XrInstance m_xrInstance = {};
	std::vector<const char*> m_activeAPILayers = {};
	std::vector<const char*> m_activeInstanceExtensions = {};
	std::vector<std::string> m_apiLayers = {};
	std::vector<std::string> m_instanceExtensions = {};

	XrDebugUtilsMessengerEXT m_debugUtilsMessenger = {};

	XrFormFactor m_formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	XrSystemId m_systemID = {};
	XrSystemProperties m_systemProperties = { XR_TYPE_SYSTEM_PROPERTIES };

	GraphicsAPI_Type m_apiType = UNKNOWN;

	std::unique_ptr<GraphicsAPI> m_graphicsAPI = nullptr;

	XrSession m_session = XR_NULL_HANDLE;

	XrSessionState m_sessionState = XR_SESSION_STATE_UNKNOWN;

	bool m_applicationRunning = true;
	bool m_sessionRunning = false;


	//std::string GetTrackedDeviceString(vr::IVRSystem *pHmd, vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *peError = NULL);

	std::map<std::string, serialInfo> serialID;
	size_t maxBodyNumber;
    size_t numControllers;
    size_t numTrackers;
    size_t numBaseStations;
	bool haveTrackerOrigin;
	bool m_transformOriginToLighthouse;	//Config variable for origin transform
	osg::Matrix LighthouseMatrix;


};
#endif
