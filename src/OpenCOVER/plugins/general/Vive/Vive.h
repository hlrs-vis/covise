/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VIVE_H
#define VIVE_H

/****************************************************************************\ 
 **                                                            (C)2016 HLRS  **
 **                                                                          **
 ** Description: Vive Plugin				                                 **
 **                                                                          **
 **                                                                          **
 ** Author: Uwe Woessner		                                             **
 **                                                                          **
 ** History:  								                                 **
 ** Sep-16  v1	    				       		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>

#include <openvr.h>

#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/Version>
#include <osg/FrameBufferObject>
#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>
#include <cover/input/inputdevice.h>


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

class OpenVRTextureBuffer : public osg::Referenced
{
public:
	OpenVRTextureBuffer(osg::ref_ptr<osg::State> state, int width, int height, int msaaSamples);
	void destroy(osg::GraphicsContext* gc);
	GLuint getTexture() { return m_Resolve_ColorTex; }
	int textureWidth() const { return m_width; }
	int textureHeight() const { return m_height; }
	int samples() const { return m_samples; }
	void onPreRender(osg::RenderInfo& renderInfo);
	void onPostRender(osg::RenderInfo& renderInfo);

protected:
	~OpenVRTextureBuffer() {}

	friend class OpenVRMirrorTexture;
	GLuint m_Resolve_FBO; // MSAA FBO is copied to this FBO after render.
	GLuint m_Resolve_ColorTex; // color texture for above FBO.
	GLuint m_MSAA_FBO; // framebuffer for MSAA RTT
	GLuint m_MSAA_ColorTex; // color texture for MSAA RTT 
	GLuint m_MSAA_DepthTex; // depth texture for MSAA RTT
	GLint m_width; // width of texture in pixels
	GLint m_height; // height of texture in pixels
	int m_samples;  // sample width for MSAA

};

class OpenVRMirrorTexture : public osg::Referenced
{
public:
	OpenVRMirrorTexture(osg::ref_ptr<osg::State> state, GLint width, GLint height);
	void destroy(osg::GraphicsContext* gc);
	void blitTexture(osg::GraphicsContext* gc, OpenVRTextureBuffer* leftEye, OpenVRTextureBuffer* rightEye);
protected:
	~OpenVRMirrorTexture() {}

	GLuint m_mirrorFBO;
	GLuint m_mirrorTex;
	GLint m_width;
	GLint m_height;
};




class Vive : public opencover::coVRPlugin, public opencover::InputDevice
{
public:
    Vive();
    ~Vive();
	void preFrame();
	void postFrame();
	//! this function is called from the draw thread before swapbuffers
	virtual void preSwapBuffers(int /*windowNumber*/);
	bool init();
	virtual bool needsThread() const; //< we don't needan extra thread
    bool trackingOnly = false;

private:


	vr::IVRSystem *ivrSystem;
	vr::IVRRenderModels* ivrRenderModels;

	std::string GetTrackedDeviceString(vr::IVRSystem *pHmd, vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *peError = NULL);

	std::string m_strDriver;
	std::string m_strDisplay;

	std::string m_strPoseClasses;                            // what classes we saw poses for this frame
	char m_rDevClassChar[vr::k_unMaxTrackedDeviceCount];   // for each device, a character representing its class
	int m_DeviceID[vr::k_unMaxTrackedDeviceCount];  // cover ID for a specific DeviceID
	int m_ControllerID[vr::k_unMaxTrackedDeviceCount];  // controller ID for a specific DeviceID for button states
	
	std::string m_DeviceSerial[vr::k_unMaxTrackedDeviceCount];   // for each device, a character representing its class // index is device ID
	std::map<std::string, serialInfo> serialID;
	size_t maxBodyNumber;
	size_t numControllers;
	vr::TrackedDevicePose_t m_rTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
	bool haveTrackerOrigin;
	bool m_transformOriginToCamera;	//Config variable for origin transform
	osg::Matrix LighthouseMatrix;


};
#endif
