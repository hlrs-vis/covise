/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

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

#include "OpenXR.h"

#include <iostream>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <osg/io_utils>
#include <cover/input/input.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <config/CoviseConfig.h>
#include <osg/Texture2D>
#include <osg/GraphicsContext>
#include <osgViewer/GraphicsWindow>
#include <osgViewer/api/Win32/GraphicsWindowWin32>

using namespace opencover;
#define OPENXR_CHECK_RETURN(x, y)                                                                                                                                  \
    {                                                                                                                                                       \
        XrResult result = (x);                                                                                                                              \
        if (!XR_SUCCEEDED(result)) {                                                                                                                        \
            std::cerr << "ERROR: OPENXR: " << int(result) << "(" << (m_xrInstance ? GetXRErrorString(m_xrInstance, result) : "") << ") " << y << std::endl; \
            return false;                                                                                                                             \
        }                                                                                                                                                   \
    }


static osg::Matrix convertMatrix44(const XrMatrix4x4f &mat44)
{
	osg::Matrix matrix(
		mat44.m
	);
	return matrix;
}

OpenXR::OpenXR()
: coVRPlugin(COVER_PLUGIN_NAME)
, InputDevice("COVER.Input.Device.OpenXR")
{
	Input::instance()->addDevice("OpenXR", this);
	haveTrackerOrigin = false;
	LighthouseMatrix.makeIdentity();
	maxBodyNumber = 0;


	bool exists;
    trackingOnly = covise::coCoviseConfig::isOn("trackingOnly", "COVER.Plugin.OpenXR", false, &exists);
	if (!trackingOnly)
	{
		if (coVRConfig::instance()->numPBOs() == 0)
		{ // no PBOs configured, thus try an outo config
			PBOStruct pbol;
			PBOStruct pbor;
			uint32_t sx=1920, sy=1080;
			//ivrSystem->GetRecommendedRenderTargetSize(&sx, &sy);
			pbol.PBOsx = pbor.PBOsx = sx;
			pbol.PBOsy = pbor.PBOsy = sy;
			pbor.windowNum = 0;
			pbol.windowNum = 0;
			coVRConfig::instance()->PBOs.push_back(pbol);
			coVRConfig::instance()->PBOs.push_back(pbor);

			channelStruct chanl;
			channelStruct chanr;
			chanl.fixedViewer = false;
			chanl.name = "OpenXRL";
			chanl.PBONum = 0;
			chanl.screenNum = 0;
			chanl.stereo = true;
			chanl.stereoMode = osg::DisplaySettings::LEFT_EYE;
			chanl.viewportNum = -1;

			chanr.fixedViewer = false;
			chanr.name = "OpenXRR";
			chanr.PBONum = 1;
			chanr.screenNum = 0;
			chanr.stereo = true;
			chanr.stereoMode = osg::DisplaySettings::RIGHT_EYE;
			chanr.viewportNum = -1;

			coVRConfig::instance()->channels.clear();
			coVRConfig::instance()->channels.push_back(chanl);
			coVRConfig::instance()->channels.push_back(chanr);

			viewportStruct vpl;
			viewportStruct vpr;
			vpl.mode = viewportStruct::Channel;
			vpl.sourceXMin = vpl.sourceYMin = 0.0;
			vpl.sourceXMax = vpl.sourceYMax = -1.0;
			vpl.viewportXMin = vpl.viewportYMin = 0.0;
			vpl.viewportXMax = vpl.viewportYMax = 1.0;
			vpl.window = 0;
			vpl.PBOnum = -1;
			vpl.distortMeshName = "";
			vpl.blendingTextureName = "";
			vpr.mode = viewportStruct::PBO;
			vpr.sourceXMin = vpr.sourceYMin = 0.0;
			vpr.sourceXMax = vpr.sourceYMax = 1.0;
			vpr.viewportXMin = vpr.viewportYMin = 0.0;
			vpr.viewportXMax = vpr.viewportYMax = 1.0;
			vpr.window = 0;
			vpr.PBOnum = 1;
			vpr.distortMeshName = "";
			vpr.blendingTextureName = "";

			coVRConfig::instance()->viewports.clear();
			coVRConfig::instance()->viewports.push_back(vpl);
			coVRConfig::instance()->viewports.push_back(vpr);
		}

	}
}

bool OpenXR::needsThread() const
{
	return false;
} 

void OpenXR::GetInstanceProperties()
{
	XrInstanceProperties instanceProperties{ XR_TYPE_INSTANCE_PROPERTIES };
	OPENXR_CHECK(xrGetInstanceProperties(m_xrInstance, &instanceProperties), "Failed to get InstanceProperties.");

	XR_TUT_LOG("OpenXR Runtime: " << instanceProperties.runtimeName << " - "
		<< XR_VERSION_MAJOR(instanceProperties.runtimeVersion) << "."
		<< XR_VERSION_MINOR(instanceProperties.runtimeVersion) << "."
		<< XR_VERSION_PATCH(instanceProperties.runtimeVersion));
}

void OpenXR::CreateDebugMessenger() {
	// Check that "XR_EXT_debug_utils" is in the active Instance Extensions before creating an XrDebugUtilsMessengerEXT.
	if (IsStringInVector(m_activeInstanceExtensions, XR_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		m_debugUtilsMessenger = CreateOpenXRDebugUtilsMessenger(m_xrInstance);  // From OpenXRDebugUtils.h.
	}
}
void OpenXR::DestroyDebugMessenger()
{
	// Check that "XR_EXT_debug_utils" is in the active Instance Extensions before destroying the XrDebugUtilsMessengerEXT.
	if (m_debugUtilsMessenger != XR_NULL_HANDLE) {
		DestroyOpenXRDebugUtilsMessenger(m_xrInstance, m_debugUtilsMessenger);  // From OpenXRDebugUtils.h.
	}
}
void OpenXR::GetSystemID()
{
	// Get the XrSystemId from the instance and the supplied XrFormFactor.
	XrSystemGetInfo systemGI{ XR_TYPE_SYSTEM_GET_INFO };
	systemGI.formFactor = m_formFactor;
	OPENXR_CHECK(xrGetSystem(m_xrInstance, &systemGI, &m_systemID), "Failed to get SystemID.");

	// Get the System's properties for some general information about the hardware and the vendor.
	OPENXR_CHECK(xrGetSystemProperties(m_xrInstance, m_systemID, &m_systemProperties), "Failed to get SystemProperties.");
}
bool OpenXR::init()
{
	m_apiType = OPENGL;
	
	std::string formFactor = covise::coCoviseConfig::getEntry("formFactor", "COVER.Input.Device.OpenXR", "HMD");
	if (formFactor == "HMD")
		m_formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	if (formFactor == "HANDHELD")
		m_formFactor = XR_FORM_FACTOR_HANDHELD_DISPLAY;



    strncpy(AI.applicationName, "OpenXR plugin", XR_MAX_APPLICATION_NAME_SIZE);
    AI.applicationVersion = 1;
    strncpy(AI.engineName, "OpenCOVER", XR_MAX_ENGINE_NAME_SIZE);
    AI.engineVersion = 1;
    AI.apiVersion = XR_CURRENT_API_VERSION;
    m_instanceExtensions.push_back(XR_EXT_DEBUG_UTILS_EXTENSION_NAME);
    // Ensure m_apiType is already defined when we call this line.
    m_instanceExtensions.push_back(XR_KHR_OPENGL_ENABLE_EXTENSION_NAME);
    // Get all the API Layers from the OpenXR runtime.
    uint32_t apiLayerCount = 0;
    std::vector<XrApiLayerProperties> apiLayerProperties;
	OPENXR_CHECK_RETURN(xrEnumerateApiLayerProperties(0, &apiLayerCount, nullptr), "Failed to enumerate ApiLayerProperties.");
    apiLayerProperties.resize(apiLayerCount, { XR_TYPE_API_LAYER_PROPERTIES });
	OPENXR_CHECK_RETURN(xrEnumerateApiLayerProperties(apiLayerCount, &apiLayerCount, apiLayerProperties.data()), "Failed to enumerate ApiLayerProperties.");

    // Check the requested API layers against the ones from the OpenXR. If found add it to the Active API Layers.
    for (auto& requestLayer : m_apiLayers) {
        for (auto& layerProperty : apiLayerProperties) {
            // strcmp returns 0 if the strings match.
            if (strcmp(requestLayer.c_str(), layerProperty.layerName) != 0) {
                continue;
            }
            else {
                m_activeAPILayers.push_back(requestLayer.c_str());
                break;
            }
        }
    }

    // Get all the Instance Extensions from the OpenXR instance.
    uint32_t extensionCount = 0;
    std::vector<XrExtensionProperties> extensionProperties;
	OPENXR_CHECK_RETURN(xrEnumerateInstanceExtensionProperties(nullptr, 0, &extensionCount, nullptr), "Failed to enumerate InstanceExtensionProperties.");
    extensionProperties.resize(extensionCount, { XR_TYPE_EXTENSION_PROPERTIES });
	OPENXR_CHECK_RETURN(xrEnumerateInstanceExtensionProperties(nullptr, extensionCount, &extensionCount, extensionProperties.data()), "Failed to enumerate InstanceExtensionProperties.");

    // Check the requested Instance Extensions against the ones from the OpenXR runtime.
    // If an extension is found add it to Active Instance Extensions.
    // Log error if the Instance Extension is not found.
    for (auto& requestedInstanceExtension : m_instanceExtensions) {
        bool found = false;
        for (auto& extensionProperty : extensionProperties) {
            // strcmp returns 0 if the strings match.
            if (strcmp(requestedInstanceExtension.c_str(), extensionProperty.extensionName) != 0) {
                continue;
            }
            else {
                m_activeInstanceExtensions.push_back(requestedInstanceExtension.c_str());
                found = true;
                break;
            }
        }
        if (!found) {
            XR_TUT_LOG_ERROR("Failed to find OpenXR instance extension: " << requestedInstanceExtension);
        }
    }
    XrInstanceCreateInfo instanceCI{ XR_TYPE_INSTANCE_CREATE_INFO };
    instanceCI.createFlags = 0;
    instanceCI.applicationInfo = AI;
    instanceCI.enabledApiLayerCount = static_cast<uint32_t>(m_activeAPILayers.size());
    instanceCI.enabledApiLayerNames = m_activeAPILayers.data();
    instanceCI.enabledExtensionCount = static_cast<uint32_t>(m_activeInstanceExtensions.size());
    instanceCI.enabledExtensionNames = m_activeInstanceExtensions.data();
	OPENXR_CHECK_RETURN(xrCreateInstance(&instanceCI, &m_xrInstance), "Failed to create Instance.");

    CreateDebugMessenger();

	GetInstanceProperties();
	GetSystemID();

	// createSession
	XrSessionCreateInfo sessionCI{ XR_TYPE_SESSION_CREATE_INFO };


    //graphicsBinding = { XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR };
#if defined(WIN32)
    //osgViewer::GraphicsWindowWin32 *gw = dynamic_cast<osgViewer::GraphicsWindowWin32 *>(opencover::coVRConfig::instance()->windows[0].context.get());

    //graphicsBinding.hDC = gw->getHDC();
    //graphicsBinding.hGLRC = gw->getWGLContext();
#else
#endif
    osgViewer::GraphicsWindow * gw = dynamic_cast<osgViewer::GraphicsWindow*>(opencover::coVRConfig::instance()->windows[0].context.get());

    sessionCI.next = gw;
	sessionCI.createFlags = 0;
	sessionCI.systemId = m_systemID;

	OPENXR_CHECK_RETURN(xrCreateSession(m_xrInstance, &sessionCI, &m_session), "Failed to create Session.");
	

	return true;
}

// this is called if the plugin is removed at runtime
OpenXR::~OpenXR()
{
    OPENXR_CHECK(xrDestroySession(m_session), "Failed to destroy Session.");
	DestroyDebugMessenger();
	
	OPENXR_CHECK(xrDestroyInstance(m_xrInstance), "Failed to destroy Instance.");

	Input::instance()->removeDevice("OpenXR", this);
	fprintf(stderr, "OpenXR::~OpenXR\n");
}

void OpenXR::preFrame()
{
    while (m_applicationRunning)
    {
        PollSystemEvents();// Poll OpenXR for a new event.
        XrEventDataBuffer eventData{ XR_TYPE_EVENT_DATA_BUFFER };
        auto XrPollEvents = [&]() -> bool
            {
            eventData = { XR_TYPE_EVENT_DATA_BUFFER };
            return xrPollEvent(m_xrInstance, &eventData) == XR_SUCCESS;
            };

        while (XrPollEvents()) 
        {
            switch (eventData.type)
            {
                // Log the number of lost events from the runtime.
            case XR_TYPE_EVENT_DATA_EVENTS_LOST:
            {
                XrEventDataEventsLost* eventsLost = reinterpret_cast<XrEventDataEventsLost*>(&eventData);
                XR_TUT_LOG("OPENXR: Events Lost: " << eventsLost->lostEventCount);
                break;
            }
                                               // Log that an instance loss is pending and shutdown the application.
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
            {
                XrEventDataInstanceLossPending* instanceLossPending = reinterpret_cast<XrEventDataInstanceLossPending*>(&eventData);
                XR_TUT_LOG("OPENXR: Instance Loss Pending at: " << instanceLossPending->lossTime);
                m_sessionRunning = false;
                m_applicationRunning = false;
                break;
            }
                                                         // Log that the interaction profile has changed.
            case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
            {
                XrEventDataInteractionProfileChanged* interactionProfileChanged = reinterpret_cast<XrEventDataInteractionProfileChanged*>(&eventData);
                XR_TUT_LOG("OPENXR: Interaction Profile changed for Session: " << interactionProfileChanged->session);
                if (interactionProfileChanged->session != m_session) {
                    XR_TUT_LOG("XrEventDataInteractionProfileChanged for unknown Session");
                    break;
                }
                break;
            }
                                                               // Log that there's a reference space change pending.
            case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
            {
                XrEventDataReferenceSpaceChangePending* referenceSpaceChangePending = reinterpret_cast<XrEventDataReferenceSpaceChangePending*>(&eventData);
                XR_TUT_LOG("OPENXR: Reference Space Change pending for Session: " << referenceSpaceChangePending->session);
                if (referenceSpaceChangePending->session != m_session) {
                    XR_TUT_LOG("XrEventDataReferenceSpaceChangePending for unknown Session");
                    break;
                }
                break;
            }
                                                                  // Session State changes:
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
            {
                XrEventDataSessionStateChanged* sessionStateChanged = reinterpret_cast<XrEventDataSessionStateChanged*>(&eventData);
                if (sessionStateChanged->session != m_session)
                {
                    XR_TUT_LOG("XrEventDataSessionStateChanged for unknown Session");
                    break;
                }

                if (sessionStateChanged->state == XR_SESSION_STATE_READY)
                {
                    // SessionState is ready. Begin the XrSession using the XrViewConfigurationType.
                    XrSessionBeginInfo sessionBeginInfo{ XR_TYPE_SESSION_BEGIN_INFO };
                    sessionBeginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    OPENXR_CHECK(xrBeginSession(m_session, &sessionBeginInfo), "Failed to begin Session.");
                    m_sessionRunning = true;
                }
                if (sessionStateChanged->state == XR_SESSION_STATE_STOPPING)
                {
                    // SessionState is stopping. End the XrSession.
                    OPENXR_CHECK(xrEndSession(m_session), "Failed to end Session.");
                    m_sessionRunning = false;
                }
                if (sessionStateChanged->state == XR_SESSION_STATE_EXITING)
                {
                    // SessionState is exiting. Exit the application.
                    m_sessionRunning = false;
                    m_applicationRunning = false;
                }
                if (sessionStateChanged->state == XR_SESSION_STATE_LOSS_PENDING)
                {
                    // SessionState is loss pending. Exit the application.
                    // It's possible to try a reestablish an XrInstance and XrSession, but we will simply exit here.
                    m_sessionRunning = false;
                    m_applicationRunning = false;
                }
                // Store state for reference across the application.
                m_sessionState = sessionStateChanged->state;
                break;
            }
            default:
            {
                break;
            }
            }
        }

    }
}

void OpenXR::postFrame()
{


}

void OpenXR::preSwapBuffers(int /*windowNumber*/)
{
	/*
    if (!trackingOnly)
    {
        if (coVRConfig::instance()->PBOs.size() > 1)
        {
            vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)coVRConfig::instance()->PBOs[0].renderTargetTexture.get()->getTextureObject(coVRConfig::instance()->windows[0].context->getState()->getContextID())->id(), vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
            vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
            //vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)rightEyeDesc.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
            vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)coVRConfig::instance()->PBOs[1].renderTargetTexture.get()->getTextureObject(coVRConfig::instance()->windows[0].context->getState()->getContextID())->id(), vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
            vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);

        }
        else
        {
            fprintf(stderr, "configure two PBOs\n");
        }
    }*/
}


COVERPLUGIN(OpenXR)

