/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

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

#include "Vive.h"

#include <iostream>

#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <osg/io_utils>
#include <cover/input/input.h>

using namespace opencover;


static osg::Matrix convertMatrix34(const vr::HmdMatrix34_t &mat34)
{
	osg::Matrix matrix(
		mat34.m[0][0], mat34.m[1][0], mat34.m[2][0], 0.0,
		mat34.m[0][1], mat34.m[1][1], mat34.m[2][1], 0.0,
		mat34.m[0][2], mat34.m[1][2], mat34.m[2][2], 0.0,
		mat34.m[0][3], mat34.m[1][3], mat34.m[2][3], 1.0f
	);
	return matrix;
}

static osg::Matrix convertMatrix44(const vr::HmdMatrix44_t &mat44)
{
	osg::Matrix matrix(
		mat44.m[0][0], mat44.m[1][0], mat44.m[2][0], mat44.m[3][0],
		mat44.m[0][1], mat44.m[1][1], mat44.m[2][1], mat44.m[3][1],
		mat44.m[0][2], mat44.m[1][2], mat44.m[2][2], mat44.m[3][2],
		mat44.m[0][3], mat44.m[1][3], mat44.m[2][3], mat44.m[3][3]
	);
	return matrix;
}

Vive::Vive()
	: InputDevice("COVER.Input.Device.Vive")
{
	Input::instance()->addDevice("Vive", this);
	haveTrackerOrigin = false;
	LighthouseMatrix.makeIdentity();
}

bool Vive::needsThread() const
{
	return false;
} 
bool Vive::init()
{
	fprintf(stderr, "Vive::init\n");

	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	ivrSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None)
	{
		ivrSystem = nullptr;
		fprintf(stderr, "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		return false;
	}

	if (!vr::VRCompositor())
	{
		ivrSystem = nullptr;
		vr::VR_Shutdown();
		osg::notify(osg::WARN) << "Error: Compositor initialization failed" << std::endl;
		return false;
	}

	ivrRenderModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &eError);
	if (ivrRenderModels == nullptr)
	{
		ivrSystem = nullptr;
		vr::VR_Shutdown();
		osg::notify(osg::WARN)
			<< "Error: Unable to get render model interface!\n"
			<< "Reason: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
		return false;
	}

	m_strDriver = "No Driver";
	m_strDisplay = "No Display";

	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		m_rDevClassChar[nDevice] = 0;
	}
	numControllers = 0;

	m_strDriver = GetTrackedDeviceString(ivrSystem, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
	m_strDisplay = GetTrackedDeviceString(ivrSystem, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);

	return true;
}

// this is called if the plugin is removed at runtime
Vive::~Vive()
{
	fprintf(stderr, "Vive::~Vive\n");
}

void Vive::preFrame()
{
	// Process SteamVR events
	vr::VREvent_t event;
	while (ivrSystem->PollNextEvent(&event, sizeof(event)))
	{
		//ProcessVREvent(event);
	}

	vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		if (m_rDevClassChar[nDevice] == 0)
		{
			switch (ivrSystem->GetTrackedDeviceClass(nDevice))
			{
			case vr::TrackedDeviceClass_Controller:        m_rDevClassChar[nDevice] = 'C'; numControllers++; break;
			case vr::TrackedDeviceClass_HMD:               m_rDevClassChar[nDevice] = 'H'; break;
			case vr::TrackedDeviceClass_Invalid:           m_rDevClassChar[nDevice] = 'I'; break;
			case vr::TrackedDeviceClass_GenericTracker:    m_rDevClassChar[nDevice] = 'G'; break;
			case vr::TrackedDeviceClass_TrackingReference: m_rDevClassChar[nDevice] = 'T'; break;
			default:                                       m_rDevClassChar[nDevice] = '?'; break;
			}
		}
		if (m_rTrackedDevicePose[nDevice].bPoseIsValid)
		{
			maxBodyNumber = nDevice;
		}
	}
	if (maxBodyNumber + 1 > m_bodyMatrices.size())
	{
		m_mutex.lock();
		m_bodyMatrices.resize(maxBodyNumber + 1);
		m_bodyMatricesValid.resize(maxBodyNumber + 1);
		m_mutex.unlock();
	}
	if (numControllers * 4 > m_buttonStates.size())
	{
		m_buttonStates.resize(numControllers * 4);
	}

	// Process SteamVR controller state
	size_t controllerNumber = 0;
	for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++)
	{
		vr::VRControllerState_t state;
		bool gotState = ivrSystem->GetControllerState(unDevice, &state, sizeof(state));
		if ((m_rDevClassChar[unDevice] == 'C'))
		{
			m_buttonStates[(controllerNumber * 4) + 0] = ((state.ulButtonPressed & ((uint64_t)1 << 33)) != 0);
			m_buttonStates[(controllerNumber * 4) + 1] = ((state.ulButtonPressed & ((uint64_t)1 << 32)) != 0);
			m_buttonStates[(controllerNumber * 4) + 2] = ((state.ulButtonPressed & ((uint64_t)1 << 1)) != 0);
			m_buttonStates[(controllerNumber * 4) + 3] = ((state.ulButtonPressed & ((uint64_t)1 << 2)) != 0);
			controllerNumber++;
		}
	}

	m_mutex.lock();

	for (int nDevice = 0; nDevice < maxBodyNumber + 1; ++nDevice)
	{
		m_bodyMatricesValid[nDevice] = m_rTrackedDevicePose[nDevice].bPoseIsValid;
		if (m_rTrackedDevicePose[nDevice].bPoseIsValid)
		{
			m_bodyMatrices[nDevice] = convertMatrix34(m_rTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
		    // convert to mm
			m_bodyMatrices[nDevice](3, 0) *= 1000;
			m_bodyMatrices[nDevice](3, 1) *= 1000;
			m_bodyMatrices[nDevice](3, 2) *= 1000;
			m_bodyMatrices[nDevice] *= LighthouseMatrix; // transform to first Lighthouse coordinate system as this is fixed in our case
		}
	}
	if (!haveTrackerOrigin && (m_rDevClassChar[1] == 'T'))
	{
		haveTrackerOrigin = true;
		LighthouseMatrix.invert_4x4(m_bodyMatrices[1]);
	}
	m_mutex.unlock();
}


//-----------------------------------------------------------------------------
// Purpose: Helper to get a string from a tracked device property and turn it
//			into a std::string
//-----------------------------------------------------------------------------
std::string Vive::GetTrackedDeviceString(vr::IVRSystem *pHmd, vr::TrackedDeviceIndex_t unDevice, vr::TrackedDeviceProperty prop, vr::TrackedPropertyError *peError)
{
	uint32_t unRequiredBufferLen = pHmd->GetStringTrackedDeviceProperty(unDevice, prop, NULL, 0, peError);
	if (unRequiredBufferLen == 0)
		return "";

	char *pchBuffer = new char[unRequiredBufferLen];
	unRequiredBufferLen = pHmd->GetStringTrackedDeviceProperty(unDevice, prop, pchBuffer, unRequiredBufferLen, peError);
	std::string sResult = pchBuffer;
	delete[] pchBuffer;
	return sResult;
}

COVERPLUGIN(Vive)

