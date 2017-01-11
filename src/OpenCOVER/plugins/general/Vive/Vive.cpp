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

using namespace opencover;

Vive::Vive()
{
}

bool Vive::init()
{
	fprintf(stderr, "Vive::init\n");

	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	ivrSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None)
	{
		ivrSystem = NULL;
		fprintf(stderr, "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		return false;
	}

	m_strDriver = "No Driver";
	m_strDisplay = "No Display";

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

	// Process SteamVR controller state
	for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++)
	{
		vr::VRControllerState_t state;
		if (ivrSystem->GetControllerState(unDevice, &state, sizeof(state)))
		{
			//m_rbShowTrackedDevice[unDevice] = state.ulButtonPressed == 0;
		}
	}

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

