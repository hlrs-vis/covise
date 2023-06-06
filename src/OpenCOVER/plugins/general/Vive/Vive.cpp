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
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>
#include <config/CoviseConfig.h>
#include <osg/Texture2D>
#include <osg/GraphicsContext>

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
: coVRPlugin(COVER_PLUGIN_NAME)
, InputDevice("COVER.Input.Device.Vive")
{
	Input::instance()->addDevice("Vive", this);
	haveTrackerOrigin = false;
	LighthouseMatrix.makeIdentity();
	maxBodyNumber = 0;

	m_strDriver = "No Driver";
	m_strDisplay = "No Display";
	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	ivrSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None)
	{
		ivrSystem = nullptr;
		fprintf(stderr, "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		return;
	}

	if (!vr::VRCompositor())
	{
		ivrSystem = nullptr;
		vr::VR_Shutdown();
		osg::notify(osg::WARN) << "Error: Compositor initialization failed" << std::endl;
		return;
	}

	ivrRenderModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &eError);
	if (ivrRenderModels == nullptr)
	{
		ivrSystem = nullptr;
		vr::VR_Shutdown();
		osg::notify(osg::WARN)
			<< "Error: Unable to get render model interface!\n"
			<< "Reason: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
		return;
	}


	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		m_rDevClassChar[nDevice] = 0;
	}
	numControllers = 0;

	m_strDriver = GetTrackedDeviceString(ivrSystem, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_TrackingSystemName_String);
	m_strDisplay = GetTrackedDeviceString(ivrSystem, vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SerialNumber_String);

	bool exists;
    trackingOnly = covise::coCoviseConfig::isOn("trackingOnly", "COVER.Plugin.Vive", false, &exists);
	if (!trackingOnly)
	{
		if (coVRConfig::instance()->numPBOs() == 0)
		{ // no PBOs configured, thus try an outo config
			PBOStruct pbol;
			PBOStruct pbor;
			uint32_t sx, sy;
			ivrSystem->GetRecommendedRenderTargetSize(&sx, &sy);
			pbol.PBOsx = pbor.PBOsx = sx;
			pbol.PBOsy = pbor.PBOsy = sy;
			pbor.windowNum = 0;
			pbol.windowNum = 0;
			coVRConfig::instance()->PBOs.push_back(pbol);
			coVRConfig::instance()->PBOs.push_back(pbor);

			channelStruct chanl;
			channelStruct chanr;
			chanl.fixedViewer = false;
			chanl.name = "ViveL";
			chanl.PBONum = 0;
			chanl.screenNum = 0;
			chanl.stereo = true;
			chanl.stereoMode = osg::DisplaySettings::LEFT_EYE;
			chanl.viewportNum = -1;

			chanr.fixedViewer = false;
			chanr.name = "ViveR";
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

bool Vive::needsThread() const
{
	return false;
} 
bool Vive::init()
{
    if (!ivrSystem)
    {
        fprintf(stderr, "Vive::init() failed -- ivrSystem is null\n");
        return false;
    }

    fprintf(stderr, "Vive::init\n");
    vr::HmdMatrix44_t mat = ivrSystem->GetProjectionMatrix(vr::Eye_Left, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
    osg::Matrix lProj = convertMatrix44(mat);
    mat = ivrSystem->GetProjectionMatrix(vr::Eye_Right, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
    osg::Matrix rProj = convertMatrix44(mat);
    coVRConfig::instance()->channels[0].leftProj = lProj;
    coVRConfig::instance()->channels[0].rightProj = rProj;
    if (coVRConfig::instance()->channels.size() > 1)
    {
        coVRConfig::instance()->channels[1].leftProj = lProj;
        coVRConfig::instance()->channels[1].rightProj = rProj;
    }


	coVRConfig::instance()->OpenVR_HMD = true;

	m_transformOriginToLighthouse =covise::coCoviseConfig::isOn("COVER.Input.Device.Vive.TransformOriginToLighthouse",false);


	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		m_DeviceID[nDevice] = -1;  // cover ID for a specific DeviceID
		m_ControllerID[nDevice] = 0;
		m_DeviceSerial[nDevice] = "";   // for each device, a character representing its class // index is device ID
	}

	int numSerialConfigs = 0;
	char str[200];
	bool exists=false;
	do {
		sprintf(str, "COVER.Input.Device.Vive.Device:%d", numSerialConfigs);
		std::string serial = covise::coCoviseConfig::getEntry("serial", str, "", &exists);
		serialInfo si;
		si.ID = covise::coCoviseConfig::getInt("index", str, 0);
		si.controllerID = covise::coCoviseConfig::getInt("controllerID", str, 0);
		serialID[serial] = si;
		numSerialConfigs++;
	} while (exists);
	


	if (m_strDriver == "No Driver")
		return false;
	

	return true;
}

// this is called if the plugin is removed at runtime
Vive::~Vive()
{
	vr::VR_Shutdown();
	Input::instance()->removeDevice("Vive", this);
	fprintf(stderr, "Vive::~Vive\n");
}

void Vive::preFrame()
{
	// Process SteamVR events
	vr::VREvent_t event;
	while (ivrSystem->PollNextEvent(&event, sizeof(event)))
	{
		//ProcessVREvent(event);
		if (event.eventType == vr::VREvent_IpdChanged)
		{
			VRViewer::instance()->setSeparation(event.data.ipd.ipdMeters * 1000.0);
		}
	}
	vr::HmdMatrix44_t mat = ivrSystem->GetProjectionMatrix(vr::Eye_Left, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
	osg::Matrix lProj = convertMatrix44(mat);
	mat = ivrSystem->GetProjectionMatrix(vr::Eye_Right, coVRConfig::instance()->nearClip(), coVRConfig::instance()->farClip());
	osg::Matrix rProj = convertMatrix44(mat);
    if(!trackingOnly)
    {
        if (coVRConfig::instance()->channels.size() > 1)
        {
            coVRConfig::instance()->channels[0].leftProj = lProj;
            coVRConfig::instance()->channels[0].rightProj = rProj;
            coVRConfig::instance()->channels[1].leftProj = lProj;
            coVRConfig::instance()->channels[1].rightProj = rProj;
        }
        else
        {
            fprintf(stderr, "configure two channes\n");
        }
    }

}

void Vive::postFrame()
{


    size_t controllerNumber = 0;
    size_t trackerNumber = 0;
	size_t baseStationNumber = 0;

	vr::VRCompositor()->WaitGetPoses(m_rTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);
	for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	{
		// device configuration can be changed on-the-fly!
		// map devices to numbers depending on their serial number
		std::string serial = GetTrackedDeviceString(ivrSystem, nDevice, vr::Prop_SerialNumber_String);
		if (m_DeviceSerial[nDevice] != serial)
		{
			// new device or device changed
			switch (ivrSystem->GetTrackedDeviceClass(nDevice))
			{
			    case vr::TrackedDeviceClass_Controller:        m_rDevClassChar[nDevice] = 'C'; numControllers++; break;
			    case vr::TrackedDeviceClass_HMD:               m_rDevClassChar[nDevice] = 'H'; break;
			    case vr::TrackedDeviceClass_Invalid:           m_rDevClassChar[nDevice] = 'I'; break;
                case vr::TrackedDeviceClass_GenericTracker:    m_rDevClassChar[nDevice] = 'G'; numTrackers++;  break;
                case vr::TrackedDeviceClass_TrackingReference: m_rDevClassChar[nDevice] = 'T'; numBaseStations++; break;
			    default:                                       m_rDevClassChar[nDevice] = '?'; break;
			}
            fprintf(stderr, "DevClass:%c\n", m_rDevClassChar[nDevice]);
			int idx;
			std::map<std::string, serialInfo>::iterator it = serialID.find(std::string(serial));
			if (it != serialID.end())
			{
				idx = it->second.ID;
				controllerNumber = it->second.controllerID+1; // +1 because we remove 1 afterwards (see below)
				if (controllerNumber > numControllers)
					numControllers = controllerNumber;
			}
			else
			{

				// m_bodyMatrices device order:
				// 0 -- HMD
				// 1 -- first base station
				// 2 -- last base station
				// 3 -- first Vive controller
				// 4 -- second Vive controller
				size_t firstBaseStationIdx = 1, lastBaseStationIdx = 2, firstControllerIdx = 3, firstTrackerIdx = 5;

				switch (m_rDevClassChar[nDevice])
				{
				case 'T': // a lighthouse
					idx = firstBaseStationIdx + baseStationNumber;
					++baseStationNumber;
					if (idx > lastBaseStationIdx)
					{
                        lastBaseStationIdx++;
                        firstControllerIdx++;
                        firstTrackerIdx++;
						//cerr << "Vive:Too many baseStations;number=" << baseStationNumber - 1 << " idx= " << idx << endl;
						//continue;
					}
					break;
				case 'C': //a controller
					idx = firstControllerIdx + controllerNumber;
					++controllerNumber;
					break;
                case 'G': //a controller
                    idx = firstTrackerIdx + trackerNumber;
                    ++trackerNumber;
                    break;
				case 'H':// the HMD
					idx = 0;
					break;
				case '?':// whatever
					break;
				default:
					cerr << "Vive:Unsupported device class:" << m_rDevClassChar[nDevice] << "  nDevice=" << nDevice << endl;
					continue;
				}

				cerr << "Vive: unconfigured device! add the following config entry to assure persistant device mappings" << endl;
				cerr << " <Device serial = \""<< serial << "\" index = \""<< idx << "\" controllerID = \"" << (controllerNumber-1) << "\" name=\"<number from 0 to n>\" />" << endl;
			}
			m_DeviceID[nDevice] = idx;
			m_ControllerID[nDevice] = (controllerNumber-1);
			m_DeviceSerial[nDevice] = serial;
		}
			if ((m_DeviceID[nDevice]+1) > maxBodyNumber)
            {
				maxBodyNumber = m_DeviceID[nDevice]+1;
            }
	}


	if (maxBodyNumber > m_bodyMatrices.size())
	{
		m_mutex.lock();
		m_bodyMatrices.resize(maxBodyNumber);
		m_bodyMatricesValid.resize(maxBodyNumber);
		m_mutex.unlock();
	}
	if (numControllers * 4 > m_buttonStates.size())
	{
		m_buttonStates.resize(numControllers * 4);
	}
	bool haveBaseStation = false;
	// Process SteamVR controller state
	for (vr::TrackedDeviceIndex_t unDevice = 0; unDevice < vr::k_unMaxTrackedDeviceCount; unDevice++)
	{
		vr::VRControllerState_t state;
		bool gotState = ivrSystem->GetControllerState(unDevice, &state, sizeof(state));
		if ((m_rDevClassChar[unDevice] == 'C') && m_rTrackedDevicePose[unDevice].bPoseIsValid)
		{
			m_buttonStates[(m_ControllerID[unDevice] * 4) + 0] = ((state.ulButtonPressed & ((uint64_t)1 << 33)) != 0);
			m_buttonStates[(m_ControllerID[unDevice] * 4) + 1] = ((state.ulButtonPressed & ((uint64_t)1 << 32)) != 0);
			m_buttonStates[(m_ControllerID[unDevice] * 4) + 2] = ((state.ulButtonPressed & ((uint64_t)1 << 1)) != 0);
			m_buttonStates[(m_ControllerID[unDevice] * 4) + 3] = ((state.ulButtonPressed & ((uint64_t)1 << 2)) != 0);
		}
		if (m_rDevClassChar[unDevice] == 'T')
		{
			haveBaseStation = true;
		}
	}

	m_mutex.lock();

	for (int nDevice = 0; nDevice < maxBodyNumber + 1; ++nDevice)
	{
		if (m_rTrackedDevicePose[nDevice].bPoseIsValid)
		{
			//fprintf(stderr, "NewMatrix n %d ID %d\n",nDevice, m_DeviceID[nDevice]);
			m_bodyMatricesValid[m_DeviceID[nDevice]] = m_rTrackedDevicePose[nDevice].bPoseIsValid;
			m_bodyMatrices[m_DeviceID[nDevice]] = convertMatrix34(m_rTrackedDevicePose[nDevice].mDeviceToAbsoluteTracking);
		    // convert to mm
			m_bodyMatrices[m_DeviceID[nDevice]](3, 0) *= 1000;
			m_bodyMatrices[m_DeviceID[nDevice]](3, 1) *= 1000;
			m_bodyMatrices[m_DeviceID[nDevice]](3, 2) *= 1000;
			
			if (m_transformOriginToLighthouse)
			{
				m_bodyMatrices[m_DeviceID[nDevice]] *= LighthouseMatrix; // transform to first Lighthouse coordinate system as this is fixed in our case
			}

		}
	}
	// get the transform matrix from 1st Lighthouse if we need that
	if (!haveTrackerOrigin && haveBaseStation && m_transformOriginToLighthouse)
	{
		haveTrackerOrigin = true;
		LighthouseMatrix.invert_4x4(m_bodyMatrices[1]);
	}
	m_mutex.unlock();
}

void Vive::preSwapBuffers(int /*windowNumber*/)
{

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

