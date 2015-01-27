/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//******************************************
// class coVRTrackingUtil
// intitialize and read input device
// Date: 2007-10-16
//******************************************

#include <util/common.h>

#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include "coVRTrackingUtil.h"
#include <cover/coVRPluginList.h>

#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <osg/Matrix>
#include <ctype.h>

using namespace covise;
using namespace opencover;
int maxInt(int a, int b) { return (((a) > (b)) ? (a) : (b)); };
coVRTrackingUtil *coVRTrackingUtil::myInstance = NULL;

// constructor
coVRTrackingUtil::coVRTrackingUtil()
{
    trackingSystem = T_MOUSE;
    //set current tracking system according to information given in a config-file
    std::string systemname = coCoviseConfig::getEntry("COVER.Input.TrackingSystem");
    std::transform(systemname.begin(), systemname.end(), systemname.begin(), static_cast<int (*)(int)>(toupper));
    if (!systemname.empty())
    {
        if (systemname.compare("POLHEMUS") == 0)
        {
            trackingSystem = T_POLHEMUS;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system POLHEMUS (Fastrak)\n");
        }
        else if (systemname.compare("DTRACK") == 0)
        {
            trackingSystem = T_DTRACK;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system DTRACK (A.R.T)\n");
        }
        else if (systemname.compare("TARSUS") == 0)
        {
            trackingSystem = T_TARSUS;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system TARSUS (Vicon)\n");
        }
        else if (systemname.compare("SSD") == 0)
        {
            trackingSystem = T_SSD;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system SSD (PVR Personal Space Station)\n");
        }
        else if (systemname.compare("VRPN") == 0)
        {
            trackingSystem = T_VRPN;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system VRPN (Virtual Reality Peripheral Network)\n");
        }
        else if (systemname.compare("DYNASIGHT") == 0)
        {
            trackingSystem = T_DYNASIGHT;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system Origin Instruments DynaSight\n");
        }
        else if (systemname.compare("SEEREAL") == 0)
        {
            trackingSystem = T_SEEREAL;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system SEEREAL (Dresden 3D Display)\n");
        }
        else if (systemname.compare("CAVELIB") == 0)
        {
            trackingSystem = T_CAVELIB;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system CAVELIB\n");
        }
        else if (systemname.compare("FOB") == 0)
        {
            trackingSystem = T_FOB;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system FOB (Ascension Flock of Birds)\n");
        }
        else if (systemname.compare("VRC") == 0)
        {
            trackingSystem = T_VRC_DAEMON;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system VRC (VirCinity tracker daemon)\n");
        }
        else if (systemname.compare("FLOCK") == 0)
        {
            trackingSystem = T_FOB;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system FLOCK (old name for FOB- use FOB now)\n");
        }
        else if (systemname.compare("MOTIONSTAR") == 0)
        {
            trackingSystem = T_MOTIONSTAR;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system MOTIONSTAR (Ascension)\n");
        }
        else if (systemname.compare("SPACEPOINTER") == 0)
        {
            trackingSystem = T_SPACEPOINTER;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system SPACEPOINTER\n");
        }
        else if (systemname.compare("MOUSE") == 0)
        {
            trackingSystem = T_MOUSE;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system MOUSE\n");
        }
        else if (systemname.compare("SPACEBALL") == 0)
        {
            trackingSystem = T_SPACEBALL;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system SPACEBALL\n");
        }

        else if (systemname.compare("COVER_BEEBOX") == 0)
        {
            trackingSystem = T_COVER_BEEBOX;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system COVER_BEEBOX\n");
        }

        else if (systemname.compare("COVER_FLYBOX") == 0)
        {
            trackingSystem = T_COVER_FLYBOX;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system COVER_FLYBOX\n");
        }
        else if (systemname.compare("PHANTOM") == 0)
        {
            trackingSystem = T_PHANTOM;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system PHANTOM\n");
        }
        else if (systemname.compare("NONE") == 0)
        {
            trackingSystem = T_NONE;
            if (cover->debugLevel(2))
                fprintf(stderr, "\ttracking system NONE\n");
        }
        else
        {
            trackingSystemPlugin = coVRPluginList::instance()->addPlugin(systemname.c_str());
            if (trackingSystemPlugin)
            {
                trackingSystem = T_PLUGIN;
            }
        }
    }
    else
    {
        trackingSystem = T_MOUSE;
    }

    numStations = 0;
    for (int i = 0; i < numDevices; i++)
    {
        deviceAddress[i] = 0;
    }

    // set addresses for the different devices according to information given in a config-file
    deviceAddress[headDev] = coCoviseConfig::getInt("COVER.Input.HeadAddress", -1);
    deviceAddress[handDev] = coCoviseConfig::getInt("COVER.Input.HandAddress", -1);
    deviceAddress[secondHandDev] = coCoviseConfig::getInt("COVER.Input.SecondHandAddress", -1);

    if (deviceAddress[handDev] == -1)
    {
        if (trackingSystem == T_MOUSE)
        {
            deviceAddress[handDev] = 0;
        }
    }

    deviceAddress[secondHandDev] = coCoviseConfig::getInt("COVER.Input.SecondHandAddress", -1);

    deviceAddress[worldDev] = coCoviseConfig::getInt("COVER.Input.WorldXFormAddress", -1);
    deviceAddress[hmdDev] = coCoviseConfig::getInt("COVER.Input.HMDAddress", -1);
    deviceAddress[cameraDev] = coCoviseConfig::getInt("COVER.Input.CameraAddress", -1);
    deviceAddress[objectDev] = coCoviseConfig::getInt("COVER.Input.ObjectAddress", -1);

    for (int i = 0; i < numDevices; i++)
    {

        //cerr << "address : "<< deviceAddress[i] << " device " << i << endl;
        if (deviceAddress[i] >= numStations)
            numStations = deviceAddress[i] + 1;
    }

    numStations = coCoviseConfig::getInt("COVER.Input.NumSensors", numStations);

    if (numStations < 0) // || numStations > numDevices don't if we have ART Wands with IDs > 10 this won't work
        numStations = numDevices - 1;

#ifdef OLDINPUT
    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((const char *)&numStations, sizeof(int));
        coVRMSController::instance()->sendSlaves((const char *)&deviceAddress, sizeof(int) * numDevices);
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&numStations, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&deviceAddress, sizeof(int) * numDevices);
    }
#endif

    readOffset();
    createTUI();
}

bool coVRTrackingUtil::hasHead() const
{
    if (((trackingSystem == T_POLHEMUS)
         || (trackingSystem == T_FOB)
         || (trackingSystem == T_MOTIONSTAR)
         || (trackingSystem == T_DTRACK)
         || (trackingSystem == T_PLUGIN)
         || (trackingSystem == T_TARSUS)
         || (trackingSystem == T_SSD)
         || (trackingSystem == T_VRPN)
         || (trackingSystem == T_DYNASIGHT)
         || (trackingSystem == T_CAVELIB)
         || (trackingSystem == T_SEEREAL)
         || (trackingSystem == T_VRC_DAEMON))
        && haveDevice(coVRTrackingUtil::headDev))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool coVRTrackingUtil::hasHand() const
{
    if (((trackingSystem == T_POLHEMUS)
         || (trackingSystem == T_FOB)
         || (trackingSystem == T_MOTIONSTAR)
         || (trackingSystem == T_DTRACK)
         || (trackingSystem == T_PLUGIN)
         || (trackingSystem == T_TARSUS)
         || (trackingSystem == T_SSD)
         || (trackingSystem == T_VRPN)
         || (trackingSystem == T_CAVELIB)
         || (trackingSystem == T_SEEREAL)
         || (trackingSystem == T_VRC_DAEMON))
        && haveDevice(coVRTrackingUtil::handDev))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool coVRTrackingUtil::hasSecondHand() const
{
    if (((trackingSystem == T_POLHEMUS)
         || (trackingSystem == T_FOB)
         || (trackingSystem == T_MOTIONSTAR)
         || (trackingSystem == T_DTRACK)
         || (trackingSystem == T_TARSUS)
         || (trackingSystem == T_SSD)
         || (trackingSystem == T_VRPN)
         || (trackingSystem == T_CAVELIB)
         || (trackingSystem == T_SEEREAL)
         || (trackingSystem == T_VRC_DAEMON))
        && haveDevice(coVRTrackingUtil::secondHandDev))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// destructor
coVRTrackingUtil::~coVRTrackingUtil()
{
    delete deviceChoice;
    delete deviceLabel;
    for (int i = 0; i < 3; i++)
    {
        delete trans[i];
        delete transLabel[i];
        delete rot[i];
        delete rotLabel[i];
    }
    delete trackingTab;
}

const char *coVRTrackingUtil::deviceNames[numDevices] = { "TrackingSystem", "HeadDevice", "HandDevice", "SecondHandDevice", "WorldDevice", "HMDDevice", "CameraDevice", "ObjectDevice" };
const char *coVRTrackingUtil::cmDeviceNames[numDevices] = { "Transmitter", "HeadSensor", "HandSensor", "SecondHandSensor", "WorldSensor", "HMDSensor", "CameraSensor", "ObjectSensor" };
int coVRTrackingUtil::deviceAddress[numDevices];

void coVRTrackingUtil::createTUI()
{
    trackingTab = new coTUITab("Tracking", coVRTui::instance()->mainFolder->getID());
    trackingTab->setPos(0, 0);
    deviceChoice = new coTUIComboBox("device", trackingTab->getID());
    deviceChoice->setPos(1, 0);
    deviceChoice->setEventListener(this);
    for (int i = 0; i < numDevices; i++)
    {
        deviceChoice->addEntry(deviceNames[i]);
    }
    deviceLabel = new coTUILabel("device:", trackingTab->getID());
    deviceLabel->setPos(0, 0);
    trans[0] = new coTUIEditFloatField("xe", trackingTab->getID());
    transLabel[0] = new coTUILabel("x", trackingTab->getID());
    trans[1] = new coTUIEditFloatField("ye", trackingTab->getID());
    transLabel[1] = new coTUILabel("y", trackingTab->getID());
    trans[2] = new coTUIEditFloatField("ze", trackingTab->getID());
    transLabel[2] = new coTUILabel("z", trackingTab->getID());
    rot[0] = new coTUIEditFloatField("he", trackingTab->getID());
    rotLabel[0] = new coTUILabel("h", trackingTab->getID());
    rot[1] = new coTUIEditFloatField("pe", trackingTab->getID());
    rotLabel[1] = new coTUILabel("p", trackingTab->getID());
    rot[2] = new coTUIEditFloatField("re", trackingTab->getID());
    rotLabel[2] = new coTUILabel("r", trackingTab->getID());
    for (int i = 0; i < 3; i++)
    {
        trans[i]->setPos(i * 2, 1);
        transLabel[i]->setPos(i * 2 + 1, 1);
        trans[i]->setEventListener(this);
        rot[i]->setPos(i * 2, 2);
        rotLabel[i]->setPos(i * 2 + 1, 2);
        rot[i]->setEventListener(this);
    }
    currentDevice = 0;
    updateTUI();
}

void coVRTrackingUtil::updateTUI()
{
    for (int i = 0; i < 3; i++)
    {
        trans[i]->setValue(deviceOffsets[currentDevice].trans[i]);
        rot[i]->setValue(deviceOffsets[currentDevice].rot[i]);
    }
}

void coVRTrackingUtil::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == deviceChoice)
    {
        currentDevice = deviceChoice->getSelectedEntry();
        updateTUI();
    }
    for (int i = 0; i < 3; i++)
    {
        if (tUIItem == trans[i])
        {
            deviceOffsets[currentDevice].trans[i] = trans[i]->getValue();
        }
        if (tUIItem == rot[i])
        {
            deviceOffsets[currentDevice].rot[i] = rot[i]->getValue();
        }
    }
}

void coVRTrackingUtil::setDeviceOffset(IDOfDevice device_ID, osg::Vec3 &t, osg::Vec3 &r)
{
    for (int i = 0; i < 3; i++)
    {
        deviceOffsets[device_ID].trans[i] = t[i];
        deviceOffsets[device_ID].rot[i] = r[i];
        if (currentDevice == device_ID)
        {
            trans[i]->setValue(t[i]);
            rot[i]->setValue(r[i]);
        }
    }
}

float *coVRTrackingUtil::getDeviceOffsetTrans(IDOfDevice device_ID)
{
    return deviceOffsets[device_ID].trans;
}

float *coVRTrackingUtil::getDeviceOffsetRot(IDOfDevice device_ID)
{
    return deviceOffsets[device_ID].rot;
}

//--------------------------------------------------------

int coVRTrackingUtil::readOffset()
{
    // control variables
    bool foundCmDevice = false;
    bool foundMmDevice = false;
    bool useCmDevices = false;

    const char *entries[2] = { "Offset", "Orientation" };
    // check for mm device entries
    for (int val = 0; val < 2; ++val)
    {
        for (int i = 0; i < numDevices; ++i)
        {
            char configName[1000];
            sprintf(configName, "COVER.Input.%s.%s", deviceNames[i], entries[val]);

            bool exist_1 = false, exist_2 = false, exist_3 = false;
            coCoviseConfig::getFloat("x", configName, 0., &exist_1);
            coCoviseConfig::getFloat("y", configName, 0., &exist_2);
            coCoviseConfig::getFloat("z", configName, 0., &exist_3);

            foundMmDevice |= exist_1;
            foundMmDevice |= exist_2;
            foundMmDevice |= exist_3;
        }
    }

    // check for cm device entries
    for (int val = 0; val < 2; ++val)
    {
        for (int i = 0; i < numDevices; ++i)
        {
            char configName[1000];
            sprintf(configName, "COVER.Input.%s.%s", cmDeviceNames[i], entries[val]);

            bool exist_1 = false, exist_2 = false, exist_3 = false;
            coCoviseConfig::getFloat("x", configName, 0., &exist_1);
            coCoviseConfig::getFloat("y", configName, 0., &exist_2);
            coCoviseConfig::getFloat("z", configName, 0., &exist_3);

            foundCmDevice |= exist_1;
            foundCmDevice |= exist_2;
            foundCmDevice |= exist_3;
        }
    }

    if (foundMmDevice && foundCmDevice)
    {
        fprintf(stderr, " WARNING: found both new (unit: mm) and old (unit: cm) device names\n");
        fprintf(stderr, " WARNING: ignoring all old device names:");
        for (int i = 0; i < numDevices; ++i)
        {
            fprintf(stderr, " %s", cmDeviceNames[i]);
        }
        fprintf(stderr, "\n");
    }
    else if (foundCmDevice)
    {
        fprintf(stderr, " WARNING: found deprecated (unit: cm) device names\n");
        useCmDevices = true;
    }

    // read config
    for (int i = 0; i < numDevices; i++)
    {
        char configName[1000];
        sprintf(configName, "COVER.Input.%s.Offset", useCmDevices ? cmDeviceNames[i] : deviceNames[i]);
        // TrackingSystem Position Offset
        deviceOffsets[i].trans[0] = coCoviseConfig::getFloat("x", configName, 0.);
        deviceOffsets[i].trans[1] = coCoviseConfig::getFloat("y", configName, 0.);
        deviceOffsets[i].trans[2] = coCoviseConfig::getFloat("z", configName, 0.);
        if (useCmDevices)
        {
            for (int j = 0; j < 3; ++j)
                deviceOffsets[i].trans[j] *= 10.;
        };

        // TrackingSystem Orientation Offset
        sprintf(configName, "COVER.Input.%s.Orientation", useCmDevices ? cmDeviceNames[i] : deviceNames[i]);
        deviceOffsets[i].rot[0] = coCoviseConfig::getFloat("h", configName, 0.);
        deviceOffsets[i].rot[1] = coCoviseConfig::getFloat("p", configName, 0.);
        deviceOffsets[i].rot[2] = coCoviseConfig::getFloat("r", configName, 0.);
    }

    return (0);
}

//--------------------------------------------------------
// returns the computed transformation matrix for the HeadDevice
osg::Matrix coVRTrackingUtil::computeDeviceOffsetMat(IDOfDevice device_ID)
{
    (void)device_ID;
    osg::Matrix deviceOffsetMat;
    osg::Matrix translationMat;

    MAKE_EULER_MAT(deviceOffsetMat, deviceOffsets[device_ID].rot[0], deviceOffsets[device_ID].rot[1], deviceOffsets[device_ID].rot[2]);
    //fprintf(stderr, "offset from device('%d) %f %f %f\n", device_ID, deviceOffsets[device_ID].trans[0], deviceOffsets[device_ID].trans[1], deviceOffsets[device_ID].trans[2]);
    translationMat.makeTranslate(deviceOffsets[device_ID].trans[0], deviceOffsets[device_ID].trans[1], deviceOffsets[device_ID].trans[2]);
    deviceOffsetMat.postMult(translationMat);
    return deviceOffsetMat;
}

// EOF
