/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// class VRTracker
// intitialize and read input devices
// authors: F. Foehl, D. Rainer, A. werner, U. Woessner
// (C) 1996-2003 University of Stuttgart
// (C) 1997-2003 Vircinity GmbH
#include "coRawMouse.h"
#include <config/CoviseConfig.h>
#include <util/common.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/VRVruiRenderInterface.h>
#include <cover/coVRConfig.h>

#include "VRTracker.h"
#include "VRCTracker.h"
#include "coVRTrackingSystems.h"
#include <cover/VRSceneGraph.h>
#include "VRSpacePointer.h"
#include "coVRTrackingUtil.h"

using namespace opencover;
using namespace vrui;
using covise::coCoviseConfig;

#include "../../../OpenCOVER.h" // needed for quit event

VRTracker *VRTracker::instance()
{
    static VRTracker *singleton = NULL;
    if (!singleton)
        singleton = new VRTracker;
    return singleton;
}

VRTracker::VRTracker()
    : m_buttonState(0)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew VRTracker\n");

#ifdef WIN32
    rawMouseManager = coRawMouseManager::instance();
#endif

    doCalibrate = false;
    d_debugTracking = false;
    d_debugStation = 0;
    trackingUtil = coVRTrackingUtil::instance();
    enableTracking_ = true;

    d_saveTracking = d_loadTracking = NULL;
    doJoystick = false;
    myIdentity.makeIdentity();

    trackingSystems = NULL;
    spacePointer = NULL;
    vrcTracker = NULL;

    readConfigFile();

    if (doCalibrate)
    {
        doCalibrate = readInterpolationFile(interpolationFile);
    }

    //determinate xyz_velocity
    if (doCalibrate)
    {
        int xyz_velocity;
        xyz_velocity = find_xyz_velocity();
        switch (xyz_velocity)
        {
        case 123:
            doCalibrate = true; //allready good
            break;

        case 213:
            reorganize_data(213);
            xyz_velocity = find_xyz_velocity();
            if (xyz_velocity == 123)
            {
                //cout << "data was reorganizated so that real z change faster "
                //        << "than real y and than real x" << endl;
                doCalibrate = true;
            }
            else
            {
                //sprintf (interp_message,
                //        "Data reorganization for interpolation was not posible" );
                //cout << interp_message << endl;
                doCalibrate = false;
            }
            break;

        default:
            //sprintf (interp_message,
            //        "Wrong interpolation data file,xyz_(changing)_velocity = %d is not supported by interpolation",xyz_velocity);
            //cout << interp_message << endl;
            doCalibrate = false;
            break;
        }
    }

    //init array for the station (head, hand, world ...) matrices
    trackerStationMat = new osg::Matrix *[trackingUtil->getNumStations()];
    for (int i = 0; i < trackingUtil->getNumStations(); i++)
    {
        trackerStationMat[i] = new osg::Matrix();
        trackerStationMat[i]->makeIdentity();
    }

#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif

    // input devices directly connected to the COVER renderer host
    if ((trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_POLHEMUS)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_FOB)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_MOTIONSTAR)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_DTRACK)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_TARSUS)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SSD)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_VRPN)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_DYNASIGHT)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_CAVELIB)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_PLUGIN)
        || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SEEREAL))
    {
        if (trackingUtil->haveDevice(coVRTrackingUtil::worldDev))
        {
            trackingSystems = new coVRTrackingSystems(trackingUtil->getNumStations(), trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev), trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev), trackingUtil->getDeviceAddress(coVRTrackingUtil::worldDev));
        }
        else
        {
            trackingSystems = new coVRTrackingSystems(trackingUtil->getNumStations(), trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev), trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev));
        }

        trackingSystems->config();
    }

    // VRC tracker gets data via udp from separate device daemons
    else if (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_VRC_DAEMON)
    {
        int port = 7777;
        int debugLevel = 0;
        float scale; // factor to make cm from "unit"
        port = coCoviseConfig::getInt("COVER.Input.VRC.Port", 7777);
        debugLevel = coCoviseConfig::getInt("COVER.Input.VRC.DebugLevel", 0);
        std::string unit = coCoviseConfig::getEntry("COVER.Input.VRC.Unit");
        std::string options = coCoviseConfig::getEntry("COVER.Input.VRC.Options");
        if (unit.empty())
            scale = 1.0; // assume tracker sends cm
        else if (0 == strncasecmp(unit.c_str(), "cm", 2))
            scale = 1.0;
        else if (0 == strncasecmp(unit.c_str(), "mm", 2))
            scale = 10.0;
        else if (0 == strncasecmp(unit.c_str(), "inch", 4))
            scale = 1.0 / 2.54;
        else
        {
            if (sscanf(unit.c_str(), "%f", &scale) != 1)
            {
                cerr << "VRTracker::VRTracker:: sscanf failed" << endl;
            }
        }

        vrcTracker = new VRCTracker(port, debugLevel, scale, options.c_str());

        if (vrcTracker->isOk())
            vrcTracker->mainLoop();
        else
        {
            cerr << "could not create VRC Tracking Server - exiting"
                 << endl;
            exit(-1);
        }
    }

    // other input devices
    else if ((trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SPACEPOINTER)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_COVER_BEEBOX)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_COVER_FLYBOX)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_PHANTOM)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SPACEBALL))
    {
        spacePointer = new VRSpacePointer();
        spacePointer->init(trackingUtil->getTrackingSystem());
    }

    //Open file for saving/recording
    //Tracking info
    string saveTrackPath = coCoviseConfig::getEntry("COVER.Input.Save");
    if (!saveTrackPath.empty())
    {
        d_saveTracking = fopen(saveTrackPath.c_str(), "w");
        if (d_saveTracking == NULL)
        {
            fprintf(stderr, "Could not open %s", saveTrackPath.c_str());
        }
    }

    //Open file for recording tracking info
    std::string loadTrackPath = coCoviseConfig::getEntry("COVER.Input.Load");
    if (!loadTrackPath.empty())
    {
        d_loadTracking = fopen(loadTrackPath.c_str(), "r");
        if (d_loadTracking == NULL)
        {
            fprintf(stderr, "Could not open %s", loadTrackPath.c_str());
        }
        else
        {
            fprintf(stderr, "Successfully opened %s", loadTrackPath.c_str());
        }
    }
}

VRTracker::~VRTracker()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete VRTracker\n");

    for (int i = 0; i < trackingUtil->getNumStations(); i++)
        delete trackerStationMat[i];

    delete[] trackerStationMat;

    delete trackingSystems;
    delete spacePointer;
    if (NULL != d_saveTracking)
    {
        fclose(d_saveTracking);
    }
    if (NULL != d_saveTracking)
    {
        fclose(d_loadTracking);
    }

    if (vrcTracker)
        delete vrcTracker;
}

void VRTracker::setButtonState(unsigned int s)
{
    m_buttonState = s;
}

void
VRTracker::update()
{
#ifdef WIN32
    rawMouseManager->update();
#endif

    if (!isTrackingOn())
        return;

    if (cover->debugLevel(5))
        fprintf(stderr, "\nupdate VRTracker\n");

#ifdef OLDINPUT
    if (!coVRMSController::instance()->isSlave())
#endif
    {
        if (d_loadTracking)
        {
            //Get tracking info from file
            int button;
            if (trackingUtil->haveDevice(coVRTrackingUtil::headDev))
            {
                int headSensorStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev);
                int handSensorStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev);
                if (handSensorStation >= 0 && headSensorStation >= 0)
                {
                    int iret = fscanf(d_loadTracking, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d",
                                      &(*trackerStationMat[headSensorStation])(0, 0), &(*trackerStationMat[headSensorStation])(0, 1), &(*trackerStationMat)[headSensorStation](0, 2),
                                      &(*trackerStationMat[headSensorStation])(1, 0), &(*trackerStationMat[headSensorStation])(1, 1), &(*trackerStationMat[headSensorStation])(1, 2),
                                      &(*trackerStationMat[headSensorStation])(2, 0), &(*trackerStationMat[headSensorStation])(2, 1), &(*trackerStationMat[headSensorStation])(2, 2),
                                      &(*trackerStationMat[headSensorStation])(3, 0), &(*trackerStationMat[headSensorStation])(3, 1), &(*trackerStationMat[headSensorStation])(3, 2),
                                      &(*trackerStationMat[handSensorStation])(0, 0), &(*trackerStationMat[handSensorStation])(0, 1), &(*trackerStationMat[handSensorStation])(0, 2),
                                      &(*trackerStationMat[handSensorStation])(1, 0), &(*trackerStationMat[handSensorStation])(1, 1), &(*trackerStationMat[handSensorStation])(1, 2),
                                      &(*trackerStationMat[handSensorStation])(2, 0), &(*trackerStationMat[handSensorStation])(2, 1), &(*trackerStationMat[handSensorStation])(2, 2),
                                      &(*trackerStationMat[handSensorStation])(3, 0), &(*trackerStationMat[handSensorStation])(3, 1), &(*trackerStationMat[handSensorStation])(3, 2), &button);
                    if (iret != 25)
                        fprintf(stderr, "VRTracker::update() fscanf failed no = %d", iret);
                }
            }
            else
            {
                int handSensorStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev);
                if (handSensorStation >= 0)
                {
                    int iret = fscanf(d_loadTracking, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d",
                                      &(*trackerStationMat[handSensorStation])(0, 0), &(*trackerStationMat[handSensorStation])(0, 1), &(*trackerStationMat[handSensorStation])(0, 2),
                                      &(*trackerStationMat[handSensorStation])(1, 0), &(*trackerStationMat[handSensorStation])(1, 1), &(*trackerStationMat[handSensorStation])(1, 2),
                                      &(*trackerStationMat[handSensorStation])(2, 0), &(*trackerStationMat[handSensorStation])(2, 1), &(*trackerStationMat[handSensorStation])(2, 2),
                                      &(*trackerStationMat[handSensorStation])(3, 0), &(*trackerStationMat[handSensorStation])(3, 1), &(*trackerStationMat[handSensorStation])(3, 2), &button);

                    if (iret != 13)
                        fprintf(stderr, "VRTracker::update() fscanf2 failed no = %d", iret);
                }
            }
#ifdef OLDINPUT
            cover->getPointerButton()->setState(button);
#endif
            setButtonState(button);
        }
        else
        {
            for (int i = 1; i < trackingUtil->numDevices; i++)
            {
                updateDevice(i);
            }
            updateHead();
            updateHand();
            updateWheel();
        }
    }
}

void
VRTracker::reset()
{
#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif
    if (trackingSystems)
    {
        trackingSystems->reset();
    }
}

void VRTracker::enableTracking(bool on)
{
    // reset tracking to pos 0 0 0
    if (on && !enableTracking_)
    {
        // get actual pos of hand
        osg::Vec3 pos = trackerStationMat[trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev)]->getTrans();
        // get the offset
        float *offTrans = VRTracker::instance()->trackingUtil->getDeviceOffsetTrans((coVRTrackingUtil::IDOfDevice)0);
        // calculate new offset
        offTrans[0] = -pos[0] + offTrans[0];
        offTrans[2] = -pos[2] + offTrans[2];
        osg::Vec3 trans, rot;
        trans = osg::Vec3(offTrans[0], offTrans[1], offTrans[2]);
        rot = osg::Vec3(0, 0, 0);
        // set new offset
        VRTracker::instance()->trackingUtil->setDeviceOffset((coVRTrackingUtil::IDOfDevice)0, trans, rot);
    }
    enableTracking_ = on;
    if (coVRTrackingUtil::instance()->hasHand())
    {
        if (!on)
            VRSceneGraph::instance()->getScene()->removeChild(VRSceneGraph::instance()->getHandTransform());
        else
            VRSceneGraph::instance()->getScene()->addChild(VRSceneGraph::instance()->getHandTransform());
    }
}

int
VRTracker::readConfigFile()
{
    std::string line;

    doJoystick = coCoviseConfig::isOn("COVER.Input.Joystick", false);
    doCalibrateOrientation = coCoviseConfig::isOn("COVER.Input.CalibrateOrientation", false);

    // get the file name for the trilinear interpolation
    // for tracker correction
    line = coCoviseConfig::getEntry("COVER.Input.InterpolationFile");
    if (!line.empty())
    {
        interpolationFile = coVRFileManager::instance()->getName(line.c_str());
        if (interpolationFile)
            doCalibrate = true;
    }
    else
    {
        doCalibrate = false;
    }

    // tracker configuration

    // menu selection via joystick
    doJoystickVisenso = coCoviseConfig::isOn("COVER.Input.VisensoJoystick", false);
    visensoJoystickSpeed = coCoviseConfig::getFloat("COVER.Input.VisensoJoystick.Speed", 10.0f);
#ifdef OLDINPUT
    renderInterface->getJoystickManager()->setActive(doJoystickVisenso);
#endif
    visensoJoystickAnalog = coCoviseConfig::isOn("COVER.Input.VisensoJoystick.Analog", false);

    buttonStation = coCoviseConfig::getInt("COVER.Input.ButtonAddress", -1);
    if (buttonStation == -1)
    {
        buttonStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev);
    }
    if (buttonStation == -1)
    {
        cerr << "failed to read COVER.Input.ButtonAddress from config file in VRTracker::readConfigFile" << endl;
    }
    if (doJoystick || visensoJoystickAnalog)
    {
        analogStation = coCoviseConfig::getInt("COVER.Input.AnalogAddress", -1);
        if (analogStation == -1) // integrated joystick
        {
            analogStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev);
        }
    }

    line = coCoviseConfig::getEntry("COVER.Input.DebugTracking");
    if (line.compare("APP") == 0)
    {
        d_debugTracking = true;
    }
    d_debugStation = coCoviseConfig::getInt("COVER.Input.DebugStation", 0);

    const coCoviseConfig::ScopeEntries mapEntries = coCoviseConfig::getScopeEntries("COVER.Input.ButtonConfig", "Map");
    const char **maps = mapEntries.getValue();

    if (maps)
    {
        int i = 0;
        while (maps[i])
        {
            i++;
            if (maps[i])
            {
                char buf[256];
                unsigned int bitmask = 1;
                if (sscanf(maps[i], "%u %s", &bitmask, buf) != 2)
                {
                    cerr << "VRTracker:: error reading button mapping: sould be \"bitmask BUTTON_NAME\" example \"4 XFORM_BUTTON\"" << endl;
                }
                if (strcasecmp(buf, "ACTION_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::ACTION_BUTTON));
                else if (strcasecmp(buf, "DRIVE_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::DRIVE_BUTTON));
                else if (strcasecmp(buf, "XFORM_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::XFORM_BUTTON));
                else if (strcasecmp(buf, "USER1_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::USER1_BUTTON));
                else if (strcasecmp(buf, "USER4_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::USER4_BUTTON));
#if 0
            else if(strcasecmp(buf,"USER2_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER2_BUTTON));
            else if(strcasecmp(buf,"USER3_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER3_BUTTON));
            else if(strcasecmp(buf,"USER5_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER5_BUTTON));
            else if(strcasecmp(buf,"USER6_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER6_BUTTON));
            else if(strcasecmp(buf,"USER7_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER7_BUTTON));
            else if(strcasecmp(buf,"USER8_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER8_BUTTON));
            else if(strcasecmp(buf,"USER9_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER9_BUTTON));
            else if(strcasecmp(buf,"USER10_BUTTON")==0)
               buttonMap.insert(Data_Pair(bitmask,vruiButtons::USER10_BUTTON));
#endif
                else if (strcasecmp(buf, "DRAG_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::DRAG_BUTTON));
                else if (strcasecmp(buf, "WHEEL_UP_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::WHEEL_UP));
                else if (strcasecmp(buf, "WHEEL_DOWN_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::WHEEL_DOWN));
                else if (strcasecmp(buf, "JOYSTICK_DOWN") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::JOYSTICK_DOWN));
                else if (strcasecmp(buf, "JOYSTICK_LEFT") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::JOYSTICK_LEFT));
                else if (strcasecmp(buf, "JOYSTICK_UP") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::JOYSTICK_UP));
                else if (strcasecmp(buf, "JOYSTICK_RIGHT") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::JOYSTICK_RIGHT));
                else if (strcasecmp(buf, "FORWARD_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::FORWARD_BUTTON));
                else if (strcasecmp(buf, "BACKWARD_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::BACKWARD_BUTTON));
                else if (strcasecmp(buf, "MENU_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::MENU_BUTTON));
                else if (strcasecmp(buf, "INTER_NEXT") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::INTER_NEXT));
                else if (strcasecmp(buf, "INTER_PREV") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::INTER_PREV));
                else if (strcasecmp(buf, "TOGGLE_DOCUMENTS") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::TOGGLE_DOCUMENTS));
                else if (strcasecmp(buf, "QUIT_BUTTON") == 0)
                    buttonMap.insert(Data_Pair(bitmask, vruiButtons::QUIT_BUTTON));
                else
                {
                    cerr << "unknown button name in ButtonConfig.MAP" << maps[i] << endl;
                }
                i++;
            }
        }
    }

    return (0);
}

void
VRTracker::updateHead()
{
#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif
    if (trackingUtil->hasHead())
    {

        int headSensorStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev);
        // done in OpenCOVER.cpp VRViewer::instance()->updateViewerMat(*trackerStationMat[headSensorStation]);
        // don�t do it here, otherwise we will have different view matrices on master and slave renderers
        if (d_saveTracking)
        {
            fprintf(d_saveTracking, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    (*trackerStationMat[headSensorStation])(0, 0), (*trackerStationMat[headSensorStation])(0, 1), (*trackerStationMat[headSensorStation])(0, 2),
                    (*trackerStationMat[headSensorStation])(1, 0), (*trackerStationMat[headSensorStation])(1, 1), (*trackerStationMat[headSensorStation])(1, 2),
                    (*trackerStationMat[headSensorStation])(2, 0), (*trackerStationMat[headSensorStation])(2, 1), (*trackerStationMat[headSensorStation])(2, 2),
                    (*trackerStationMat[headSensorStation])(3, 0), (*trackerStationMat[headSensorStation])(3, 1), (*trackerStationMat[headSensorStation])(3, 2));
        }
    }
}

static int mapButton(int raw, const std::map<int, int> &buttonMap)
{
    int mapped = 0;
    for (int bit = 1; bit; bit <<= 1)
    {
        if ((raw & bit) == 0)
            continue;

        if (buttonMap.find(bit) == buttonMap.end())
            mapped |= bit;
        else
            mapped |= const_cast<std::map<int, int> &>(buttonMap)[bit];
    }
    return mapped;
}

void
VRTracker::updateHand()
{
    unsigned int button = 0, rawButton = 0, mouseButton = 0;
    int handSensorStation = trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev);

#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif

    if (trackingSystems && VRTracker::instance()->trackingUtil->haveDevice(coVRTrackingUtil::handDev))
    {

#ifdef OLDINPUT
        trackingSystems->getButton(buttonStation, &rawButton);
        button = mapButton(rawButton, buttonMap);
        if (button == QUIT_BUTTON)
        {
            OpenCOVER::instance()->quitCallback(NULL, NULL);
        }
        if (doJoystick)
        {
            trackingSystems->getAnalog(handSensorStation, VRSceneGraph::instance()->joyStickX(), VRSceneGraph::instance()->joyStickY());
        }
        cover->getPointerButton()->setState(button);
#endif
        setButtonState(button);
    }
    else if ((trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_VRC_DAEMON) && VRTracker::instance()->trackingUtil->haveDevice(coVRTrackingUtil::handDev))
    {

        unsigned int mask = vrcTracker->getButtons(buttonStation);
        if (mask)
        {
            for (int i = 0; i < 32; ++i)
            {
                if (mask & (1 << i))
                {
                    if (buttonMap[i + 1] == vruiButtons::QUIT_BUTTON)
                    {
                        OpenCOVER::instance()->quitCallback(NULL, NULL);
                    }
                }
            }
        }

        rawButton = vrcTracker->getButton(buttonStation);
        button = mapButton(rawButton, buttonMap);
        if (doJoystick)
        {
            vrcTracker->getAnalog(analogStation, coVRNavigationManager::instance()->AnalogX, coVRNavigationManager::instance()->AnalogY);
        }

#ifdef OLDINPUT
        if (doJoystickVisenso && !visensoJoystickAnalog)
        {
            if (button == vruiButtons::JOYSTICK_DOWN)
            {
                coVRNavigationManager::instance()->AnalogX = 0.0f;
                coVRNavigationManager::instance()->AnalogY = -visensoJoystickSpeed;
                if (coVRConfig::instance()->isMenuModeOn())
                    renderInterface->getJoystickManager()->newUpdate(0, -1, 0);
                else
                    renderInterface->getJoystickManager()->update(0, -1);
            }
            else if (button == vruiButtons::JOYSTICK_LEFT)
            {
                coVRNavigationManager::instance()->AnalogX = -visensoJoystickSpeed;
                coVRNavigationManager::instance()->AnalogY = 0.0f;
                if (coVRConfig::instance()->isMenuModeOn())
                    renderInterface->getJoystickManager()->newUpdate(-1, 0, 0);
                else
                    renderInterface->getJoystickManager()->update(-1, 0);
            }
            else if (button == vruiButtons::JOYSTICK_UP)
            {
                coVRNavigationManager::instance()->AnalogX = 0.0f;
                coVRNavigationManager::instance()->AnalogY = visensoJoystickSpeed;
                if (coVRConfig::instance()->isMenuModeOn())
                    renderInterface->getJoystickManager()->newUpdate(0, 1, 0);
                else
                    renderInterface->getJoystickManager()->update(0, 1);
            }
            else if (button == vruiButtons::JOYSTICK_RIGHT)
            {
                coVRNavigationManager::instance()->AnalogX = visensoJoystickSpeed;
                coVRNavigationManager::instance()->AnalogY = 0.0f;
                if (coVRConfig::instance()->isMenuModeOn())
                    renderInterface->getJoystickManager()->newUpdate(1, 0, 0);
                else
                    renderInterface->getJoystickManager()->update(1, 0);
            }
            else
            {
                coVRNavigationManager::instance()->AnalogX = 0.0f;
                coVRNavigationManager::instance()->AnalogY = 0.0f;
                if (coVRConfig::instance()->isMenuModeOn())
                    renderInterface->getJoystickManager()->newUpdate(0, 0, button);
                else
                    renderInterface->getJoystickManager()->update(0, 0);
            }
#ifdef OLDINPUT
            cover->getPointerButton()->setState(button);
#endif
            setButtonState(button);
        }
        else if (doJoystickVisenso)
        {
            float x, y;
            vrcTracker->getAnalog(buttonStation, x, y);
            if (x < -0.5)
                button = vruiButtons::JOYSTICK_LEFT;
            else if (x > 0.5)
                button = vruiButtons::JOYSTICK_RIGHT;
            else if (y < -0.5)
                button = vruiButtons::JOYSTICK_DOWN;
            else if (y > 0.5)
                button = vruiButtons::JOYSTICK_UP;
#ifdef OLDINPUT
            cover->getPointerButton()->setState(button);
#endif
            setButtonState(button);
            coVRNavigationManager::instance()->AnalogX = x * visensoJoystickSpeed;
            coVRNavigationManager::instance()->AnalogY = y * visensoJoystickSpeed;
            if (coVRConfig::instance()->isMenuModeOn())
            {
                renderInterface->getJoystickManager()->newUpdate(x, y, button);
            }
            else
            {
                renderInterface->getJoystickManager()->update(x, y);
            }
        }
        else
        {
#ifdef OLDINPUT
            cover->getPointerButton()->setState(button);
#endif
            setButtonState(button);
        }
#endif
    }

    else if ((trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SPACEPOINTER)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_COVER_BEEBOX)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_COVER_FLYBOX)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_PHANTOM)
             || (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SPACEBALL))
    {
        button = mapButton(mouseButton, buttonMap);
        spacePointer->update(*trackerStationMat[handSensorStation], &button);
        calibrate(*trackerStationMat[handSensorStation]);
#ifdef OLDINPUT
        cover->getPointerButton()->setState(button);
#endif
        setButtonState(button);
    }

    // make sure that cover->getMouseButton()->setState() is called whenever the mouse is not
    // the primary button device
    if (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_MOUSE || trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SEEREAL)
    {
        *trackerStationMat[handSensorStation] = mouseMat;
#ifdef OLDINPUT
        cover->getPointerButton()->setState(mouseButton);
#endif
        setButtonState(button);
    }
    else
    {
#ifdef OLDINPUT
        cover->getMouseButton()->setState(mouseButton);
#endif
        setButtonState(button);
    }

    if (d_saveTracking)
    {
        fprintf(d_saveTracking, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n",
                (*trackerStationMat[handSensorStation])(0, 0), (*trackerStationMat[handSensorStation])(0, 1), (*trackerStationMat[handSensorStation])(0, 2),
                (*trackerStationMat[handSensorStation])(1, 0), (*trackerStationMat[handSensorStation])(1, 1), (*trackerStationMat[handSensorStation])(1, 2),
                (*trackerStationMat[handSensorStation])(2, 0), (*trackerStationMat[handSensorStation])(2, 1), (*trackerStationMat[handSensorStation])(2, 2),
                (*trackerStationMat[handSensorStation])(3, 0), (*trackerStationMat[handSensorStation])(3, 1), (*trackerStationMat[handSensorStation])(3, 2), button);
    }
}

void
VRTracker::updateWheel()
{
    if (trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_TARSUS
        || trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_SSD
        || trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_VRPN
        || trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_DYNASIGHT
        || trackingUtil->getTrackingSystem() == coVRTrackingUtil::T_DTRACK)
    {
        int wheel = 0;
        trackingSystems->getWheel(trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev), &wheel);
#ifdef OLDINPUT
        cover->getPointerButton()->setWheel(wheel);
#endif
    }
}

void
VRTracker::updateDevice(int device)
{

#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif
    if (trackingUtil->haveDevice((coVRTrackingUtil::IDOfDevice)device))
    {
        int station = trackingUtil->getDeviceAddress((coVRTrackingUtil::IDOfDevice)device);
        if (trackingSystems)
        {
            trackingSystems->getMatrix(station, *trackerStationMat[station]);

            trackerStationMat[station]->preMult(VRTracker::instance()->trackingUtil->computeDeviceOffsetMat((coVRTrackingUtil::IDOfDevice)device));
            trackerStationMat[station]->postMult(VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::trackingSys));
        }
        else if (vrcTracker)
        {

            vrcTracker->getMatrix(station, *trackerStationMat[station]);
            osg::Vec3 pos; // vrc tracker provides cm, we need mm
            pos = trackerStationMat[station]->getTrans();
            pos *= vrcTracker->unit;
            trackerStationMat[station]->setTrans(pos);
            trackerStationMat[station]->preMult(VRTracker::instance()->trackingUtil->computeDeviceOffsetMat((coVRTrackingUtil::IDOfDevice)device));
            trackerStationMat[station]->postMult(VRTracker::instance()->trackingUtil->computeDeviceOffsetMat(coVRTrackingUtil::trackingSys));
        }
        else
        {
        }
        calibrate(*trackerStationMat[station]);

        if (d_debugTracking && station == d_debugStation)
        {
            const char *note = "";
            if (station == getHandSensorStation())
                note = " (hand)";
            else if (station == getHeadSensorStation())
                note = " (head)";

            fprintf(stderr, "Station %d%s APP [mm]: [%7.1f %7.1f %7.1f]\n",
                    station, note,
                    (*trackerStationMat[station])(3, 0),
                    (*trackerStationMat[station])(3, 1),
                    (*trackerStationMat[station])(3, 2));
        }
    }
}

/*______________________________________________________________________*/
void
VRTracker::initCereal(angleStruct *screen_angle)
{
#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif
    if (trackingSystems)
    {
        //cerr << "calling getCerealAnalog\n";
        if (screen_angle)
            trackingSystems->getCerealAnalog(screen_angle->screen, &screen_angle->value);
        else
        {
#ifdef DEBUG
            cerr << "Not calling trackingSystems->getCerealAnalog as"
                 << " no CerealConfig.SCREEN_ANGLE entry exists in covise.config"
                 << endl;
#endif
        }
    }
    else
    {
        //cerr << "NOT calling getCerealAnalog\n";
    }
}

osg::Matrix
VRTracker::getDeviceMat(coVRTrackingUtil::IDOfDevice device)
{
    osg::Matrix m = getStationMat(trackingUtil->getDeviceAddress(device));
    m.preMult(trackingUtil->computeDeviceOffsetMat(device));
    m.postMult(trackingUtil->computeDeviceOffsetMat(device));
    return m;
}

osg::Matrix &
VRTracker::getViewerMat()
{
    return getStationMat(trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev));
}

osg::Matrix &
VRTracker::getHandMat()
{
    return getStationMat(trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev));
}

osg::Matrix &
VRTracker::getMouseMat()
{
    return mouseMat;
}

osg::Matrix &
VRTracker::getWorldMat()
{
    return getStationMat(trackingUtil->getDeviceAddress(coVRTrackingUtil::worldDev));
}

osg::Matrix &
VRTracker::getCameraMat()
{
    return getStationMat(trackingUtil->getDeviceAddress(coVRTrackingUtil::cameraDev));
}

osg::Matrix &
VRTracker::getStationMat(int station)
{
    if ((station > trackingUtil->getNumStations()) | (station < 0))
    {
        if (cover->debugLevel(4))
            cerr << "Station " << station << " does not exist." << endl;
        return myIdentity;
    }
    else
        return *trackerStationMat[station];
}

void
VRTracker::setHandMat(const osg::Matrix &m)
{
    *trackerStationMat[trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev)] = m;
}

void
VRTracker::setViewerMat(const osg::Matrix &m)
{
    *trackerStationMat[trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev)] = m;
}

void VRTracker::setStationMat(const osg::Matrix &m, int station)
{
    if ((station > trackingUtil->getNumStations()) | (station < 0))
    {
        if (cover->debugLevel(4))
            cerr << "Station " << station << " does not exist." << endl;
    }
    else
    {
        *trackerStationMat[station] = m;
    }
}

void
VRTracker::setWorldMat(const osg::Matrix &m)
{
    if (trackingUtil->haveDevice(coVRTrackingUtil::worldDev))
    {
        *trackerStationMat[trackingUtil->getDeviceAddress(coVRTrackingUtil::worldDev)] = m;
    }
}

void
VRTracker::setMouseMat(const osg::Matrix &m)
{
    mouseMat = m;
}

int
VRTracker::getNumStation()
{
    return trackingUtil->getNumStations();
}

unsigned int
VRTracker::getButtonState() const
{
#ifdef OLDINPUT
    return (cover->getPointerButton()->getState());
#else
    return m_buttonState;
#endif
}

void
VRTracker::updateHandPublic()
{
#ifdef OLDINPUT
    if (coVRMSController::instance()->isSlave())
    {
        return;
    }
#endif
    updateHand();
}

int
VRTracker::getHandSensorStation()
{
    return (trackingUtil->getDeviceAddress(coVRTrackingUtil::handDev));
}

int
VRTracker::getHeadSensorStation()
{
    return (trackingUtil->getDeviceAddress(coVRTrackingUtil::headDev));
}

int
VRTracker::getCameraSensorStation()
{
    return (trackingUtil->getDeviceAddress(coVRTrackingUtil::cameraDev));
}

int
VRTracker::getWorldSensorStation()
{
    return (trackingUtil->getDeviceAddress(coVRTrackingUtil::worldDev));
}

coVRTrackingSystems *
VRTracker::getTrackingSystemsImpl()
{
    return (trackingSystems);
}

// tracker calibration

void VRTracker::calibrate(osg::Matrix &mat)
{
    if (doCalibrate)
    {
        float matrix_in[4][4];
        float matrix_tr[4][4];
        int vector, elem;

        for (vector = 0; vector < 4; vector++)
        {
            for (elem = 0; elem < 4; elem++)
            {
                matrix_in[vector][elem] = mat(vector, elem);
            }
        }

        interpolate(matrix_in, matrix_tr);

        for (vector = 0; vector < 4; vector++)
        {
            for (elem = 0; elem < 4; elem++)
            {
                mat(vector, elem) = matrix_tr[vector][elem];
            }
        }
    }
}

bool
VRTracker::readInterpolationFile(const char *datapath)
{

    int i, j, k, line = 0, index, vector;
    char buf[1000];
    float minP[3], maxP[3];
    bool hasDim = false;
    bool hasMin = false;
    bool hasMax = false;
    FILE *fp = fopen(datapath, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "File not found: %s\n", datapath);
        doCalibrate = false;
        return false;
    }
    // read header
    while ((!feof(fp)) && (hasDim == false || hasMin == false || hasMax == false))
    {
        if (fgets(buf, 1000, fp) == NULL)
        {
            cerr << "VRTracker::readInterpolationFile fgets failed " << endl;
            break;
        }
        line++;
        if ((buf[0] != '%') && (strlen(buf) > 5))
        {
            if (strncasecmp(buf, "DIM", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%d %d %d", &nx, &ny, &nz);
                if (iret != 3)
                    cerr << "VRTracker::readInterpolationFile sscanf failed: read " << iret << endl;
                hasDim = true;
            }
            else if (strncasecmp(buf, "MIN", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%f %f %f", &minP[0], &minP[1], &minP[2]);
                if (iret != 3)
                    cerr << "VRTracker::readInterpolationFile sscanf2 failed: read " << iret << endl;
                hasMin = true;
            }
            else if (strncasecmp(buf, "MAX", 3) == 0)
            {
                int iret = sscanf(buf + 3, "%f %f %f", &maxP[0], &maxP[1], &maxP[2]);
                if (iret != 3)
                    cerr << "VRTracker::readInterpolationFile sscanf3 failed: read " << iret << endl;
                hasMax = true;
            }
            else
            {
                cerr << "Unknown statement in line " << line << endl;
            }
        }
    }

    if (nx < 3 || ny < 3 || nz < 3)
    {
        cout << "Interpolation data file is too small,\n"
             << "it should be at least a 3*3*3 grid" << endl;
        return false;
    }

    //buid the arrays
    x_coord = new float[nx * ny * nz];
    y_coord = new float[nx * ny * nz];
    z_coord = new float[nx * ny * nz];

    n1 = new float[nx * ny * nz];
    n2 = new float[nx * ny * nz];
    n3 = new float[nx * ny * nz];

    trans_basis = new float **[nx * ny * nz];
    for (index = 0; index < nx * ny * nz; index++)
    {
        trans_basis[index] = new float *[3];
        for (vector = 0; vector < 3; vector++)
        {
            trans_basis[index][vector] = new float[3];
        }
    }

    float px, py, pz;
    float orientation[9];

    float unit_mat[3][3] = {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };

    while (!feof(fp))
    {
        if (fgets(buf, 1000, fp) == NULL)
            cerr << "VRTracker::readInterpolationFile fgets2 failed " << endl;
        line++;
        if ((buf[0] != '%') && (strlen(buf) > 5))
        {
            int ntok = sscanf(buf, "%d %d %d", &i, &j, &k);
            if (ntok == 3) // all three numbers parsed
            {
                int nums = sscanf(buf, "%d %d %d    %f %f %f    %f %f %f  %f %f %f  %f %f %f",
                                  &i, &j, &k,
                                  &px, &py, &pz,
                                  &orientation[0], &orientation[1], &orientation[2],
                                  &orientation[3], &orientation[4], &orientation[5],
                                  &orientation[6], &orientation[7], &orientation[8]);
                if (nums != 15)
                {
                    fprintf(stderr, "error parsing calib file\n");
                }
                int index = i * ny * nz + ((ny - 1) - j) * nz + k;
                x_coord[index] = px;
                y_coord[index] = py;
                z_coord[index] = pz;
                n1[index] = (minP[0] + i * (maxP[0] - minP[0]) / (nx - 1));
                n2[index] = (minP[1] + ((ny - 1) - j) * (maxP[1] - minP[1]) / (ny - 1));
                n3[index] = (minP[2] + k * (maxP[2] - minP[2]) / (nz - 1));

                linear_equations_sys(&orientation[0], &orientation[3], &orientation[6],
                                     unit_mat[0], trans_basis[index][0]);
                linear_equations_sys(&orientation[0], &orientation[3], &orientation[6],
                                     unit_mat[1], trans_basis[index][1]);
                linear_equations_sys(&orientation[0], &orientation[3], &orientation[6],
                                     unit_mat[2], trans_basis[index][2]);
            }
        }
    }
    fclose(fp);
    return true;
}

//void VRTracker::interpolate(        const float tracker_p[3] , float p_interp[3] )
void
VRTracker::interpolate(const float mat_in[4][4], float mat_tr[4][4])

{
    //        tracker_p :analized point                                                                                          /-----/-----q
    //  cl_ :closest grid point to tracker_p                                                                                        /     /  tp /
    //  q_  :q_ and cl are on the oposite corners of a 3d cuadrant (sector)                              /----cl-----/
    //        int cl_i=0 , cl_j=0 ,cl_k=0;        //,cl_index=0;

    float m_bas[3][3]; // interpolated middle tranformation basis for the exact pos. of tracker_p
    float mat_no[3][3]; // not orthogonal transformated matrix.
    float mat_o_nu[3][3]; // orthogonal but not unitary matrix
    float v_len[3];
    float comp_2in1;
    // diff_bas[direction][vector][comp]
    float tracker_p[3];
    int cl[3]; //  cl_ :closest grid point to tracker_p ( the 3 indexes )                                                                                                /     /  dp /
    int di, dj, dk;
    //float da_001,da_010,da_100;         //distances from 3 planes that have cl_ to tracker_p
    //float db_001,db_010,db_100;                //distances from the planes that are in front of the da_ to tracker_p
    float da[3]; //distances from 3 planes that have cl_ to tracker_p as ve
    float db[3]; //distances from the planes that are in front of the da_ to tracker_p
    float dr[3]; //relativ position of tracker_p in the cell
    bool quad_found;

    //8 points of a cell
    int index_000, index_001, index_010, index_011, index_100, index_101, index_110, index_111;
    float point_000[3];
    float point_001[3];
    float point_010[3];
    float point_011[3];
    float point_100[3];
    float point_101[3];
    float point_110[3];
    //float point_111[3];
    int vector, elem;

    tracker_p[0] = mat_in[3][0];
    tracker_p[1] = mat_in[3][1];
    tracker_p[2] = mat_in[3][2];

    //___________________________________________________________
    //find the closest point

    find_closest_g_point(tracker_p, cl);

    //__________________________________________________________________________________
    //        determinate in which of the 8 quadrants (sectors )
    //        around the closest grid point
    //        is the analized point
    //        the grid point has three neighbors in every cuadrant neig_i,neig_j,neig_k

    quad_found = false;
    for (di = -1; di <= 1; di = di + 2)
    {
        for (dj = -1; dj <= 1; dj = dj + 2)
        {
            for (dk = -1; dk <= 1; dk = dk + 2)
            {
                index_000 = (cl[0]) * ny * nz + (cl[1]) * nz + (cl[2]);
                index_001 = (cl[0]) * ny * nz + (cl[1]) * nz + (cl[2] + dk);
                index_010 = (cl[0]) * ny * nz + (cl[1] + dj) * nz + (cl[2]);
                index_011 = (cl[0]) * ny * nz + (cl[1] + dj) * nz + (cl[2] + dk);
                index_100 = (cl[0] + di) * ny * nz + (cl[1]) * nz + (cl[2]);
                index_101 = (cl[0] + di) * ny * nz + (cl[1]) * nz + (cl[2] + dk);
                index_110 = (cl[0] + di) * ny * nz + (cl[1] + dj) * nz + (cl[2]);
                index_111 = (cl[0] + di) * ny * nz + (cl[1] + dj) * nz + (cl[2] + dk);

                point_000[0] = x_coord[index_000];
                point_001[0] = x_coord[index_001];
                point_010[0] = x_coord[index_010];
                point_011[0] = x_coord[index_011];
                point_100[0] = x_coord[index_100];
                point_101[0] = x_coord[index_101];
                point_110[0] = x_coord[index_110];
                //        point_111[0] = x_coord[index_111] ;

                point_000[1] = y_coord[index_000];
                point_001[1] = y_coord[index_001];
                point_010[1] = y_coord[index_010];
                point_011[1] = y_coord[index_011];
                point_100[1] = y_coord[index_100];
                point_101[1] = y_coord[index_101];
                point_110[1] = y_coord[index_110];
                //        point_111[1] = y_coord[index_111] ;

                point_000[2] = z_coord[index_000];
                point_001[2] = z_coord[index_001];
                point_010[2] = z_coord[index_010];
                point_011[2] = z_coord[index_011];
                point_100[2] = z_coord[index_100];
                point_101[2] = z_coord[index_101];
                point_110[2] = z_coord[index_110];
                //        point_111[2] = z_coord[index_111] ;

                /*                                da_001= dis_pn_pt (         point_000, point_100, point_010, tracker_p   );
                                              da_010= dis_pn_pt (         point_000, point_001, point_100, tracker_p   );
                                              da_100= dis_pn_pt (         point_000, point_010, point_001, tracker_p   );

                                              db_001= dis_pn_pt (         point_001, point_101, point_011, tracker_p   );
                                              db_010= dis_pn_pt (         point_010, point_011, point_110, tracker_p   );
                                              db_100= dis_pn_pt (         point_100, point_110, point_101, tracker_p   );

                                              if (   (da_001*db_001<=0) && (da_010*db_010<=0) && (da_100*db_100<=0)   )
                                              {
            //the point tracker_p is in this cell
            quad_found = true;
            break;
            }
             */
                da[2] = dis_pn_pt(point_000, point_100, point_010, tracker_p);
                da[1] = dis_pn_pt(point_000, point_001, point_100, tracker_p);
                da[0] = dis_pn_pt(point_000, point_010, point_001, tracker_p);

                db[2] = dis_pn_pt(point_001, point_101, point_011, tracker_p);
                db[1] = dis_pn_pt(point_010, point_011, point_110, tracker_p);
                db[0] = dis_pn_pt(point_100, point_110, point_101, tracker_p);

                if ((da[0] * db[0] <= 0) && (da[1] * db[1] <= 0) && (da[2] * db[2] <= 0))
                {
                    //the point tracker_p is in this cell
                    quad_found = true;
                    break;
                }
            }
            if (quad_found == true)
            {
                break;
            }
        }
        if (quad_found == true)
        {
            break;
        }
    }

    dr[0] = da[0] / (da[0] - db[0]); //reativ pos of P_tracker in the cell
    dr[1] = da[1] / (da[1] - db[1]);
    dr[2] = da[2] / (da[2] - db[2]);

    //only copy
    for (vector = 0; vector < 4; vector++)
    {
        for (elem = 0; elem < 4; elem++)
        {
            mat_tr[vector][elem] = mat_in[vector][elem];
        }
    }

    //changes
    mat_tr[3][0] = n1[index_000] + (n1[index_100] - n1[index_000]) * dr[0];
    mat_tr[3][1] = n2[index_000] + (n2[index_010] - n2[index_000]) * dr[1];
    mat_tr[3][2] = n3[index_000] + (n3[index_001] - n3[index_000]) * dr[2];

    if (doCalibrateOrientation)
    {
        // find a intermediary basis for the tracker pos
        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {
                val8[0][0][0] = trans_basis[index_000][vector][elem];
                val8[0][0][1] = trans_basis[index_001][vector][elem];
                val8[0][1][0] = trans_basis[index_010][vector][elem];
                val8[0][1][1] = trans_basis[index_011][vector][elem];
                val8[1][0][0] = trans_basis[index_100][vector][elem];
                val8[1][0][1] = trans_basis[index_101][vector][elem];
                val8[1][1][0] = trans_basis[index_110][vector][elem];
                val8[1][1][1] = trans_basis[index_111][vector][elem];

                val4[0][0] = val8[0][0][0] + (val8[0][0][1] - val8[0][0][0]) * dr[2];
                val4[0][1] = val8[0][1][0] + (val8[0][1][1] - val8[0][1][0]) * dr[2];
                val4[1][0] = val8[1][0][0] + (val8[1][0][1] - val8[1][0][0]) * dr[2];
                val4[1][1] = val8[1][1][0] + (val8[1][1][1] - val8[1][1][0]) * dr[2];

                val2[0] = val4[0][0] + (val4[0][1] - val4[0][0]) * dr[1];
                val2[1] = val4[1][0] + (val4[1][1] - val4[1][0]) * dr[1];

                val1 = val2[0] + (val2[1] - val2[0]) * dr[0];
                m_bas[vector][elem] = val1;
            }
        }

        //matrix multiplication
        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {

                mat_no[vector][elem] = (m_bas[0][elem] * mat_in[vector][0] + m_bas[1][elem] * mat_in[vector][1] + m_bas[2][elem] * mat_in[vector][2]);
            }
        }

        //make the vectors orthogonal
        // vector[1] derection will be keeped
        for (elem = 0; elem < 3; elem++)
        {
            mat_o_nu[1][elem] = mat_no[1][elem];
        }
        v_len[1] = sqrt(mat_no[1][0] * mat_no[1][0] + mat_no[1][1] * mat_no[1][1] + mat_no[1][2] * mat_no[1][2]);

        // vector[2]
        comp_2in1 = (mat_no[1][0] * mat_no[2][0] + mat_no[1][1] * mat_no[2][1] + mat_no[1][2] * mat_no[2][2]) / v_len[1];

        mat_o_nu[2][0] = mat_no[2][0] - comp_2in1 * mat_no[1][0] / v_len[1];
        mat_o_nu[2][1] = mat_no[2][1] - comp_2in1 * mat_no[1][1] / v_len[1];
        mat_o_nu[2][2] = mat_no[2][2] - comp_2in1 * mat_no[1][2] / v_len[1];
        v_len[2] = sqrt(mat_o_nu[2][0] * mat_o_nu[2][0] + mat_o_nu[2][1] * mat_o_nu[2][1] + mat_o_nu[2][2] * mat_o_nu[2][2]);

        mat_o_nu[0][0] = +mat_no[1][1] * mat_no[2][2] - mat_no[1][2] * mat_no[2][1];
        mat_o_nu[0][1] = -mat_no[1][0] * mat_no[2][2] + mat_no[1][2] * mat_no[2][0];
        mat_o_nu[0][2] = +mat_no[1][0] * mat_no[2][1] - mat_no[1][1] * mat_no[2][0];

        v_len[0] = sqrt(mat_o_nu[0][0] * mat_o_nu[0][0] + mat_o_nu[0][1] * mat_o_nu[0][1] + mat_o_nu[0][2] * mat_o_nu[0][2]);

        for (vector = 0; vector < 3; vector++)
        {
            for (elem = 0; elem < 3; elem++)
            {
                mat_tr[vector][elem] = mat_o_nu[vector][elem] / v_len[vector];
            }
        }
    }
}

//_____________________________________________________________________
void
VRTracker::find_closest_g_point(const float tracker_p[3], int cl[3])
/*
   void
   VRTracker::find_closest_g_point (float dp_x, float dp_y,        float dp_z,
   int *cl_i,         int *cl_j,        int *cl_k        )
*/
{

    // find the closest grid point
    // the points on the borders are excluded
    int closest_i = 0, closest_j = 0, closest_k = 0;
    bool first;
    int iii, jjj, kkk, index;
    float vgp_x, vgp_y, vgp_z; // vector from a grid point to the tracker point
    float dis_min = 0, distance = 0;

    first = true;
    for (iii = 1; iii <= (nx - 2); iii++)
    {
        for (jjj = 1; jjj <= (ny - 2); jjj++)
        {
            for (kkk = 1; kkk <= (nz - 2); kkk++)
            {
                index = (iii * ny * nz + jjj * nz + kkk);
                /*                                cout << "iii =" << iii << "\tjjj =" << jjj << "\tkkk =" << kkk << "\tindex=" << index << endl;
             */
                vgp_x = tracker_p[0] - x_coord[index];
                vgp_y = tracker_p[1] - y_coord[index];
                vgp_z = tracker_p[2] - z_coord[index];
                distance = sqrt(vgp_x * vgp_x + vgp_y * vgp_y + vgp_z * vgp_z);

                if ((first == true) || (distance < dis_min))
                {
                    dis_min = distance;
                    closest_i = iii;
                    closest_j = jjj;
                    closest_k = kkk;
                }
                first = false;
            }
        }
    }

    cl[0] = closest_i;
    cl[1] = closest_j;
    cl[2] = closest_k;
}

//___________________________________________________________________________
float
VRTracker::dis_pn_pt(const float a[3],
                     const float b[3],
                     const float c[3],
                     const float p[3])
/*
   float
   VRTracker::dis_pn_pt (         float ax , float ay , float az ,
   float bx , float by , float bz ,
   float cx , float cy , float cz ,
   float px , float py , float pz    )
 */
{
    // Calculate the distance from a plane (a_ b_ c_) to a point (p_)
    float abx, aby, abz;
    float acx, acy, acz;
    float apx, apy, apz;
    float tx, ty, tz;
    float area, volumen, dist;

    abx = b[0] - a[0];
    aby = b[1] - a[1];
    abz = b[2] - a[2];

    acx = c[0] - a[0];
    acy = c[1] - a[1];
    acz = c[2] - a[2];

    apx = p[0] - a[0];
    apy = p[1] - a[1];
    apz = p[2] - a[2];

    tx = aby * acz - abz * acy;
    ty = -abx * acz + abz * acx;
    tz = abx * acy - aby * acx;

    area = sqrt(tx * tx + ty * ty + tz * tz);
    volumen = (abx * acy * apz + aby * acz * apx + abz * acx * apy - apx * acy * abz - apy * acz * abx - apz * acx * aby);

    if (area != 0)
    {
        dist = volumen / area;
    }
    else
    {
        dist = 0;
        cout << "division by 0 in funktion dis_pn_pt"
             << "three poins of the plane are in the same line " << endl;
    }
    return dist;
}

//_________________________________________________________________________
int
VRTracker::find_xyz_velocity(void)
//find in which order are changing x,y,z of the real values of the
// interpolation file,for example:
// find_xyz_velocity = 123 means that z chance faster, than y and than x
// find_xyz_velocity = 213 means that z chance faster, than x and than y (cave)
//        in case of 213 the data will be reorganized to 123 (standard)
{
    int n1_velocity = 0, n2_velocity = 0, n3_velocity = 0;

    //n1 change with i
    if (n1[ny * nz] != n1[0])
    {
        n1_velocity = 1;
    }
    else if (n1[nz] != n1[0]) //n1 change with j
    {
        n1_velocity = 2;
    }
    //n1 change with k
    else if (n1[1] != n1[0])
    {
        n1_velocity = 3;
    }

    //n2 change with i
    if (n2[ny * nz] != n2[0])
    {
        n2_velocity = 1;
    }
    //n2 change with j
    else if (n2[nz] != n2[0])
    {
        n2_velocity = 2;
    }
    //n2 change with k
    else if (n2[1] != n2[0])
    {
        n2_velocity = 3;
    }

    //n3 change with i
    if (n3[ny * nz] != n3[0])
    {
        n3_velocity = 1;
    }
    //n3 change with j
    else if (n3[nz] != n3[0])
    {
        n3_velocity = 2;
    }
    //n3 change with k
    else if (n3[1] != n3[0])
    {
        n3_velocity = 3;
    }

    return n1_velocity * 100 + n2_velocity * 10 + n3_velocity;
}

//_________________________________________________________________________
void
VRTracker::reorganize_data(int xyz_vel)
//reorganize interpolation data file
// from xyz_velocity=213 to xyz_velocity=123
// so that z chance faster, than y and than x
{
    int t_nx, t_ny, t_nz;
    int i, j, k, m;
    int index, t_index;

    // t_ for a temporary copy
    float *t_x_coord = 0, *t_y_coord = 0, *t_z_coord = 0;
    float *t_n1 = 0, *t_n2 = 0, *t_n3 = 0;

    t_nx = nx;
    t_ny = ny;
    t_nz = nz;

    t_x_coord = new float[nx * ny * nz];
    t_y_coord = new float[nx * ny * nz];
    t_z_coord = new float[nx * ny * nz];

    t_n1 = new float[nx * ny * nz];
    t_n2 = new float[nx * ny * nz];
    t_n3 = new float[nx * ny * nz];

    for (m = 0; m < nx * ny * nz; m++)
    {

        t_x_coord[m] = x_coord[m];
        t_y_coord[m] = y_coord[m];
        t_z_coord[m] = z_coord[m];

        t_n1[m] = n1[m];
        t_n2[m] = n2[m];
        t_n3[m] = n3[m];
    }

    //vertauchen
    if (xyz_vel == 213)
    {
        nx = t_ny;
        ny = t_nx;
        nz = t_nz;

        for (i = 0; i < nx; i++)
        {
            for (j = 0; j < ny; j++)
            {
                for (k = 0; k < nz; k++)
                {
                    index = j * t_ny * t_nz + i * t_nz + k;
                    t_index = i * ny * nz + j * nz + k;

                    x_coord[t_index] = t_x_coord[index];
                    y_coord[t_index] = t_y_coord[index];
                    z_coord[t_index] = t_z_coord[index];

                    n1[t_index] = t_n1[index];
                    n2[t_index] = t_n2[index];
                    n3[t_index] = t_n3[index];
                }
            }
        }
    }
    /*
      for (index=0 ; index < nx*ny*nz ; index++)
      {
      sprintf (interp_message,"%15f %15f %15f %15f %15f %15f",
      x_coord[index],y_coord[index],z_coord[index],
      n1[index],n2[index],n3[index]);
      cout << interp_message << endl;
      }
    */

    if (t_x_coord != 0)
    {
        delete[] t_x_coord;
    }
    if (t_y_coord != 0)
    {
        delete[] t_y_coord;
    }
    if (t_z_coord != 0)
    {
        delete[] t_z_coord;
    }

    if (t_n1 != 0)
    {
        delete[] t_n1;
    }
    if (t_n2 != 0)
    {
        delete[] t_n2;
    }
    if (t_n3 != 0)
    {
        delete[] t_n3;
    }
}

//        functions for the orientation

//__________________________________________________________________________
void
VRTracker::create_trans_basis(void)
{
    int counter, axis, vector, comp;
    int iii, jjj, kkk, index, index_p, index_m;

    int di[3];
    int v_index[3], v_max[3];
    float unit_mat[3][3] = {
        { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 }
    };

    float pp[3][3]; // 3 points around the knot on the pos axis
    float pm[3][3]; // on the negativ axis
    float vt[3][3]; // 3 vectors tangent to the knot
    float vt_len;

    //the same but with the real values
    float ppr[3][3]; // 3 points around the knot on the pos axis
    float pmr[3][3]; // on the negativ axis
    float vtr, vtr_sg;

    if (trans_basis == NULL)
    {
        trans_basis = new float **[nx * ny * nz];
        for (counter = 0; counter < nx * ny * nz; counter++)
        {
            trans_basis[counter] = new float *[3];
            for (vector = 0; vector < 3; vector++)
            {
                trans_basis[counter][vector] = new float[3];
            }
        }
    }

    v_max[0] = nx - 1;
    v_max[1] = ny - 1;
    v_max[2] = nz - 1;
    for (iii = 0; iii < (nx); iii++)
    {
        for (jjj = 0; jjj < (ny); jjj++)
        {
            for (kkk = 0; kkk < (nz); kkk++)
            {
                index = iii * ny * nz + jjj * nz + kkk;
                v_index[0] = iii;
                v_index[1] = jjj;
                v_index[2] = kkk;

                for (axis = 0; axis < 3; axis++)
                {
                    di[0] = 0;
                    di[1] = 0;
                    di[2] = 0;
                    di[axis] = 1;

                    index_p = (iii + di[0]) * ny * nz + (jjj + di[1]) * nz + (kkk + di[2]);
                    index_m = (iii - di[0]) * ny * nz + (jjj - di[1]) * nz + (kkk - di[2]);

                    if (v_index[axis] == 0)
                    {
                        index_m = index;
                    }

                    if (v_index[axis] == v_max[axis])
                    {
                        index_p = index;
                    }

                    pp[axis][0] = x_coord[index_p];
                    pp[axis][1] = y_coord[index_p];
                    pp[axis][2] = z_coord[index_p];
                    pm[axis][0] = x_coord[index_m];
                    pm[axis][1] = y_coord[index_m];
                    pm[axis][2] = z_coord[index_m];

                    ppr[axis][0] = n1[index_p];
                    ppr[axis][1] = n2[index_p];
                    ppr[axis][2] = n3[index_p];
                    pmr[axis][0] = n1[index_m];
                    pmr[axis][1] = n2[index_m];
                    pmr[axis][2] = n3[index_m];

                    vtr = ppr[axis][axis] - pmr[axis][axis];
                    vtr_sg = vtr / fabs(vtr);

                    for (comp = 0; comp < 3; comp++)
                    {
                        vt[axis][comp] = (pp[axis][comp] - pm[axis][comp]);
                    }
                    vt_len = sqrt(vt[axis][0] * vt[axis][0] + vt[axis][1] * vt[axis][1] + vt[axis][2] * vt[axis][2]);

                    for (comp = 0; comp < 3; comp++)
                    {
                        vt[axis][comp] = vt[axis][comp] / vt_len * vtr_sg;
                    }
                }
                for (axis = 0; axis < 3; axis++)
                {

                    linear_equations_sys(vt[0], vt[1], vt[2],
                                         unit_mat[axis], trans_basis[index][axis]);
                };
            }
        }
    }
}

void
VRTracker::linear_equations_sys(const float c0[3], const float c1[3], const float c2[3],
                                const float b[3], float a[3])
{
    // c_ are the columns of the matrix M, Ma=b,only b is known.
    float divisor;

    divisor = determinante(c0, c1, c2);
    if (divisor == 0)
    {
        fprintf(stderr, "M.ERROR:linear_equations_sys:Determinante = 0");
        a[0] = 1;
        a[1] = 0;
        a[2] = 0;
    }
    else
    {
        a[0] = determinante(b, c1, c2) / divisor;
        a[1] = determinante(c0, b, c2) / divisor;
        a[2] = determinante(c0, c1, b) / divisor;
    }
}

float
VRTracker::determinante(const float c0[3], const float c1[3], const float c2[3])
{
    float det;
    det = ((c0[0]) * (c1[1]) * (c2[2])
           + (c0[1]) * (c1[2]) * (c2[0])
           + (c0[2]) * (c1[0]) * (c2[1])
           - (c2[0]) * (c1[1]) * (c0[2])
           - (c2[1]) * (c1[2]) * (c0[0])
           - (c2[2]) * (c1[0]) * (c0[1]));

    if (det == 0)
    {
        cout << "VRTracker::determinante: det = " << det << endl;
    }

    return det;
}
