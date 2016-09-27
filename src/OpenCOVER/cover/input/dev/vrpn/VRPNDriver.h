/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * vrpn.h
 *
 *  Created on: Dec 9, 2015
 *      Author: uwe
 */

#ifndef VRPN_DRIVER_H
#define VRPN_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>


#include <cover/input/inputdevice.h>

/**
 * @brief The VRPNDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */

#define _TIMEZONE_DEFINED

#include <vrpn_Tracker.h>
#include <vrpn_Button.h>

class VRPNDriver : public opencover::InputDevice
{
    //-------------------VRPN related stuff
    
    vrpn_Tracker_Remote *vrpnTracker;
    vrpn_Button_Remote *vrpnButton;
    
    
    std::string trackerid;
    std::string buttonid;

    virtual bool poll();
    
    void processTrackerData(const vrpn_TRACKERCB &vrpnData);
    void processButtonData(const vrpn_BUTTONCB &vrpnButtonData);

public:
    VRPNDriver(const std::string &name);
    virtual ~VRPNDriver();
    
    static void vrpnCallback(void *thisclass, const vrpn_TRACKERCB t);
    static void vrpnButtonCallback(void *userdata, const vrpn_BUTTONCB t);
    
};
#endif
