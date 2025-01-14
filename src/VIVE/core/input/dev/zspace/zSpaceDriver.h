/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * zSpace.h
 *
 *  Created on: Dec 9, 2015
 *      Author: uwe
 */

#ifndef zSpace_DRIVER_H
#define zSpace_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>


#include <cover/input/inputdevice.h>

/**
 * @brief The zSpaceDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */

#define _TIMEZONE_DEFINED

#include <iostream>

#include <Windows.h>
#include <zSpace.h>



class zSpaceDriver : public opencover::InputDevice
{
    //-------------------zSpace related stuff

    ZSContext zSpaceContext;
    ZSHandle displayHandle;
    ZSHandle primaryStylusHandle;
    int numPrimaryButtons;
    ZSHandle secondaryStylusHandle;
    int numSecondaryButtons;
    ZSHandle headHandle;

    
    //void processButtonEvent(ZSHandle targetHandle, const ZSTrackerEventData* eventData);

    bool init();
    virtual bool poll();
    
    void setMatrix(int num, const ZSTrackerPose* pose);
    virtual void update(); //< called by Input::update()

public:
    zSpaceDriver(const std::string &name);
    virtual ~zSpaceDriver();
    virtual bool needsThread() const {return false; } ; //< whether a thread should be spawned - reimplement if not necessary   we don't need a thread


    //static void handleButtonEvent(ZSHandle targetHandle, const ZSTrackerEventData* eventData, const void* userData);

    bool checkError(ZSError error);

};
#endif
