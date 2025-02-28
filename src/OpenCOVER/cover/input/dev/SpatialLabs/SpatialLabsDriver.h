/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * SpatialLabs.h
 *
 *  Created on: Dec 9, 2015
 *      Author: uwe
 */

#ifndef SpatialLabs_DRIVER_H
#define SpatialLabs_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>


#include <cover/input/inputdevice.h>

/**
 * @brief The SpatialLabsDriver class interacts with input hardware
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
#include <SpatialLabsCoreLib.h>



class SpatialLabsDriver : public opencover::InputDevice
{
    //-------------------SpatialLabs related stuff


    float FOVPlayerCamera = 90;
    float FocalLengthPlayerCamera = 0.4;

    float CachedRawEyeLeft[3] = { 0.0, 0.0, 0.0 };
    float CachedRawEyeRight[3] = { 0.0, 0.0, 0.0 };
    float CachedEyeLeft[3] = { 0.0, 0.0, 0.0 };
    float CachedEyeRight[3] = { 0.0, 0.0, 0.0 };
    float CachedEyeLeftScreenSpace[3] = { 0.0, 0.0, 0.0 };
    float CachedEyeRightScreenSpace[3] = { 0.0, 0.0, 0.0 };
    float CachedViewportSize[2] = { 0.0, 0.0 };

    int numPrimaryButtons;
    int numSecondaryButtons;
    SpatialLabsCoreLib::SpatialLabsCoreLibAPI *spatialLabsAPI;

    
    //void processButtonEvent(ZSHandle targetHandle, const ZSTrackerEventData* eventData);

    bool init();
    virtual bool poll();
    
    void setMatrix(int num, const osg::Matrix* pose);
    virtual void update(); //< called by Input::update()

public:
    SpatialLabsDriver(const std::string &name);
    virtual ~SpatialLabsDriver();
    virtual bool needsThread() const {return false; } ; //< whether a thread should be spawned - reimplement if not necessary   we don't need a thread


};
#endif
