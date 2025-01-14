/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/*
* inputhdw.cpp
*
*  Created on: Dec 9, 2014
*      Author: woessner
*/
#include "zSpaceDriver.h"

#include <config/CoviseConfig.h>

#include <iostream>
#include <osg/Matrix>

#include <OpenVRUI/osg/mathUtils.h> //for MAKE_EULER_MAT

using namespace std;
using namespace covise;

#include <util/unixcompat.h>
#include <iostream>

//#include <quat.h>

using namespace std;
zSpaceDriver::zSpaceDriver(const std::string &config)
    : InputDevice(config)
{

    zSpaceContext=NULL;
    primaryStylusHandle = NULL;
    secondaryStylusHandle = NULL;
    headHandle = NULL;
    displayHandle = NULL;

    ZSError   error = zsInitialize(&zSpaceContext);
    if(checkError(error)==false)
    {

        ZSError   error = zsFindDisplayByIndex(&zSpaceContext,0,&displayHandle);
        if(checkError(error)==false)
        {

            m_bodyMatricesValid.resize(3);
            m_bodyMatrices.resize(3);

            m_bodyMatricesValid[0]=false;
            m_bodyMatricesValid[1]=false;
            m_bodyMatricesValid[2]=false;
            numPrimaryButtons = 0;
            numSecondaryButtons = 0;

            error = zsFindTargetByType(zSpaceContext, ZS_TARGET_TYPE_SECONDARY, 0, &secondaryStylusHandle);
            if(checkError(error)==false)
            {
                zsGetNumTargetButtons(primaryStylusHandle,&numSecondaryButtons);
                m_buttonStates.resize(numPrimaryButtons+numSecondaryButtons);
            }

            error = zsFindTargetByType(zSpaceContext, ZS_TARGET_TYPE_HEAD, 0, &headHandle);
            if(checkError(error)==false)
            {

                error = zsFindTargetByType(zSpaceContext, ZS_TARGET_TYPE_PRIMARY, 0, &primaryStylusHandle);
                if(checkError(error)==false)
                {
                    zsGetNumTargetButtons(primaryStylusHandle,&numPrimaryButtons);
                    m_buttonStates.resize(numPrimaryButtons+numSecondaryButtons);
                }
                /*    // Set the stylus' move event threshold to the following: 
                // Time     -> 0.01  seconds 
                // Distance -> 0.001 meters 
                // Angle    -> 0.01  degrees 
                error = zsSetTargetMoveEventThresholds(stylusHandle, 0.01f, 0.001f, 0.01f);
                if(checkError(error)==false)
                {
                // Register event handlers.
                error = zsAddTrackerEventHandler(stylusHandle, ZS_TRACKER_EVENT_ALL, &handleButtonEvent, this);
                checkError(error);
                }*/
            }
        }
    }
}

bool zSpaceDriver::poll()
{
    if (zSpaceContext==NULL || primaryStylusHandle==NULL || headHandle==NULL)
        return false;
    zsUpdate(zSpaceContext);
    ZSTrackerPose pose;
    zsGetTargetPose(headHandle,&pose);
    setMatrix(0,&pose);
    zsGetTargetPose(primaryStylusHandle,&pose);
    setMatrix(1,&pose);
    zsGetTargetPose(secondaryStylusHandle,&pose);
    setMatrix(2,&pose);
    for(int i=0;i<numPrimaryButtons;i++)
    {
        ZSBool state;
        zsIsTargetButtonPressed(primaryStylusHandle,i,&state);
        m_buttonStates[i]=state!=0;
    }
    for(int i=0;i<numSecondaryButtons;i++)
    {
        ZSBool state;
        zsIsTargetButtonPressed(secondaryStylusHandle,i,&state);
        m_buttonStates[numPrimaryButtons+i]=state!=0;
    }
    float x,y,z;
    zsGetDisplayAngle(displayHandle,&x,&y,&z);
    //fprintf(stderr,"x: %f  y:%f  z:%f\n",x,y,z);

    return true;
}

void zSpaceDriver::setMatrix(int num, const ZSTrackerPose* pose)
{
    osg::Matrix matrix;
    for(int n=0;n<4;n++)
        for(int m=0;m<4;m++)
            matrix(n,m) = pose->matrix.f[m+ 4* n];
    matrix(3,0)*=1000;
    matrix(3,1)*=1000;
    matrix(3,2)*=1000;
    m_bodyMatricesValid[num]=true;
    m_bodyMatrices[num]=matrix;
}

//====================END of init section============================


zSpaceDriver::~zSpaceDriver()
{
    stopLoop();

    ZSError error = zsShutdown(zSpaceContext);
    checkError(error);
}

//==========================main loop =================


/*
void zSpaceDriver::processButtonEvent(ZSHandle targetHandle, const ZSTrackerEventData* eventData)
{

m_mutex.lock();

if (m_buttonStates.size() <= eventData->buttonId)
{
m_buttonStates.resize(eventData->buttonId+1);
}
if(eventData->type == ZS_TRACKER_EVENT_BUTTON_PRESS)
{
m_buttonStates[eventData->buttonId]=true;
}
else if(eventData->type == ZS_TRACKER_EVENT_BUTTON_RELEASE)
{
m_buttonStates[eventData->buttonId]=false;
}
int stylusNumber=0;

if (m_bodyMatricesValid.size() <= stylusNumber)
{
m_bodyMatricesValid.resize(stylusNumber+1);
m_bodyMatrices.resize(stylusNumber+1);
}

osg::Matrix matrix;
for(int n=0;n<3;n++)
for(int m=0;m<3;m++)
matrix(n,m) = eventData->poseMatrix.f[n+3*m];
m_bodyMatricesValid[stylusNumber]=true;
m_bodyMatrices[stylusNumber]=matrix;
m_mutex.unlock();

}*/

bool zSpaceDriver::checkError(ZSError error)
{
    if (error != ZS_ERROR_OKAY)
    {
        char errorString[256];
        zsGetErrorString(error, errorString, sizeof(errorString));
        fprintf(stderr, "%s\n", errorString);
        return true;
    }
    return false;
}

/*void zSpaceDriver::handleButtonEvent(ZSHandle targetHandle, const ZSTrackerEventData* eventData, const void* userData)
{
zSpaceDriver *zSpace = reinterpret_cast<zSpaceDriver *>(userData);
zSpace->processButtonEvent(targetHandle,eventData);
}*/

void zSpaceDriver::update() //< called by Input::update()
{
    poll();
    InputDevice::update();
}

INPUT_PLUGIN(zSpaceDriver)
