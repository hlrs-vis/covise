/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#ifdef WIN32
#include <windows.h>
//#include "lusb0_usb.h"
#include <conio.h>
#else
//#include <usb.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPlugin.h>
#include <OpenThreads/Thread>
#include <cover/input/dev/Joystick/Joystick.h>



#define MAX_CALLSIGN_LEN        8



#ifndef Wheelchair_H
#define Wheelchair_H

#pragma pack(push, 1)
struct SBData 
{
    float fl;
    float fr;
    float rl;
    float rr;
    unsigned int button;
};
#pragma pack(pop)



#pragma pack(push, 1)
struct SBCtrlData
{
    unsigned char  cmd;
    int 	  value; 
};
#pragma pack(pop)



class PLUGINEXPORT Wheelchair :  public opencover::coVRPlugin, public OpenThreads::Thread, public opencover::coVRNavigationProvider
{
public:
    Wheelchair();
    ~Wheelchair();
    bool update();
    osg::Vec3d getNormal();
    float getWeight();// weight in Kg
    virtual void run();
    volatile bool running;
    void stop();
    void Initialize();
    unsigned char getButton();
    void syncData();
    bool doStop;
private:
    float stepSizeUp;
    float stepSizeDown;
    bool init();
    void MoveToFloor();
    virtual void setEnabled(bool);
    void updateThread();

    SBData sbData;
    SBCtrlData sbControl;
    int ret;
    OpenThreads::Mutex mutex;
    float speed=0.0;
    osg::Node *oldFloorNode;
    osg::NodePath oldNodePath;
    osg::Matrix oldFloorMatrix;
    osg::Matrix TransformMat;
    Joystick* dev;
    int joystickNumber;
    int xIndex;
    int yIndex;
    float xScale;
    float yScale;
    bool debugPrint;
};

#endif /* Wheelchair_H */