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



#define MAX_CALLSIGN_LEN        8


class UDPComm;

#ifndef Skateboard_H
#define Skateboard_H

#pragma pack(push, 1)
struct SBData 
{
    float rr;
    float rl;
    float fr;
    float fl;
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



class PLUGINEXPORT Skateboard : public opencover::coVRPlugin, public OpenThreads::Thread, public opencover::coVRNavigationProvider
{
public:
    Skateboard();
    ~Skateboard();
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
    float getYAccelaration();
    float oldAngle = 0;
    float wheelBase = 0.445;
    osg::Matrix getBoardXMatrix();
    osg::Matrix getBoardMatrix();
    virtual void setEnabled(bool);
    void updateThread();

    UDPComm *udp;
    SBData sbData;
    SBCtrlData sbControl;
    int ret;
    OpenThreads::Mutex mutex;
    float speed=0.0;
    osg::Node *oldFloorNode;
    osg::NodePath oldNodePath;
    osg::Matrix oldFloorMatrix;
    osg::Matrix TransformMat;
    osg::Matrix SkateboardPos;
};

#endif /* Skateboard_H */
