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



#define MAX_CALLSIGN_LEN        8


class BicyclePlugin;
class UDPComm;

#ifndef Skateboard_H
#define Skateboard_H

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



class PLUGINEXPORT Skateboard :  public OpenThreads::Thread
{
public:
    Skateboard(BicyclePlugin*);
    ~Skateboard();
    void update();
    osg::Vec3d getNormal();
    float getWeight();// weight in Kg
    virtual void run();
    volatile bool running;
    void stop();
    void Initialize();
    unsigned char getButton();
    bool doStop;
private:
    void init();

    UDPComm *udp;
    BicyclePlugin* bicycle;
    SBData sbData;
    SBCtrlData sbControl;
    int ret;
    OpenThreads::Mutex mutex;
};

#endif /* Skateboard_H */
