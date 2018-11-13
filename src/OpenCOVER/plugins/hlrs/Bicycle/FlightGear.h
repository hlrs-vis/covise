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

#ifndef FLIGHTGEAR_H
#define FLIGHTGEAR_H

#pragma pack(push, 1)
struct FGData 
{
    //Header
    unsigned int  Magic;                  // Magic Value
    unsigned int  Version;                // Protocoll version
    unsigned int  MsgId;                  // Message identifier 
    unsigned int  MsgLen;                 // absolute length of message
    unsigned int  ReplyAddress;           // (player's receiver address
    unsigned int  ReplyPort;              // player's receiver port
    char Callsign[MAX_CALLSIGN_LEN];    // Callsign used by the player
    
    //Data
    unsigned char Model[96]; //1-96 
    double time; //97-104
    double lag; //105-112
    double position[3]; // 113-136
    float orientation[3]; // 137-148
    float linearVel[3]; // 149-160
    float angularVel[3]; // 161-172
    float linearAccel[3]; // 173-184
    float angularAccel[3]; // 184-196
    float pad; //197-200

};
#pragma pack(pop)



#pragma pack(push, 1)
struct FGControl
{
    int           magnetos;        
    bool 	  starter; 
    double 	  throttle; 
    double        parkingBrake;
};
#pragma pack(pop)



class PLUGINEXPORT FlightGear :  public OpenThreads::Thread
{
public:
    FlightGear(BicyclePlugin*);
    ~FlightGear();
    void update();
    osg::Vec3d getPosition();
    osg::Vec3d getOrientation();
    virtual void run();
    volatile bool running;
    void stop();
    bool doStop;
    void setThermal(bool thermalActivity);
private:
    void init();

    UDPComm *udp;
    BicyclePlugin* bicycle;
    FGData fgdata;
    FGControl fgcontrol;
    int ret;
    OpenThreads::Mutex mutex;
    bool thermal;
};

#endif /* FLIGHTGEAR_H */
