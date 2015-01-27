/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FKFS_H
#define __FKFS_H

#include <util/common.h>
#include "FFWheel.h"

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>
#include "UDPComm.h"

#ifndef WIN32
#include <termios.h>
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <termio.h> /* tcsetattr */
#include <limits.h> /* sginap */
#endif

typedef struct
{
    double t_req;
    double Enable;
    double Mode;
    double msgcount;
    double req_inertia;
} SIM2RT;

typedef struct
{
    double Alpha;
    double Speed;
    double Accel;
    double Brake;
    double Clutch;
    double selectorLever;
    //double gearSelector;
    double Ignition;
    double parkBrake;
    double Ready;
    double auxsw1;
    double auxsw2;
    double auxpot1;
    double auxpot2;
    double reserved1;
    double Simtime;
} RT2SIM;

class PLUGINEXPORT FKFS : public FFWheel
{
public:
    FKFS();
    virtual ~FKFS();
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits
    virtual void update();
    virtual double getAngle(); // return steering wheel angle
    virtual double getfastAngle(); // return steering wheel angle
    virtual void setRoadFactor(float){}; // set roughness

protected:
    void sendData();
    bool readData(); // returns true on success, false if no data has been received.
    RT2SIM receiveBuffer;
    RT2SIM appReceiveBuffer;
    SIM2RT sendBuffer;
    UDPComm *toFKFS;
    OpenThreads::Barrier endBarrier;
    double maxAngle;
    double origin;
};
#endif
