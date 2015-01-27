/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __FFWheel_H
#define __FFWheel_H

#include <util/common.h>

#include <OpenThreads/Thread>
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>

class PLUGINEXPORT FFWheel : public OpenThreads::Thread
{
public:
    FFWheel();
    virtual ~FFWheel();
    virtual void run() = 0; // receiving and sending thread, also does the low level simulation like hard limits
    virtual void update() = 0;
    virtual double getAngle() = 0; // return steering wheel angle
    virtual double getfastAngle() = 0; // return steering wheel angle
    virtual void setRoadFactor(float) = 0; // set roughness
    volatile bool doRun;
    virtual void softResetWheel();
    virtual void cruelResetWheel();
    virtual void shutdownWheel();

    float carVel;

protected:
    OpenThreads::Barrier endBarrier;
    double maxAngle;
    double origin;
};
#endif
