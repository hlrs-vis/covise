/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CAR_SOUND_H
#define CAR_SOUND_H

#include <util/common.h>

#include "UDPComm.h"
struct SoundData
{
    float engineSpeed;
    float carSpeed;
    float torque;
    float slip;
};
class CarSound
{
public:
    enum SoundSource
    {
        Horn = 0,
        Ignition = 1,
        GearMiss = 2
    };
    ~CarSound();
    static CarSound *instance();
    void setCarSpeed(float speed);
    void setSpeed(float speed);
    void setTorque(float t);
    void setSlip(float s);
    void start(enum SoundSource ss);
    void stop(enum SoundSource ss);
    void rewind(enum SoundSource ss);
    void continuePlaying(enum SoundSource ss);
    void loop(enum SoundSource ss, bool state);

private:
    CarSound();
    static CarSound *theInstance;
    SoundData sd;
    UDPComm *toCarSound;
};

#endif
