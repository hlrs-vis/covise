/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __KEYBOARD_H
#define __KEYBOARD_H

#include <util/common.h>
#include "FFWheel.h"
// #include "VehicleDynamicsInstitutes.h"

typedef struct
{
    double alpha;
    double gas;
    double brakePedal;
    double clutch;
    int gearDiff;
    bool horn;
    bool reset;
} KeyboardState;

class PLUGINEXPORT Keyboard : public FFWheel
{
public:
    Keyboard();
    virtual ~Keyboard();
    virtual void run(); // receiving and sending thread, also does the low level simulation like hard limits
    void update();
    double getAngle(); // return steering wheel angle
    double getfastAngle(); // return steering wheel angle
    virtual void setRoadFactor(float); // set roughness 0 == Road 1 == rough

    void leftKeyDown();
    void leftKeyUp();
    void rightKeyDown();
    void rightKeyUp();
    void foreKeyDown();
    void foreKeyUp();
    void backKeyDown();
    void backKeyUp();
    void gearShiftUpKeyDown();
    void gearShiftUpKeyUp();
    void gearShiftDownKeyDown();
    void gearShiftDownKeyUp();
    void hornKeyDown();
    void hornKeyUp();
    void resetKeyDown();
    void resetKeyUp();

    double getGas();
    double getBrake();
    double getClutch();

    int getGearDiff();
    bool getHorn();
    bool getReset();

protected:
    KeyboardState receiveBuffer;
    KeyboardState appReceiveBuffer;

    double angle;
    double roadFactor;

    double k;
    double m;
    double l;

    double steeringForce;
};
#endif
