/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Keyboard.h"

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#include "SteeringWheel.h"

Keyboard::Keyboard()
{
    receiveBuffer.alpha = 0;
    receiveBuffer.gas = 0;
    receiveBuffer.brakePedal = 0;
    receiveBuffer.clutch = 0;
    receiveBuffer.gearDiff = 0;
    receiveBuffer.horn = false;
    receiveBuffer.reset = false;
    appReceiveBuffer.alpha = 0;
    appReceiveBuffer.gas = 0;
    appReceiveBuffer.brakePedal = 0;
    appReceiveBuffer.clutch = 0;
    appReceiveBuffer.gearDiff = 0;
    appReceiveBuffer.horn = false;
    appReceiveBuffer.reset = false;

    k = 0.04;
    m = 0.02;
    l = 0.02;
    maxAngle = 2.0;
    origin = 0;
    carVel = 0;

    if (coVRMSController::instance()->isMaster())
    {
        doRun = true;
        //wheel->niceProcess(1);
        //wheel->setAffinity(0);
        Init();
        startThread();
    }
}

Keyboard::~Keyboard()
{
    doRun = false;
}

double Keyboard::getAngle() // return steering wheel receiveBuffer.alpha
{
    return appReceiveBuffer.alpha;
}

double Keyboard::getfastAngle() // return steering wheel receiveBuffer.alpha
{
    return receiveBuffer.alpha;
}

void Keyboard::setRoadFactor(float r) // set roughness
{
    roadFactor = r;
}

void Keyboard::update()
{
    if (coVRMSController::instance()->isMaster())
    {
        memcpy(&appReceiveBuffer, &receiveBuffer, sizeof(KeyboardState));
        coVRMSController::instance()->sendSlaves((char *)&appReceiveBuffer, sizeof(KeyboardState));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&appReceiveBuffer, sizeof(KeyboardState));
    }
}

void Keyboard::run() // receiving and sending thread, also does the low level simulation like hard limits
{
    while (doRun)
    {
        receiveBuffer.alpha += l * steeringForce - k * tanh(m * fabs(carVel)) * receiveBuffer.alpha;
        if (receiveBuffer.alpha < -maxAngle)
            receiveBuffer.alpha = -maxAngle;
        else if (receiveBuffer.alpha > maxAngle)
            receiveBuffer.alpha = maxAngle;

        //std::cerr << "Gear Diff: " << receiveBuffer.gearDiff << std::endl;

        usleep(1000);
    }
}

void Keyboard::leftKeyDown()
{
    steeringForce = 1.0;
}
void Keyboard::leftKeyUp()
{
    steeringForce = 0.0;
}

void Keyboard::rightKeyDown()
{
    steeringForce = -1.0;
}
void Keyboard::rightKeyUp()
{
    steeringForce = 0.0;
}

void Keyboard::foreKeyDown()
{
    receiveBuffer.gas = 1.0;
}
void Keyboard::foreKeyUp()
{
    receiveBuffer.gas = 0.0;
}
void Keyboard::gearShiftUpKeyDown()
{
    receiveBuffer.gearDiff = 1;
}
void Keyboard::gearShiftUpKeyUp()
{
    receiveBuffer.gearDiff = 0;
}

void Keyboard::gearShiftDownKeyDown()
{
    receiveBuffer.gearDiff = -1;
}
void Keyboard::gearShiftDownKeyUp()
{
    receiveBuffer.gearDiff = 0;
}

void Keyboard::backKeyDown()
{
    receiveBuffer.brakePedal = 1.0;
}
void Keyboard::backKeyUp()
{
    receiveBuffer.brakePedal = 0.0;
}

void Keyboard::hornKeyDown()
{
    receiveBuffer.horn = true;
}
void Keyboard::hornKeyUp()
{
    receiveBuffer.horn = false;
}

void Keyboard::resetKeyDown()
{
    receiveBuffer.reset = true;
}
void Keyboard::resetKeyUp()
{
    receiveBuffer.reset = false;
}

double Keyboard::getGas()
{
    return appReceiveBuffer.gas;
}
double Keyboard::getBrake()
{
    return appReceiveBuffer.brakePedal;
}
double Keyboard::getClutch()
{
    return appReceiveBuffer.clutch;
}

int Keyboard::getGearDiff()
{
    return appReceiveBuffer.gearDiff;
}

bool Keyboard::getHorn()
{
    return appReceiveBuffer.horn;
}

bool Keyboard::getReset()
{
    return appReceiveBuffer.reset;
}
