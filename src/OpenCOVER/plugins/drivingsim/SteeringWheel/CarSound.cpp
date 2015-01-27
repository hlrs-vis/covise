/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
#include "CarSound.h"

#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>
using namespace covise;
using namespace opencover;

CarSound *CarSound::theInstance = NULL;

CarSound *CarSound::instance()
{
    if (theInstance == NULL)
    {
        theInstance = new CarSound();
    }
    return theInstance;
}

CarSound::CarSound()
{
    std::string hostname;
    int port;

    port = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.CarSound", 31804);
    hostname = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.CarSound", "localhost");
    toCarSound = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        toCarSound = new UDPComm(port, hostname.c_str());
    }
}
CarSound::~CarSound()
{
    delete toCarSound;
    theInstance = NULL;
}
void CarSound::setSlip(float s)
{
    sd.slip = s;
}
void CarSound::setTorque(float t)
{
    sd.torque = t;
}
void CarSound::setCarSpeed(float s)
{
    sd.carSpeed = s;
}
void CarSound::setSpeed(float speed)
{
    sd.engineSpeed = speed;
    if (sd.engineSpeed < 13.33)
        sd.engineSpeed = 13.33f;
    if (toCarSound)
        toCarSound->send(&sd, sizeof(sd));
}
void CarSound::stop(enum SoundSource ss)
{
    char buf[2];
    buf[0] = '\0';
    buf[1] = (char)ss;
    if (toCarSound)
        toCarSound->send(&buf, 2);
}
void CarSound::start(enum SoundSource ss)
{
    char buf[2];
    buf[0] = '\1';
    buf[1] = (char)ss;
    if (toCarSound)
        toCarSound->send(&buf, 2);
}
void CarSound::continuePlaying(enum SoundSource ss)
{
    char buf[2];
    buf[0] = '\2';
    buf[1] = (char)ss;
    if (toCarSound)
        toCarSound->send(&buf, 2);
}
void CarSound::loop(enum SoundSource ss, bool state)
{
    char buf[2];
    if (state)
        buf[0] = '\3';
    else
        buf[0] = '\4';
    buf[1] = (char)ss;
    if (toCarSound)
        toCarSound->send(&buf, 2);
}
void CarSound::rewind(enum SoundSource ss)
{
    char buf[2];
    buf[0] = '\5';
    buf[1] = (char)ss;
    if (toCarSound)
        toCarSound->send(&buf, 2);
}
