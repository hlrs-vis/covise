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
    sd.msgType = TypeCarSound;
    ssd.msgType = TypeSimpleSound;
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
    ssd.action = '\0';
    ssd.soundNum = (char)ss;
    if (toCarSound)
        toCarSound->send(&ssd, sizeof(ssd));
}
void CarSound::start(enum SoundSource ss)
{
    ssd.action = '\1';
    ssd.soundNum = (char)ss;
    if (toCarSound)
        toCarSound->send(&ssd, sizeof(ssd));
}
void CarSound::continuePlaying(enum SoundSource ss)
{
    ssd.action = '\2';
    ssd.soundNum = (char)ss;
    if (toCarSound)
        toCarSound->send(&ssd, sizeof(ssd));
}
void CarSound::loop(enum SoundSource ss, bool state)
{
    if (state)
        ssd.action = '\3';
    else
        ssd.action = '\4';
    ssd.soundNum = (char)ss;
    if (toCarSound)
        toCarSound->send(&ssd, sizeof(ssd));
}
void CarSound::rewind(enum SoundSource ss)
{
    ssd.action = '\5';
    ssd.soundNum = (char)ss;
    if (toCarSound)
        toCarSound->send(&ssd, sizeof(ssd));
}
