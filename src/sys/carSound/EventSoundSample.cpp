/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "EventSoundSample.h"
#include <iostream>
#include <strstream>
#include <QMessageBox>
#include "mainWindow.h"

EventSoundSample::EventSoundSample(std::string name)
{
    FMOD_RESULT result;
	sound = NULL;
	channel = NULL;
    result = mainWindow::instance()->system->createSound(name.c_str(), FMOD_DEFAULT | FMOD_LOOP_NORMAL, 0, &sound); // FMOD_DEFAULT uses the defaults.  These are the same as FMOD_LOOP_OFF | FMOD_2D | FMOD_HARDWARE.
    //ERRCHECK(result);
    result = mainWindow::instance()->system->playSound(sound, NULL, true, &channel);
    //ERRCHECK(result);
}

EventSoundSample::~EventSoundSample()
{
}

void EventSoundSample::start()
{
    bool isp;
    channel->isPlaying(&isp);
    if (!isp)
    {
		mainWindow::instance()->system->playSound(sound, NULL, true, &channel);
        if (looping)
            channel->setLoopCount(-1);
        else
            channel->setLoopCount(0);
    }
    playing = true;
    FMOD_RESULT result;
    result = channel->setPosition(0, FMOD_TIMEUNIT_PCM);
    channel->setPaused(!playing); // This is where the sound really starts.
    //ERRCHECK(result);
};
void EventSoundSample::continuePlaying()
{
    playing = true;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    //ERRCHECK(result);
};
void EventSoundSample::stop()
{
    playing = false;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    channel->setPosition(0, FMOD_TIMEUNIT_PCM);
    //ERRCHECK(result);
};
void EventSoundSample::rewind(){};
void EventSoundSample::loop(bool l)
{
    looping = l;
    if (l)
        channel->setLoopCount(-1);
    else
        channel->setLoopCount(0);
};
