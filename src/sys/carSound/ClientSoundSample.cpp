/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ClientSoundSample.h"
#include "soundClient.h"
#include <iostream>
#include <strstream>
#include <QMessageBox>
#include "mainWindow.h"
#include <net/tokenbuffer.h>
#include "remoteSoundMessages.h"

ClientSoundSample::ClientSoundSample(std::string name, soundClient *c)
{
    FMOD_RESULT result;
	sound = NULL;
	channel = NULL;
    fileName = name;
    client = c;
    stopIcon = QIcon(":icons/media-playback-stop.svg");
    playIcon = QIcon(":icons/media-playback-start.svg");
    pauseIcon = QIcon(":icons/media-playback-pause.svg");
    backwardIcon = QIcon(":icons/media-skip-backward.svg");
    forwardIcon = QIcon(":icons/media-skip-forward.svg");

    ID = IDCounter++;

    result = mainWindow::instance()->system->createSound(name.c_str(), FMOD_DEFAULT | FMOD_LOOP_NORMAL, 0, &sound); // FMOD_DEFAULT uses the defaults.  These are the same as FMOD_LOOP_OFF | FMOD_2D | FMOD_HARDWARE.
    //ERRCHECK(result);
    result = mainWindow::instance()->system->playSound(sound, NULL, true, &channel);
    //ERRCHECK(result);

    myItem = new QTreeWidgetItem(mainWindow::instance()->soundTable);
    myItem->setText(SoundColumns::CSoundID, QString::number(ID));
    myItem->setText(SoundColumns::CClient, QString::number(client->ID));
    myItem->setText(SoundColumns::CFileName,fileName.c_str());
    myItem->setIcon(SoundColumns::CState, stopIcon);

    covise::TokenBuffer tb;
    tb << (int)SoundMessages::SOUND_SOUND_ID;
    tb << ID;
    client->send(tb);
}

ClientSoundSample::~ClientSoundSample()
{
    delete myItem;
    channel->setPaused(true);
    sound->release();
    if (mainWindow::instance()->currentSound == this)
        mainWindow::instance()->currentSound = nullptr;
}

void ClientSoundSample::start()
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
    myItem->setIcon(SoundColumns::CState, playIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::continuePlaying()
{
    playing = true;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    myItem->setIcon(SoundColumns::CState, playIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::stop()
{
    playing = false;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    channel->setPosition(0, FMOD_TIMEUNIT_PCM);
    myItem->setIcon(SoundColumns::CState, stopIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::rewind(){};
void ClientSoundSample::loop(bool l,int count)
{
    looping = l;
    if (l)
        channel->setLoopCount(count);
    else
        channel->setLoopCount(0);
};

void ClientSoundSample::volume(float v)
{
    channel->setVolume(v);
    myItem->setText(SoundColumns::CVolume, QString::number(v));
}
void ClientSoundSample::pitch(float p)
{
    channel->setPitch(p);
    myItem->setText(SoundColumns::CPitch, QString::number(p));
}

void ClientSoundSample::setDelay(unsigned long long dspclock_start, unsigned long long dspclock_end, bool stopchannels)
{
    channel->setDelay(dspclock_start, dspclock_end, stopchannels);
}

int ClientSoundSample::IDCounter = 10;