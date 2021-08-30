#pragma once
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CLIENT_SOUND_SAMPLE_H
#define CLIENT_SOUND_SAMPLE_H

#include <string>
#include <fmod_studio.hpp>
#include <QtGui/qicon.h>

class QTreeWidgetItem;

enum  SoundColumns {
    CSoundID,
    CState,
    CClient,
    CFileName,
    CVolume,
    CPitch,
};
class soundClient;

class ClientSoundSample
{
public:
    ClientSoundSample(const std::string &name, size_t fileSize, time_t fileTime, soundClient*c);
    ~ClientSoundSample();
    bool isPlaying()
    {
        return playing;
    };
    void start();
    void continuePlaying();
    void stop();
    void rewind();
    void loop(bool l, int count);
    void volume(float v);
    void pitch(float p);
    void setDelay(unsigned long long dspclock_start, unsigned long long dspclock_end, bool stopchannels);
    int ID=-1;

    QTreeWidgetItem* myItem;
private:
    FMOD::Sound *sound;
    FMOD::Channel *channel;
    bool playing;
    bool looping;
    std::string fileName;
    std::string cacheFileName;
    soundClient* client;
    static int IDCounter;
    QIcon stopIcon;
    QIcon playIcon;
    QIcon pauseIcon;
    QIcon backwardIcon;
    QIcon forwardIcon;
    std::string createCacheFileName(const std::string &fileName);

};

#endif