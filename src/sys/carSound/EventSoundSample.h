/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EVENT_SOUND_SAMPLE_H
#define EVENT_SOUND_SAMPLE_H

#include <string>
#include <fmod_studio.hpp>

class EventSoundSample
{
public:
    EventSoundSample(std::string name);
    ~EventSoundSample();
    bool isPlaying()
    {
        return playing;
    };
    void start();
    void continuePlaying();
    void stop();
    void rewind();
    void loop(bool l);

private:
    FMOD::Sound *sound;
    FMOD::Channel *channel;
    bool playing;
    bool looping;
};

#endif