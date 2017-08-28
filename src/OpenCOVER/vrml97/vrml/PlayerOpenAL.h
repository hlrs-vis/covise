/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_OPENAL_
#define _PLAYER_OPENAL_

#include "Player.h"

#ifdef HAVE_OPENAL
#if defined(__APPLE__)
#include <OpenAL/al.h>
#elif defined(_MSC_VER)
#include <al.h>
#else
#include <AL/al.h>
#endif
#endif

namespace vrml
{

class VRMLEXPORT PlayerOpenAL : public Player
{
public:
#ifndef HAVE_OPENAL
    PlayerOpenAL(const Listener *listener)
        : Player(listener){};
#else
    PlayerOpenAL(const Listener *listener);
    virtual ~PlayerOpenAL();
    virtual void setSpeedOfSound(float speed);
    virtual void update();

    virtual Player::Source *newSource(const Audio *audio);

protected:
    class Source : public Player::Source
    {
    public:
        Source(const Audio *audio);
        virtual ~Source();

        virtual void setMute(bool mute);
        virtual void setLoop(bool loop);
        virtual void setPitch(float pitch);
        virtual void setSpatialize(bool spatialize);
        virtual void setPosition(float x, float y, float z);
        virtual void setVelocity(float vx, float vy, float vz);
        virtual void setIntensity(float intensity);
        virtual void play(double start);
        virtual void stop();
        ALuint alSource;
        ALuint alBuffer;
        ALuint alFirstBuffer;
    };
#endif
};
}
#endif
