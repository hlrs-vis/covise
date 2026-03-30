/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_OPENAL_
#define _PLAYER_OPENAL_

#include "Player.h"

#if defined(__APPLE__)
#include <OpenAL/al.h>
#elif defined(_MSC_VER)
#include <al.h>
#else
#include <AL/al.h>
#endif
#include <util/coExport.h>

#include "AlutContext.h"

namespace opencover::audio
{

class COVEREXPORT PlayerOpenAL : public Player
{
public:
    PlayerOpenAL(const Listener *listener);
    virtual void update();

    virtual Player::Source *newSource(const Audio *audio);

protected:
    class Source : public Player::Source
    {
    public:
        Source(const Audio *audio);
        virtual ~Source();

        virtual void setLoop(bool loop);
        virtual void setPitch(float pitch);
        virtual void setSpatialize(bool spatialize);
        virtual void setPosition(float x, float y, float z);
        virtual void setVelocity(float vx, float vy, float vz);
        virtual void setIntensity(float intensity);
        virtual void play(double start);
        virtual void stop();
        ALuint alSource;
    };

private:
    AlutContext alutContext;
};
}
#endif
