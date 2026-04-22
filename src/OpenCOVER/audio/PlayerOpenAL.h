/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_OPENAL_
#define _PLAYER_OPENAL_

#include "Player.h"

#include <al.h>
#include <util/coExport.h>

#include "AlutContext.h"

namespace opencover::audio
{

class COVRAUDIOEXPORT PlayerOpenAL : public Player
{
public:
    PlayerOpenAL(const Listener *listener);
    virtual void update();

    virtual std::unique_ptr<opencover::audio::Source> makeSource(const Audio *audio);

protected:
    class Source : public opencover::audio::Source
    {
    public:
        Source(Player *player, const Audio *audio);
        virtual ~Source();

        virtual void setLoop(bool loop) override;
        virtual void setPitch(float pitch) override;
        virtual void setSpatialize(bool spatialize) override;
        virtual void setPosition(float x, float y, float z) override;
        virtual void setVelocity(float vx, float vy, float vz) override;
        virtual void setIntensity(float intensity) override;
        virtual void play(double start) override;
        virtual void stop() override;
        ALuint alSource;
    };

private:
    AlutContext alutContext;
};
}
#endif
