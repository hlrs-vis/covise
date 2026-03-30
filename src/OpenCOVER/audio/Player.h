/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_
#define _PLAYER_

#include <util/coExport.h>
#include "Audio.h"

#include <vector>
#include <glm/vec3.hpp>

#include "Listener.h"

namespace opencover::audio
{

class COVEREXPORT Player
{
public:
    class Source
    {
        friend class Player;
        friend class PlayerOpenAL;
        friend class PlayerAServer;
        friend class PlayerOsc;

    public:
        Source(const Audio *audio);
        virtual void setPitch(float pitch);
        virtual void setIntensity(float intensity);
        virtual void setStart(double start);
        virtual void setStop(double stop);
        virtual void setLoop(bool loop);
        virtual void setSpatialize(bool spatialize);
        virtual void setPosition(float x, float y, float z);
        virtual void setVelocity(float vx, float vy, float vz);
        virtual void setAudio(const Audio *audio);
        virtual void play();
        virtual void play(double start);
        virtual void stop();
        virtual bool isPlaying();
        virtual int update(const Player *genericPlayer = 0, char *buf = 0, int bufsiz = 0)
        {
            (void)genericPlayer;
            (void)buf;
            (void)bufsiz;
            return -1;
        }
        virtual void stopForRestart() { }
        virtual void restart() { }

    protected:
        const Audio *audio;
        float pitch = 1.f;
        float intensity = 0.f;
        float startTime = 0.f;
        float stopTime = 0.f;
        bool loop = false;
        bool spatialize = true;
        bool playing = false;
        glm::vec3 x = glm::vec3(0, 0, 0);
        glm::vec3 v = glm::vec3(0, 0, 0);
    };

    Player(const Listener *listener);

    virtual void update();

    // Source related
    virtual Source *newSource(const Audio *);

    static Player *createPlayer(Listener *listener, const std::string &type);

protected:
    const Listener *listener;
};
} // namespace
#endif
