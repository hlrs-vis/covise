/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_
#define _PLAYER_

#include "vrmlexport.h"
#include "Audio.h"

#include <vector>

#include "Listener.h"

namespace vrml
{

class VRMLEXPORT Player
{
public:
    class Source
    {
        friend class Player;
        friend class PlayerMix;
        friend class PlayerOpenAL;
        friend class PlayerAServer;

    public:
        Source(const Audio *audio);
        virtual ~Source();
        virtual void setPitch(float pitch);
        virtual void setIntensity(float intensity);
        virtual void setStart(double start);
        virtual void setStop(double stop);
        virtual void setMute(bool mute);
        virtual void setLoop(bool mute);
        virtual void setSpatialize(bool spatialize);
        virtual void setPosition(float x, float y, float z);
        virtual void setPositionOC(float x, float y, float z);
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
        virtual void stopForRestart()
        {
        }
        virtual void restart()
        {
        }

    protected:
        const Audio *audio;
        float pitch;
        float intensity;
        float startTime, stopTime;
        bool mute, loop, spatialize, playing;
        vec x;
        vec v;
        Player *player;
        int handle;
    };

    Player(const Listener *listener);
    virtual ~Player();
    virtual bool isPlayerMix()
    {
        return false;
    }

    virtual void update()
    {
    }

    // mm/second
    virtual void setSpeedOfSound(float speed = 343000.0);
    virtual void setEAXEnvironment(int /*environment*/)
    {
    }

    // Listener related
    virtual vec getListenerPositionWC() const;
    virtual vec getListenerPositionVC() const;
    virtual vec getListenerPositionOC() const;
    virtual void getListenerOrientation(vec *at, vec *up) const;
    virtual vec getListenerVelocity() const;

    virtual vec WCtoVC(vec p) const;
    virtual vec WCtoOC(vec p) const;
    virtual vec VCtoWC(vec p) const;
    virtual vec VCtoOC(vec p) const;
    virtual vec OCtoWC(vec p) const;
    virtual vec OCtoVC(vec p) const;

    // Source related
    virtual Source *newSource(const Audio *);
    virtual void removeSource(int handle);

    virtual void restart(){};

protected:
    virtual int checkHandle(int handle) const;
    const Listener *listener;
    float speedOfSound;
    unsigned numSources;

    std::vector<Source *> sources;
    virtual int addSource(Source *src);
    virtual float calculateDoppler(const Source *src) const;
};
}
#endif
