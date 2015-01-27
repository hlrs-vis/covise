/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_MIX
#define _PLAYER_MIX

#include "Player.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#include <vector>

namespace vrml
{

class VRMLEXPORT PlayerMix : public Player
{
public:
    PlayerMix(const Listener *listener, bool threaded, int channels = 2, int rate = 44100, int bps = 16);
    virtual ~PlayerMix();
    virtual bool isPlayerMix()
    {
        return true;
    }

    virtual Player::Source *newSource(const Audio *audio);
    virtual void removeSource(int handle);

    virtual void update();
    virtual int writeFrames(const char *frames, int numFrames) const
    {
        (void)frames;
        (void)numFrames;
        return 0;
    }
    virtual int getQueued() const
    {
        return -1;
    }
    virtual int getWritable() const
    {
        return -1;
    }
    virtual int getPacketSize() const
    {
        return -1;
    }
    virtual double getDelay() const
    {
        return (double)getQueued() / (double)rate;
    }

    // Speaker related
    virtual void setSpeakerPosition(int speaker, float x = 0.0, float y = 0.0, float z = 0.0);
    virtual void setSpeakerSpatialize(int speaker, bool spatialize);
    virtual void setHeadphones(bool headphone);
    virtual void setDolbySurround(bool surround);
    virtual void setSeparation(float sep);

protected:
    virtual void realUpdate(double time);
    virtual void lockMutex() const;
    virtual void unlockMutex() const;
    bool threaded;
    int channels;
    int rate;
    int bps;
    int bytesPerFrame;
    bool headphones;
    bool surround;
    float separation;

    char *buf;
    int bufsiz;
    int startValid, endValid, numValid;
    double lastTime;
    void startThread();
    void stopThread();

    class Source : public Player::Source
    {
    public:
        Source(const Audio *audio);
        virtual ~Source();
        virtual int update(const Player *genericPlayer, char *buf, int bufsiz);
        virtual void play();
        virtual void play(double start);

    protected:
        double pos;
        std::vector<double> off;
    };
    friend int Source::update(const Player *genericPlayer, char *buf, int bufsiz);

    class Speaker
    {
    public:
        Speaker();
        virtual ~Speaker();
        virtual void setPosition(float x = 0.0, float y = 0.0, float z = 0.0);
        virtual void setSpatialize(bool spatialize);
        vec x;
        bool spatialize;
    };
    std::vector<Speaker *> speakers;

    int sleeptime;
    static bool output_failed;
    static bool threadStarted;
#ifdef HAVE_PTHREAD
    pthread_t mixThread;
    mutable pthread_mutex_t mixMutex;
    static void *audioThread(void *);
#endif
};
}
#endif
