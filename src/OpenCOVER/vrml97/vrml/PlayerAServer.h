/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_ASERVER_
#define _PLAYER_ASERVER_

#include "Player.h"

namespace vrml
{

class Listener;

class VRMLEXPORT PlayerAServer : public Player
{
public:
    PlayerAServer(const Listener *listener, const std::string &host, int port);
    virtual ~PlayerAServer();
    virtual Player::Source *newSource(const Audio *audio);
    virtual void setEAXEnvironment(int environment);

    virtual void update();

    virtual int send_cmd(const char *cmd) const;
    virtual int send_data(const char *data, int size, bool swapped = false) const;
    virtual int read_answer(char *buf, int maxsize) const;

    virtual void restart();

protected:
    void connect();
    class Source : public Player::Source
    {
    public:
        Source(const Audio *audio, PlayerAServer *player);
        virtual ~Source();
        virtual void setAudio(const Audio *audio);
        virtual void play(double start);
        virtual void play();
        virtual void stop();
        virtual void stopForRestart();
        virtual void restart();

        //virtual void setMute(bool mute);
        virtual void setLoop(bool loop);

        virtual int update(const Player *player, char *buf = 0, int numFrames = 0);

        int asHandle;
        float odirection;
        float ointensity;
        float opitch;
        vec ov;
        PlayerAServer *player;

    private:
        virtual void loadAudio(const Audio *audio);
    };

    mutable int asFd;
    std::string asHost;
    int asPort;
};
}
#endif
