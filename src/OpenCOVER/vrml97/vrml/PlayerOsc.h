/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_OSC_
#define _PLAYER_OSC_

#include "Player.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <string>

namespace vrml
{

class Listener;

class VRMLEXPORT PlayerOsc : public Player
{
public:
    PlayerOsc(const Listener *listener, const std::string &host, int port);
    // virtual ~PlayerOsc();
    virtual Player::Source *newSource(const Audio *audio);

    void connect();

    virtual void update();

protected:
    void write(const char *buf, size_t len);

    class Source : public Player::Source
    {
    public:
        Source(const Audio *audio, PlayerOsc *player);
        virtual ~Source();
        virtual void setAudio(const Audio *audio);
        virtual void play(double start);
        virtual void play();
        virtual void stop();
        virtual void restart();

        // virtual void setMute(bool mute);
        virtual void setLoop(bool loop);

        virtual int update(const Player *player, char *buf = 0, int numFrames = 0);

        std::string uuid;
        PlayerOsc *player;
    };

    boost::uuids::random_generator uuid_generator;

    covise::Host socket_host;
    covise::Socket socket;
};
} // namespace vrml
#endif
