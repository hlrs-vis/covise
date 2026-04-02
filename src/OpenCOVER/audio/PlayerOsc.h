/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLAYER_OSC_
#define _PLAYER_OSC_

#include "Player.h"

#include <memory>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <util/coExport.h>

#include <OpenConfig/access.h>
#include <OpenConfig/section.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <string>

namespace opencover::audio
{

class Listener;

class COVEREXPORT PlayerOsc : public Player
{
public:
    PlayerOsc(const Listener *listener);
    // ~PlayerOsc();

    virtual std::unique_ptr<Player::Source> makeSource(const Audio *audio);

    void connect();

    /**
     * Read the plugin configuration and transmit the information to the audio
     * server. This specifically includes the speaker setup.
     */
    void transmitConfiguration();

protected:
    void write(const char *buf, size_t len);

    class Source : public Player::Source
    {
    public:
        Source(Player *player, const Audio *audio);
        virtual ~Source();
        virtual void setAudio(const Audio *audio) override;
        virtual void play(double start) override;
        virtual void play() override;
        virtual void stop() override;
        virtual void setLoop(bool loop) override;

        virtual void update(const Player *player) override;

        std::string uuid;
    };

    boost::uuids::random_generator uuid_generator;

    opencover::config::Access access;
    std::unique_ptr<opencover::config::Section> config;

    covise::Host socket_host;
    std::unique_ptr<covise::Socket> socket;
};
} // namespace
#endif
