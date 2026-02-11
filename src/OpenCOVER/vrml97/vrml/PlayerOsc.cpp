/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PlayerOsc.h"
#include <fcntl.h>
#include <sys/types.h>
#include <vrml97/vrml/Audio.h>

#include <boost/uuid/uuid_io.hpp>

#include <oscpp/client.hpp>

using namespace vrml;

#define MAX_BUFLEN 1024

PlayerOsc::PlayerOsc(const Listener *listener, const std::string &host, int port)
    : Player(listener)
    , socket_host(host.c_str())
    , socket(&socket_host, port)
{
    connect();
}

void PlayerOsc::connect()
{
    /*
    char buffer[MAX_BUFLEN];

    OSCPP::Client::Packet packet(buffer, sizeof(buffer));
    packet.openMessage("/subscribe", 0).closeMessage();
    write(buffer, packet.size());
    */
}

void PlayerOsc::write(const char *buf, size_t len)
{
    if (socket.isConnected())
    {
        socket.write(buf, len);
    }
    else
    {
        // std::cout << "No write, not connected." << std::endl;
    }

    /*
    while (socket.available())
    {
        char buffer[MAX_BUFLEN];
        if (socket.read(buffer, sizeof(buffer)) < 0)
        {
            break;
        }

        std::cout << "Read from socket: " << buffer << std::endl;
    }
    */
}

PlayerOsc::Source::Source(const Audio *audio, PlayerOsc *player)
    : Player::Source(audio)
    , uuid(boost::uuids::to_string(player->uuid_generator()))
    , player(player)
{
    setAudio(audio);
}

void PlayerOsc::Source::setAudio(const Audio *audio)
{
    // TODO: send audio data or path
}

PlayerOsc::Source::~Source()
{
    stop();
    // TODO: send deletion
}

void PlayerOsc::Source::play(double start)
{
    // WRITE_MESSAGE("/source/%s/play", uuid.c_str());
    Player::Source::play(start);

    char buffer[MAX_BUFLEN];
    char addr[MAX_BUFLEN];
    snprintf(addr, MAX_BUFLEN, "/source/%s/play", uuid.c_str());

    OSCPP::Client::Packet packet(buffer, sizeof(buffer));
    packet.openMessage(addr, 0).closeMessage();
    player->write(buffer, packet.size());
}

void PlayerOsc::Source::play() { Player::Source::play(); }

void PlayerOsc::Source::stop()
{
    Player::Source::stop();
    // WRITE_MESSAGE("/source/%s/stop", uuid.c_str());
}

void PlayerOsc::Source::restart() { setAudio(audio); }

void PlayerOsc::update()
{
    Player::update();

    for (unsigned i = 0; i < sources.size(); i++)
    {
        if (sources[i])
        {
            sources[i]->update(this);
        }
    }
}

int PlayerOsc::Source::update(const Player *genericPlayer, char *buf, int bufsize)
{
    Player::Source::update(genericPlayer, buf, bufsize);

    if (!isPlaying())
    {
        return -1;
    }

    char buffer[MAX_BUFLEN];
    char addr[MAX_BUFLEN];
    snprintf(addr, MAX_BUFLEN, "/source/%s/position", uuid.c_str());

    OSCPP::Client::Packet packet(buffer, sizeof(buffer));
    packet.openMessage(addr, 3).float32(x.x).float32(x.y).float32(x.z).closeMessage();
    player->write(buffer, packet.size());

    return 0;
}

void PlayerOsc::Source::setLoop(bool loop)
{
    Player::Source::setLoop(loop);

    // TODO: send loop parameter
}

Player::Source *PlayerOsc::newSource(const Audio *audio)
{
    Source *src = new Source(audio, this);
    int handle = addSource(src);
    if (-1 == handle)
    {
        delete src;
        src = 0;
    }

    return src;
}
