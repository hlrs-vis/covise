/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PlayerOsc.h"
#include "net/covise_host.h"
#include <fcntl.h>
#include <memory>
#include <sys/types.h>
#include "Audio.h"

#include <boost/uuid/uuid_io.hpp>

#include <oscpp/client.hpp>

#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <OpenConfig/file.h>

using namespace opencover::audio;

#define MAX_BUFLEN 1024

PlayerOsc::PlayerOsc(const Listener *listener)
    : Player(listener)
{
    config = access.file("plugin/audio");

    std::string host = config->value<std::string>("osc.connection", "host", "localhost")->value();
    int64_t port = config->value<int64_t>("osc.connection", "port", 8000)->value();

    socket_host = covise::Host(host.c_str());
    socket = std::make_unique<covise::Socket>(&socket_host, port, 2);

    connect();
}

void PlayerOsc::connect()
{

    transmitConfiguration();

    /*
    char buffer[MAX_BUFLEN];

    OSCPP::Client::Packet packet(buffer, sizeof(buffer));
    packet.openMessage("/subscribe", 0).closeMessage();
    write(buffer, packet.size());
    */
}

void PlayerOsc::write(const char *buf, size_t len)
{
    if (socket->isConnected())
    {
        socket->write(buf, len);
    }
    else
    {
        // std::cout << "No write, not connected." << std::endl;
    }

    /*
    while (socket->available())
    {
        char buffer[MAX_BUFLEN];
        if (socket->read(buffer, sizeof(buffer)) < 0)
        {
            break;
        }

        std::cout << "Read from socket: " << buffer << std::endl;
    }
    */
}

void PlayerOsc::transmitConfiguration()
{
    auto speakers = config->array<opencover::config::Section>("", "speakers");

    for (size_t i = 0; i < speakers->size(); i++)
    {
        opencover::config::Section speaker = (*speakers)[i];

        std::cout << "The section has these entries:" << std::endl;
        for (auto e : speaker.entries(""))
        {
            std::cout << " - Entry: " << e << std::endl;
        }

        std::string name = speaker.value<std::string>("", "name")->value();
        std::cout << "Entry 'name' value: " << name << std::endl;
    }
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
    return new Source(audio, this);
}
