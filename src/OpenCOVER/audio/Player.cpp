/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "Player.h"
#include <assert.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include <config/CoviseConfig.h>

#include "audio/PlayerAServer.h"
#include "audio/PlayerOpenAL.h"
#include "audio/PlayerOsc.h"

using std::endl;
using namespace opencover::audio;
using covise::coCoviseConfig;

Source::Source(Player *player, const Audio *audio)
    : player(player)
    , audio(audio)
{
    player->registerSource(this);
}

Source::~Source()
{
    player->unregisterSource(this);
}

void Source::setAudio(const Audio *audio)
{
    this->audio = audio;
}

void Source::setPitch(float pitch)
{
    this->pitch = pitch;
}

void Source::setPosition(float x, float y, float z)
{
    this->x = glm::vec3(x, y, z);
}

void Source::setVelocity(float vx, float vy, float vz)
{
    this->v = glm::vec3(vx, vy, vz);
}

void Source::setIntensity(float intensity)
{
    this->intensity = intensity;
}

void Source::setStart(double start)
{
    this->startTime = (float)start;
}

void Source::setStop(double stop)
{
    this->stopTime = (float)stop;
}

void Source::setLoop(bool loop)
{
    this->loop = loop;
}

void Source::setSpatialize(bool spatialize)
{
    this->spatialize = spatialize;
}

void Source::play()
{
    play(startTime);
}

void Source::play(double start)
{
    (void)start;
    playing = true;
}

void Source::stop()
{
    playing = false;
}

bool Source::isPlaying()
{
    return playing;
}

Player::Player(const Listener *listener)
    : listener(listener)
{
}

void Player::update()
{
    for (auto &source : sources)
    {
        source->update(this);
    }
}

Player *Player::createPlayer(Listener *listener, const std::string &type)
{
    if (type.empty())
    {
        return nullptr;
    }

    if (boost::iequals(type, "aserver"))
    {
        // TODO: Remove legacy config, let PlayerAServer read the new audio
        // config file and parse it itself.
        std::string host = coCoviseConfig::getEntry("value", "COVER.Audio.Host", "localhost");
        int port = coCoviseConfig::getInt("port", "COVER.Audio.Host", 31231);

        return new PlayerAServer(listener, host, port);
    }
    else if (boost::iequals(type, "openal"))
    {
        return new PlayerOpenAL(listener);
    }
    else if (boost::iequals(type, "osc"))
    {

        return new PlayerOsc(listener);
    }
    else if (boost::iequals(type, "none"))
    {
        return nullptr;
    }

    std::cerr << "Player::createPlayer: unknown player type: " << type << endl;
    return nullptr;
}

std::unique_ptr<Source>
Player::makeSource(const Audio *audio)
{
    return std::make_unique<Source>(this, audio);
}

void Player::registerSource(Source *source)
{
    sources.insert(source);
}

void Player::unregisterSource(Source *source)
{

    sources.erase(source);
}
