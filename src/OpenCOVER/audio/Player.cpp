/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "Player.h"
#include <assert.h>
#include <boost/algorithm/string.hpp>

#include <config/CoviseConfig.h>

#include "audio/PlayerAServer.h"
#include "audio/PlayerOpenAL.h"
#include "audio/PlayerOsc.h"

using std::endl;
using namespace opencover::audio;
using covise::coCoviseConfig;

Player::Source::Source(const Audio *audio)
    : audio(audio)
    , pitch(1.0)
    , intensity(0.0)
    , startTime(0.0)
    , stopTime(0.0)
    , mute(false)
    , loop(false)
    , spatialize(true)
    , playing(false)
    , x(0.0, 0.0, 0.0)
    , v(0.0, 0.0, 0.0)
    , player(0)
    , handle(-1)
{
}

Player::Source::~Source()
{
    if (player)
        player->removeSource(handle);
}

void Player::Source::setAudio(const Audio *audio)
{
    this->audio = audio;
}

void Player::Source::setPitch(float pitch)
{
    this->pitch = pitch;
}

void Player::Source::setPosition(float x, float y, float z)
{
    this->x = glm::vec3(x, y, z);
}

void Player::Source::setVelocity(float vx, float vy, float vz)
{
    this->v = glm::vec3(vx, vy, vz);
}

void Player::Source::setIntensity(float intensity)
{
    this->intensity = intensity;
}

void Player::Source::setStart(double start)
{
    this->startTime = (float)start;
}

void Player::Source::setStop(double stop)
{
    this->stopTime = (float)stop;
}

void Player::Source::setLoop(bool loop)
{
    this->loop = loop;
}

void Player::Source::setMute(bool mute)
{
    this->mute = mute;
}

void Player::Source::setSpatialize(bool spatialize)
{
    this->spatialize = spatialize;
}

void Player::Source::play()
{
    play(startTime);
}

void Player::Source::play(double start)
{
    (void)start;
    playing = true;
}

void Player::Source::stop()
{
    playing = false;
}

bool Player::Source::isPlaying()
{
    return playing;
}

Player::Player(const Listener *listener)
    : listener(listener)
    , speedOfSound(343000.0)
    , numSources(0)
    , sources()
{
}

Player::~Player()
{
    for (std::vector<Source *>::iterator it = sources.begin();
        it != sources.end(); it++)
    {
        delete *it;
        *it = 0;
    }
    sources.resize(0);
}

int Player::addSource(Source *src)
{
    int handle = -1;

    if (sources.size() == numSources)
    {
        handle = (int)sources.size();
        sources.resize(handle + 1);
    }
    else
    {
        for (unsigned i = 0; i < sources.size(); i++)
        {
            if (0 == sources[i])
            {
                handle = i;
                break;
            }
        }
    }
    assert(handle != -1);

    sources[handle] = src;
    numSources++;

    src->handle = handle;
    src->player = this;

    return handle;
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

int Player::checkHandle(int handle) const
{
    if (handle < 0 || handle >= (int)sources.size() || 0 == sources[handle])
    {
        std::cerr << "handle(" << handle << ") out of range" << endl;
        return -1;
    }

    return handle;
}

Player::Source *
Player::newSource(const Audio *audio)
{
    Source *src = new Source(audio);
    int handle = addSource(src);
    if (-1 == handle)
    {
        delete src;
        src = 0;
    }

    return src;
}

void Player::removeSource(int handle)
{
    if (checkHandle(handle) < 0)
        return;

    sources[handle] = 0;
    if ((int)sources.size() == handle - 1)
        sources.resize(handle);

    numSources--;
}

void Player::setSpeedOfSound(float speed)
{
    speedOfSound = speed;
}
