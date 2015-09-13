/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "Player.h"
#include <assert.h>

using std::endl;
using namespace vrml;

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
    //CERR << "new source" << endl;
}

Player::Source::~Source()
{
    if (player)
        player->removeSource(handle);
}

void
Player::Source::setAudio(const Audio *audio)
{
    this->audio = audio;
}

void
Player::Source::setPitch(float pitch)
{
    this->pitch = pitch;
}

void
Player::Source::setPosition(float x, float y, float z)
{
    this->x = vec(x, y, z);
}

void
Player::Source::setPositionOC(float x, float y, float z)
{
    if (player)
    {
        vec oc = vec(x, y, z);
        vec wc = player->OCtoWC(oc);

        setPosition(wc.x, wc.y, wc.z);
    }
    else
    {
        setPosition(0.0, 0.0, 0.0);
    }
}

void
Player::Source::setVelocity(float vx, float vy, float vz)
{
    this->v = vec(vx, vy, vz);
}

void
Player::Source::setIntensity(float intensity)
{
    this->intensity = intensity;
}

void
Player::Source::setStart(double start)
{
    this->startTime = (float)start;
}

void
Player::Source::setStop(double stop)
{
    this->stopTime = (float)stop;
}

void
Player::Source::setLoop(bool loop)
{
    this->loop = loop;
}

void
Player::Source::setMute(bool mute)
{
    this->mute = mute;
}

void
Player::Source::setSpatialize(bool spatialize)
{
    this->spatialize = spatialize;
}

void
Player::Source::play()
{
    //CERR << "Player::Source::play()" << endl;
    play(startTime);
}

void
Player::Source::play(double start)
{
    //CERR << "Player::Source::play(double)" << endl;
    (void)start;
    playing = true;
}

void
Player::Source::stop()
{
    playing = false;
}

bool
Player::Source::isPlaying()
{
    return playing;
}

Player::Player(const Listener *listener)
    : listener(listener)
    , speedOfSound(343000.0)
    , // unit is mm
    numSources(0)
    , sources()
{
    if (!listener)
        CERR << "listener is NULL !!!" << endl;
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

vec
Player::getListenerPositionWC() const
{
    if (listener)
        return listener->getPositionWC();
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::getListenerPositionVC() const
{
    if (listener)
        return listener->getPositionVC();
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::getListenerPositionOC() const
{
    if (listener)
        return listener->getPositionOC();
    else
        return vec(0.0, 0.0, 0.0);
}

void
Player::getListenerOrientation(vec *at, vec *up) const
{
    if (listener)
        listener->getOrientation(at, up);
}

vec
Player::getListenerVelocity() const
{
    if (listener)
        return listener->getVelocity();
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::WCtoVC(vec p) const
{
    if (listener)
        return listener->WCtoVC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::WCtoOC(vec p) const
{
    if (listener)
        return listener->WCtoOC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::VCtoWC(vec p) const
{
    if (listener)
        return listener->VCtoWC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::VCtoOC(vec p) const
{
    if (listener)
        return listener->VCtoOC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::OCtoWC(vec p) const
{
    if (listener)
        return listener->OCtoWC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

vec
Player::OCtoVC(vec p) const
{
    if (listener)
        return listener->OCtoVC(p);
    else
        return vec(0.0, 0.0, 0.0);
}

int
Player::addSource(Source *src)
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

    //CERR << "numsources: " << numSources << endl;

    return handle;
}

int
Player::checkHandle(int handle) const
{
    if (handle < 0 || handle >= (int)sources.size() || 0 == sources[handle])
    {
        CERR << "handle(" << handle << ") out of range" << endl;
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

void
Player::removeSource(int handle)
{
    if (checkHandle(handle) < 0)
        return;

    sources[handle] = 0;
    if ((int)sources.size() == handle - 1)
        sources.resize(handle);

    numSources--;

    //CERR << "numsources: " << numSources << endl;
}

void
Player::setSpeedOfSound(float speed)
{
    speedOfSound = speed;
}

float
Player::calculateDoppler(const Source *src) const
{
    vec vc = OCtoVC(src->x);
    vec rel = vc.sub(getListenerPositionVC());

    float pitch = 1.0;

    if (rel.length() > 0.0)
    {
        float vl = rel.normalize().dot(getListenerVelocity());
        float vs = rel.normalize().dot(src->v);
        pitch = (speedOfSound - vl) / (speedOfSound + vs);
    }

    return pitch;
}
