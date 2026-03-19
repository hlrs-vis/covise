/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Listener.h"
#include "PlayerOpenAL.h"

using namespace opencover::audio;

#if defined(__APPLE__)
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#elif defined(_MSC_VER)
#include <al.h>
#include <alc.h>
#else
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>
#endif

#include <AL/alut.h>
#include <iostream>
#include <glm/gtx/string_cast.hpp>

PlayerOpenAL::PlayerOpenAL(const Listener *listener)
    : Player(listener)
{
    alDistanceModel(AL_NONE);
    alDopplerVelocity(speedOfSound);
}

void PlayerOpenAL::setSpeedOfSound(float speed)
{
    Player::setSpeedOfSound(speed);

    alDopplerVelocity(speed);
}

void PlayerOpenAL::update()
{
    Player::update();

    if (listener)
    {
        glm::vec3 x = listener->getPosition();
        alListener3f(AL_POSITION, x.x, x.y, x.z);
        std::cout << "Position " << glm::to_string(x) << std::endl;

        glm::vec3 v = listener->getVelocity();
        alListener3f(AL_VELOCITY, v.x, v.y, v.z);
        std::cout << "Velocity " << glm::to_string(v) << std::endl;

        glm::vec3 up, at;
        listener->getOrientation(&at, &up);
        std::cout << "Orientation " << glm::to_string(at) << " , " << glm::to_string(up) << std::endl;
        float orientation[6] = { at.x, at.y, at.z, up.x, up.y, up.z };
        alListenerfv(AL_ORIENTATION, orientation);
    }
    else
    {
        float orientation[6] = { 0, 1, 0, 0, 0, 1 };

        alListener3f(AL_POSITION, 0, 0, 0);
        alListener3f(AL_VELOCITY, 0, 0, 0);
        alListenerfv(AL_ORIENTATION, orientation);
    }
}

Player::Source *
PlayerOpenAL::newSource(const Audio *audio)
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

PlayerOpenAL::Source::Source(const Audio *audio)
    : Player::Source(audio)
{
    alGenSources(1, &alSource);
    alSource3f(alSource, AL_POSITION, x.x, x.y, x.z);
    alSource3f(alSource, AL_VELOCITY, v.x, v.y, v.z);
    alSourcef(alSource, AL_MIN_GAIN, 0.0);
    alSourcef(alSource, AL_MAX_GAIN, 1.0);
    alSourcef(alSource, AL_GAIN, intensity);

    // audio->loadFileToBuffer(); // ensure a buffer is created for this file
    // TODO: Audio* is const :(
    //
}

PlayerOpenAL::Source::~Source()
{
    stop();

    alDeleteSources(1, &alSource);
}

void PlayerOpenAL::Source::setIntensity(float intensity)
{
    Player::Source::setIntensity(intensity);

    alSourcef(alSource, AL_GAIN, this->intensity);
}

void PlayerOpenAL::Source::setPosition(float x, float y, float z)
{
    Player::Source::setPosition(x, y, z);

    if (spatialize)
        alSource3f(alSource, AL_POSITION, x, y, z);
    else
        alSource3f(alSource, AL_POSITION, 0, 0, 0);
}

void PlayerOpenAL::Source::setVelocity(float vx, float vy, float vz)
{
    Player::Source::setVelocity(vx, vy, vz);

    alSource3f(alSource, AL_VELOCITY, vx, vy, vz);
}

void PlayerOpenAL::Source::setPitch(float pitch)
{
    Player::Source::setPitch(pitch);

    alSourcef(alSource, AL_PITCH, pitch);
}

void PlayerOpenAL::Source::setMute(bool mute)
{
    Player::Source::setMute(mute);

    if (mute)
    {
        alSourcef(alSource, AL_GAIN, 0.0);
    }
    else
    {
        alSourcef(alSource, AL_GAIN, intensity);
    }
}

void PlayerOpenAL::Source::setSpatialize(bool spatialize)
{
    Player::Source::setSpatialize(spatialize);

    if (spatialize)
    {
        alSourcef(alSource, AL_SOURCE_RELATIVE, AL_FALSE);
        alSource3f(alSource, AL_POSITION, x.x, x.y, x.z);
    }
    else
    {
        alSourcef(alSource, AL_SOURCE_RELATIVE, AL_TRUE);
        alSource3f(alSource, AL_POSITION, 0, 0, 0);
    }
}

void PlayerOpenAL::Source::setLoop(bool loop)
{
    Player::Source::setLoop(loop);

    if (loop)
    {
        alSourcei(alSource, AL_LOOPING, AL_TRUE);
    }
    else
    {
        alSourcei(alSource, AL_LOOPING, AL_FALSE);
    }
}

void PlayerOpenAL::Source::play(double start)
{
    Player::Source::play(start);

    if (start > 0)
    {
        if (loop)
        {
            start = fmod(start, audio->duration());
        }
        alSourcef(alSource, AL_SEC_OFFSET, start);
    }

    ALuint buf = audio->buffer();
    alSourceQueueBuffers(alSource, 1, &buf);
    if (loop)
    {
        alSourcei(alSource, AL_LOOPING, AL_TRUE);
    }

    alSourcePlay(alSource);
}

void PlayerOpenAL::Source::stop()
{
    alSourceStop(alSource);

    Player::Source::stop();
}
