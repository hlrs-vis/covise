/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AL/al.h"
#include "AL/alut.h"

#include "Audio.h"
#include <util/coErr.h>
#include <iostream>

using namespace opencover::audio;

Audio::Audio() { }

Audio::Audio(const std::string &url)
    : Audio()
{
    setURL(url);
}

Audio::~Audio()
{
    unload();
}

bool Audio::setURL(const std::string &url)
{
    _url = url;

    loadFile();
    loadFileToBuffer();

    return true;
}

void Audio::loadFile()
{
    if (_sample_data != nullptr)
    {
        // Already loaded.
        return;
    }

    if (!_alut.is_initialized)
    {
        std::cerr << "Audio: ALUT not loaded" << std::endl;
        return;
    }

    ALenum format;
    ALsizei size;
    ALfloat frequency;
    ALvoid *sample = alutLoadMemoryFromFile(_url.c_str(), &format, &size, &frequency);
    if (sample == nullptr)
    {
        return;
    }

    switch (format)
    {
    case AL_FORMAT_MONO8:
        _channels = 1;
        _bits_per_sample = 8;
        break;
    case AL_FORMAT_MONO16:
        _channels = 1;
        _bits_per_sample = 16;
        break;
    case AL_FORMAT_STEREO8:
        _channels = 2;
        _bits_per_sample = 8;
        break;
    case AL_FORMAT_STEREO16:
        _channels = 2;
        _bits_per_sample = 16;
        break;
    }

    _samples_per_sec = (int)frequency;
    _num_samples = (int)size * 8 / _channels / _bits_per_sample;
}

void Audio::loadFileToBuffer()
{
    if (_buffer != AL_NONE)
    {
        // Already loaded.
        return;
    }

    if (!_alut.has_context)
    {
        std::cerr << "Audio: no ALUT context available to load audio file to buffer" << std::endl;
        return;
    }

    if (!_alut.is_initialized)
    {
        std::cerr << "Audio: ALUT not loaded" << std::endl;
        return;
    }

    _buffer = alutCreateBufferFromFile(_url.c_str());
    if (_buffer == AL_NONE)
    {
        ALenum error = alutGetError();
        std::cerr << "Audio: Error creating buffer: " << alutGetErrorString(error) << std::endl;
    }

    ALsizei size;

    alGetBufferi(_buffer, AL_FREQUENCY, &_samples_per_sec);
    alGetBufferi(_buffer, AL_BITS, &_bits_per_sample);
    alGetBufferi(_buffer, AL_CHANNELS, &_channels);
    alGetBufferi(_buffer, AL_SIZE, &size);

    _num_samples = (int)size * 8 / _channels / _bits_per_sample;
}

void Audio::unload()
{

    if (_sample_data)
    {
        delete[] _sample_data;
        _sample_data = nullptr;
    }

    if (_buffer != AL_NONE)
    {
        alDeleteBuffers(1, &_buffer);
        _buffer = AL_NONE;
    }
}
