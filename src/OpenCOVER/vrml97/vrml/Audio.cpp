/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//
//  Audio.cpp
//    contributed by Kumaran Santhanam

/*=========================================================================
| CONSTANTS
 ========================================================================*/

/*=========================================================================
| INCLUDES
 ========================================================================*/
#include "AL/al.h"
#include "alut.h"
#include "config.h"

#include "Audio.h"
#include "Doc.h"
#include "System.h"
#include <util/coErr.h>

using namespace vrml;

Audio::Audio(const char *url, Doc *relative)
    : _last_modified(System::the->time())
{
    setURL(url, relative);
}

Audio::~Audio()
{
    unload();
}

bool Audio::setURL(const char *url, Doc *relative)
{
    if (url == nullptr)
    {
        return false;
    }

    _last_modified = System::the->time();

    Doc doc(url, relative);
    FILE *fp = doc.fopen("rb");
    if (!fp)
    {
        return false;
    }
    _url = doc.url();
    doc.fclose();

    loadFile();
    loadFileToBuffer();

    return true;
}

bool Audio::tryURLs(int nUrls, const char *const *urls, Doc *relative)
{
    int i;

    for (i = 0; i < nUrls; ++i)
        if (setURL(urls[i], relative))
            return true;

    return false;
}

const char *Audio::url() const
{
    return _url.c_str();
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
        CERR << "ALUT not loaded" << std::endl;
        return;
    }

    ALenum format;
    ALsizei size;
    ALfloat frequency;
    ALvoid *sample = alutLoadMemoryFromFile(url(), &format, &size, &frequency);
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
        CERR << "no ALUT context available to load audio file to buffer" << std::endl;
        return;
    }

    if (!_alut.is_initialized)
    {
        CERR << "ALUT not loaded" << std::endl;
        return;
    }

    _buffer = alutCreateBufferFromFile(url());
    if (_buffer == AL_NONE)
    {
        ALenum error = alutGetError();
        CERR << "Error creating buffer: " << alutGetErrorString(error) << std::endl;
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
