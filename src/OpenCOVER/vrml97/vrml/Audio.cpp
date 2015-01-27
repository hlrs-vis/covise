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
#include "config.h"

#include "Audio.h"
#include "Doc.h"
#include "System.h"
#include <util/coErr.h>

#include <iostream>
using std::cerr;
using std::endl;

#if defined(HAVE_AFL)
#include <audiofile.h>
#endif
#if defined(HAVE_AL)
#include <dmedia/audiofile.h>

#include <audio.h>
#include <getopt.h>
#include <malloc.h>

#define FORCE_RATE 1
#define SET_RATE 2
#endif
#include <string.h>

using namespace vrml;

/*=========================================================================
| TYPES
 ========================================================================*/
typedef unsigned char byte;
typedef unsigned short two_bytes;
typedef unsigned int four_bytes;

/*=========================================================================
| TYPES
 ========================================================================*/
enum AudioFileType
{
    AudioFile_UNKNOWN,
    AudioFile_WAV
};

/*=========================================================================
| TYPES
 ========================================================================*/

#define WAVE_FORMAT_PCM 1

/*=========================================================================
| audioFileType
|
|--------------------------------------------------------------------------
| Determine the audio file type
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     1. URL string
|     2. File handle
|
| RETURNS
|     AudioFileType
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
// FILE * is included in case this function is to be updated to
// peek at the file header.  - ks 11Nov98
static AudioFileType audioFileType(const char *url, FILE *)
{
#if defined(HAVE_AL) || defined(HAVE_AFL)
    (void)url;
    return AudioFile_UNKNOWN;
#else
    char *suffix = strrchr((char *)url, '.');
    if (suffix)
        ++suffix;

    if (strcmp(suffix, "wav") == 0 || strcmp(suffix, "WAV") == 0)
        return AudioFile_WAV;

    else
        return AudioFile_UNKNOWN;
#endif
}

/*=========================================================================
| PUBLIC METHODS
 ========================================================================*/

/*=========================================================================
| Audio::Audio
| Audio::~Audio
|
|--------------------------------------------------------------------------
| CONSTRUCTOR
| DESTRUCTOR
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     1. URL string
|     2. Doc object
|
| RETURNS
|     None
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
Audio::Audio(const char *url, Doc *relative)
    : _doc(0)
    , _last_modified(System::the->time())
    , _encoding(AUDIO_LINEAR)
    , _channels(0)
    , _bits_per_sample(0)
    , _samples_per_sec(0)
    , _sample_blocksize(0)
    , _num_samples(0)
    , _samples(0)
{
    setURL(url, relative);
}

Audio::~Audio()
{
    delete _doc;
    delete[] _samples;
}

double
Audio::lastModified() const
{
    return _last_modified;
}

/*=========================================================================
| Audio::setURL
|
|--------------------------------------------------------------------------
| Set the URL of the audio file and read it from the document object.
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     1. URL string
|     2. Doc object
|
| RETURNS
|     True if the URL was read, false if it was not
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
bool Audio::setURL(const char *url, Doc *relative)
{
    if (url == 0)
        return false;

    _last_modified = System::the->time();

    delete _doc;
    _doc = new Doc(url, relative);
    FILE *fp = _doc->fopen("rb");

    bool success = false;
    if (fp)
    {
        switch (audioFileType(url, fp))
        {
        case AudioFile_WAV:
#if defined(HAVE_AL) || defined(HAVE_AFL)
            success = alread(fp);
#else
            success = wavread(fp);
#endif
            break;

        default:
#if defined(HAVE_AL) || defined(HAVE_AFL)
            success = alread(fp);
#else
            fprintf(stderr, "Error: unrecognized audio file format (%s).\n", url);

            // Suppress the error message below
            success = true;
#endif
            break;
        }

        if (success == false)
        {
            fprintf(stderr, "Error: unable to read audio file (%s).\n", url);
        }
        else // Audio file library already closed the file, a duplicate close causes a segfault on windows
        {
            _doc->fclose();
        }
    }
    else
    {
        fprintf(stderr, "Error: unable to find audio file (%s).\n", url);
    }
    return (_num_samples > 0);
}

/*=========================================================================
| Audio::tryURLs
|
|--------------------------------------------------------------------------
| Try a list of URLs
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     1. Number of URLs to try
|     2. List of URLs
|     3. Document object
|
| RETURNS
|     True if one of the URLs succeeded, false if they all failed
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
bool Audio::tryURLs(int nUrls, const char *const *urls, Doc *relative)
{
    int i;

    for (i = 0; i < nUrls; ++i)
        if (setURL(urls[i], relative))
            return true;

    return false;
}

/*=========================================================================
| Audio::url
|
|--------------------------------------------------------------------------
| Return the url of this clip
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     None
|
| RETURNS
|     URL if one exists
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
const char *Audio::url() const
{
    return (_doc ? _doc->url() : 0);
}

#ifndef BYTESWAP
inline void swap_byte(unsigned int &byte) // only if necessary
{
    byte = ((byte & 0x000000ff) << 24) | ((byte & 0x0000ff00) << 8) | ((byte & 0x00ff0000) >> 8) | ((byte & 0xff000000) >> 24);
}

// only if necessary
inline void swap_bytes(unsigned int *bytes, int no)
{
    for (int i = 0; i < no; i++, bytes++)
        *bytes = ((*bytes & 0x000000ff) << 24) | ((*bytes & 0x0000ff00) << 8) | ((*bytes & 0x00ff0000) >> 8) | ((*bytes & 0xff000000) >> 24);
}

// only if necessary
inline void swap_short_bytes(unsigned short *bytes, int no)
{
    for (int i = 0; i < no; i++, bytes++)
        *bytes = (((*bytes & 0x00ff) << 8) | ((*bytes & 0xff00) >> 8));
}

inline void swap_short_byte(unsigned short &byte) // only if necessary
{
    byte = ((byte & 0x00ff) << 8) | ((byte & 0xff00) >> 8);
}
#endif

#if defined(HAVE_AL) || defined(HAVE_AFL)
bool Audio::alread(FILE *fp)
{
    LOGINFO("Audio::alread() - called!");
    AFfilehandle af;
    af = afOpenFD(fileno(fp), "r", 0);
    if (af == AF_NULL_FILEHANDLE)
    {
        LOGERROR("afOpenFD() - failed!");
        return false;
    }
    afSetVirtualSampleFormat(af, AF_DEFAULT_TRACK, AF_SAMPFMT_TWOSCOMP, 16);
    _channels = afGetChannels(af, AF_DEFAULT_TRACK);

    _encoding = AUDIO_LINEAR;
    _bits_per_sample = (int)afGetVirtualFrameSize(af, AF_DEFAULT_TRACK, 1) * 8 / _channels;
    _samples_per_sec = (int)afGetRate(af, AF_DEFAULT_TRACK);
    _num_samples = afGetFrameCount(af, AF_DEFAULT_TRACK);
    _sample_blocksize = (int)afGetVirtualFrameSize(af, AF_DEFAULT_TRACK, 1);

    // LOGINFO("Audio::alread(): chan=%d, bps=%d, samp/sec=%d, numframes=%d, sampbsize=%d", _channels, _bits_per_sample, _samples_per_sec, _num_samples, _sample_blocksize);

    /*
   fprintf(stderr, "Audio::alread(): chan=%d, bps=%d, samp/sec=%d, numframes=%d, sampbsize=%d\n",
           _channels, _bits_per_sample, _samples_per_sec, _num_samples, _sample_blocksize);
    */

    // Allocate the memory required
    delete[] _samples;
    _samples = new unsigned char[_sample_blocksize * (_num_samples + 1)];
    if (_samples == 0)
        return false;
    if (afReadFrames(af, AF_DEFAULT_TRACK, _samples, _num_samples) != _num_samples)
    {
        cerr << "Audio::alread(): did not read " << _num_samples << " frames as expected" << endl;
        LOGERROR("Audio::alread(): did not read %d frames as expected!", _num_samples);
    }
    // for making resampling easier
    for (int i = 0; i < _sample_blocksize; i++)
    {
        _samples[_sample_blocksize * _num_samples + i] = _samples[i];
    }
    return true;
}
#endif
/*=========================================================================
| PRIVATE METHODS
 ========================================================================*/

/*=========================================================================
| Audio::wavread
|
|--------------------------------------------------------------------------
| Read a WAV file
|
|--------------------------------------------------------------------------
| ARGUMENTS
|     1. File handle
|
| RETURNS
|     True if the read succeeded, false if not
|
|--------------------------------------------------------------------------
| REVISION HISTORY:
| Rev     Date      Who         Description
| 0.8     11Nov98   kumaran     Created
========================================================================*/
bool Audio::wavread(FILE *fp)
{
    LOGINFO("Audio::wavread() - called!");
    WaveHeader wave_header;

    rewind(fp);
    size_t retval = 0;
    retval += fread(&wave_header.riff_id, 1, 4, fp);
    retval += fread(&wave_header.riff_size, 4, 1, fp);
    retval += fread(&wave_header.wave_id, 1, 4, fp);
    retval += fread(&wave_header.format_id, 1, 4, fp);
    retval += fread(&wave_header.format_size, 4, 1, fp);
    retval += fread(&wave_header.format_tag, 2, 1, fp);
    retval += fread(&wave_header.num_channels, 2, 1, fp);
    retval += fread(&wave_header.num_samples_per_sec, 4, 1, fp);
    retval += fread(&wave_header.num_avg_bytes_per_sec, 4, 1, fp);
    retval += fread(&wave_header.num_block_align, 2, 1, fp);
    retval += fread(&wave_header.bits_per_sample, 2, 1, fp);
    retval += fread(&wave_header.data_id, 1, 4, fp);
    retval += fread(&wave_header.num_data_bytes, 4, 1, fp);

    if (retval != 25)
    {
        std::cerr << "Audio::wavread: fread failed" << std::endl;
        return false;
    }
    //rewind (fp);

    // Do all sorts of sanity checks
    if (strncmp((const char *)wave_header.riff_id, "RIFF", 4) != 0)
    {
        cerr << "got " << (const char *)wave_header.riff_id << " expected RIFF" << endl;
        return false;
    }

    if (strncmp((const char *)wave_header.wave_id, "WAVE", 4) != 0)
    {
        cerr << "got " << (const char *)wave_header.wave_id << " expected WAVE" << endl;
        return false;
    }

    if (strncmp((const char *)wave_header.format_id, "fmt ", 4) != 0)
    {
        cerr << "got " << (const char *)wave_header.format_id << " expected fmt" << endl;
        return false;
    }

    if (strncmp((const char *)wave_header.data_id, "data", 4) != 0
        && strncmp((const char *)wave_header.data_id, "PAD ", 4) != 0)
    {
        cerr << "got " << (const char *)wave_header.data_id << " expected data" << endl;
        return false;
    }

#ifndef BYTESWAP
    swap_short_byte(wave_header.format_tag);
    swap_short_byte(wave_header.num_channels);
    swap_short_byte(wave_header.num_block_align);
    swap_short_byte(wave_header.bits_per_sample);
    swap_byte(wave_header.num_data_bytes);
    swap_byte(wave_header.num_avg_bytes_per_sec);
    swap_byte(wave_header.num_samples_per_sec);
    swap_byte(wave_header.format_size);
    swap_byte(wave_header.riff_size);
#endif
    if (wave_header.format_tag != WAVE_FORMAT_PCM)
    {
        cerr << "got " << wave_header.format_tag << "as format_tag expected WAVE_FORMAT_PCM \n\n\n ask Uwe, he knows what to do \n\n\n" << endl;
        // das war auskommentiert
        return false;
    }

    // Allocate the memory required
    delete[] _samples;
    _samples = new unsigned char[wave_header.num_data_bytes + 1];
    if (_samples == 0)
    {
        cerr << "did not get enough Memory, requested " << wave_header.num_data_bytes << " bytes" << endl;
        return false;
    }

    // Now, we are ready to read the data
    //fseek (fp, 44, SEEK_SET);
    int bytes_read = (int)fread(_samples, 1, wave_header.num_data_bytes, fp);
    // for easy resampling
    _samples[wave_header.num_data_bytes] = _samples[0];

    _encoding = AUDIO_LINEAR;
    _channels = wave_header.num_channels;
    _bits_per_sample = wave_header.bits_per_sample;
    _samples_per_sec = wave_header.num_samples_per_sec;
    _sample_blocksize = wave_header.num_block_align;
    _num_samples = bytes_read / _sample_blocksize;
#ifndef BYTESWAP
    if (wave_header.bits_per_sample == 16)
        swap_short_bytes((unsigned short *)_samples, wave_header.num_data_bytes / 2);
#endif
    return true;
}

void
Audio::createWaveHeader(WaveHeader *header) const
{
    memcpy(header->riff_id, "RIFF", 4);
    header->riff_size = 36 + _num_samples * _sample_blocksize;
    memcpy(header->wave_id, "WAVE", 4);
    memcpy(header->format_id, "fmt ", 4);
    header->format_size = 16;
    header->format_tag = WAVE_FORMAT_PCM;
    header->num_channels = _channels;
    header->num_samples_per_sec = _samples_per_sec;
    header->num_avg_bytes_per_sec = _samples_per_sec * _sample_blocksize;
    header->num_block_align = _sample_blocksize;
    header->bits_per_sample = _bits_per_sample;
    memcpy(header->data_id, "data", 4);
    header->num_data_bytes = _num_samples * _sample_blocksize;

#ifndef BYTESWAP
    swap_byte(header->riff_size);
    swap_byte(header->format_size);
    swap_short_byte(header->format_tag);
    swap_short_byte(header->num_channels);
    swap_byte(header->num_samples_per_sec);
    swap_byte(header->num_avg_bytes_per_sec);
    swap_short_byte(header->num_block_align);
    swap_short_byte(header->bits_per_sample);
    swap_byte(header->num_data_bytes);
#endif
}
