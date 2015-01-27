/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//
//  Audio.h
//    contributed by Kumaran Santhanam

#ifndef _AUDIO_
#define _AUDIO_
//
//  Audio document class
//
#include "config.h"
#ifndef _MSC_VER
#include <inttypes.h>
#endif
#include <math.h>
#include <stdio.h>
#include "vrmlexport.h"

namespace vrml
{

class Doc;

enum AudioEncoding
{
    AUDIO_LINEAR,
    AUDIO_ULAW
};

class VRMLEXPORT Audio
{
public:
    Audio(const char *url = 0, Doc *relative = 0);
    ~Audio();

    bool setURL(const char *url, Doc *relative = 0);
    bool tryURLs(int nUrls, const char *const *urls, Doc *relative = 0);

    double lastModified() const;

    const char *url() const;

    AudioEncoding encoding() const
    {
        return _encoding;
    }
    int channels() const
    {
        return _channels;
    }
    int bitsPerSample() const
    {
        return _bits_per_sample;
    }
    int samplesPerSec() const
    {
        return _samples_per_sec;
    }
    int sampleBlockSize() const
    {
        return _sample_blocksize;
    }
    int numSamples() const
    {
        return _num_samples;
    }

    int numBytes() const
    {
        return _num_samples * _sample_blocksize;
    }
    const unsigned char *samples() const
    {
        return _samples;
    }

    double duration() const
    {
        if (_samples_per_sec > 0)
            return (double)_num_samples / (double)_samples_per_sec;
        else
            return 0;
    }

    // Get the sample index given a floating point time index
    // If the time index is greater than the duration, the sample
    // index is wrapped back to the beginning of the sample.
    // From: Alex Funk <Alexander.Funk@nord-com.net>
    // Avoid int overflow when multiplying time_index by samples_per_sec
    // Modified to use fmod() by Kumaran Santhanam.
    int getByteIndex(double time_index) const
    {
        if (_num_samples > 0 && _samples_per_sec > 0)
            return _sample_blocksize * (int)(fmod(time_index, duration()) * (double)_samples_per_sec);
        else
            return -1;
    }

    struct WaveHeader
    {
        uint8_t riff_id[4];
        uint32_t riff_size;
        uint8_t wave_id[4];
        uint8_t format_id[4];
        uint32_t format_size;
        uint16_t format_tag;
        uint16_t num_channels;
        uint32_t num_samples_per_sec;
        uint32_t num_avg_bytes_per_sec;
        uint16_t num_block_align;
        uint16_t bits_per_sample;
        uint8_t data_id[4];
        uint32_t num_data_bytes;
    };
    void createWaveHeader(WaveHeader *header) const;

private:
    Doc *_doc;

    double _last_modified;

    AudioEncoding _encoding;
    int _channels;
    int _bits_per_sample;
    int _samples_per_sec;

    // Samples are stored in aligned blocks.  Sometimes, the
    // block will be larger than the sample itself.  Usually,
    // however, an 8-bit sample will be in a 1-byte block and
    // a 16-bit sample will be in a 2-byte block.
    int _sample_blocksize;

    int _num_samples;
    unsigned char *_samples;

    bool wavread(FILE *fp);
#if defined(HAVE_AL) || defined(HAVE_AFL)
    bool alread(FILE *fp);
#endif
};
}

#endif // _AUDIO_
