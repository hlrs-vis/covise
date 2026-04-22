/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AUDIO_
#define _AUDIO_

#include "AlutContext.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <util/coExport.h>

// do not include al.h here, otherwise we need OpenAL dependency everywhere
typedef unsigned int ALuint;

namespace opencover::audio
{

class COVRAUDIOEXPORT Audio
{
public:
    Audio();
    Audio(const std::string &url);
    ~Audio();

    bool setURL(const std::string &url);
    const std::string url() const { return _url; }

    /**
     * Loads the referenced audio file into memory and parses the audio
     * metadata.
     */
    void loadFile();

    /**
     * Loads the referenced audio file into a buffer for OpenAL, and extracts
     * the audio metadata from that buffer.
     */
    void loadFileToBuffer();

    /**
     * Removes audio buffer data from memory, without removing metadata.
     */
    void unload();

    /**
     * Returns the number of channels of this audio file, 1 for mono or 2 for stereo.
     */
    int channels() const
    {
        return _channels;
    }

    /**
     * Returns the bit depth (8 or 16) of the audio file.
     */
    int bitsPerSample() const
    {
        return _bits_per_sample;
    }

    /**
     * Returns the sampling rate, i.e. number of samples per second (also
     * called frequency sometimes) of this audio file.
     */
    int samplesPerSec() const
    {
        return _samples_per_sec;
    }

    /**
     * Returns the number of samples in the audio file.
     */
    int numSamples() const
    {
        return _num_samples;
    }

    /**
     * Returns a pointer to the audio data buffer.
     */
    const unsigned char *samples() const
    {
        return _sample_data;
    }

    /**
     * Returns the AL buffer.
     */
    const ALuint buffer() const
    {
        return _buffer;
    }

    /**
     * Computes and returns the duration of this audio file in seconds.
     */
    double duration() const
    {
        if (_samples_per_sec <= 0)
        {
            return 0;
        }

        return (double)_num_samples / (double)_samples_per_sec;
    }

private:
    AlutContext _alut; ///< Automatically initializes alut once instantiated.

    std::string _url;

    int _channels = 0;
    int _bits_per_sample = 0;
    int _samples_per_sec = 0;

    // Samples are stored in aligned blocks.  Sometimes, the
    // block will be larger than the sample itself.  Usually,
    // however, an 8-bit sample will be in a 1-byte block and
    // a 16-bit sample will be in a 2-byte block.
    int _sample_blocksize = 0;

    int _num_samples = 0;

    unsigned char *_sample_data = nullptr;
    ALuint _buffer = 0;
};
}

#endif // _AUDIO_
