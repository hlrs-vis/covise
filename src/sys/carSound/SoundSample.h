/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SOUND_SAMPLE_H
#define SOUND_SAMPLE_H

#include <string>
#ifdef HAVE_AUDIOFILE
#include "audiofile.h"
#endif

class SoundSample
{
public:
    SoundSample(std::string name, int rev);
    ~SoundSample();
    short *getSamples()
    {
        return samples;
    };
    int getNumSamples()
    {
        return numSamples;
    };
    int getSpeed()
    {
        return uMin;
    };
    void shiftTo(SoundSample *ss);

    void getNextSamples(short &left, short &right);
    void setPitch(float pitch); // -1.0 - 1.0 half a step down/half a step up

    void setLowPitchVolumeScale(float pitchOffset, float VolumeOffset);
    void setHighPitchVolumeScale(float pitchOffset, float VolumeOffset);
    void setLowVolumeScale(float volumeOffset)
    {
        lowVolumeOffset = volumeOffset;
    };
    void setLowPitchScale(float pitchOffset)
    {
        lowPitchOffset = pitchOffset;
    };
    void setHighPitchScale(float pitchOffset)
    {
        highPitchOffset = pitchOffset;
    };
    float getLowVolumeOffset()
    {
        return lowVolumeOffset;
    };
    float getLowPitchOffset()
    {
        return lowPitchOffset;
    };
    float getHighPitchOffset()
    {
        return highPitchOffset;
    };

    float skipSamples; // samples to advance to next skipPos

    int lastDir; // wave is going up 1 or down 0
    short lastValue;

private:
#ifdef HAVE_AUDIOFILE
    AFfilehandle file;
#endif
    int uMin;
    int channels;
    int samplesPerSec;
    int numSamples;
    int sampleBlocksize;
    int bytesPerSample;
    short *samples;

    bool faster; // pitch up or down
    int currPos; // current sample position
    float skipPos; // position where we last skipped or duplicated a sample

    float lowPitchOffset;
    float lowVolumeOffset;
    float highPitchOffset;
    float highVolumeOffset;
};

#endif