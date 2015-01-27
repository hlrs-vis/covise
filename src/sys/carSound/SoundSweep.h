/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SOUND_SWEEP_H
#define SOUND_SWEEP_H

#include "SoundSample.h"
#include <string>
#include <list>

class SoundSweep
{
public:
    SoundSweep(std::string baseName, int minVal, int maxVal, int valStep);
    ~SoundSweep();
    void getSamples(SoundSample **samples, float speed);
    void setPitchVolumeScale(int soundIndex, float pitchOffset, float VolumeOffset);
    int getValStep()
    {
        return valStep;
    };

private:
    std::list<SoundSample *> sounds;
    int valStep;
};

#endif