/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SoundSweep.h"
#include <iostream>
#include <strstream>
#include <QMessageBox>
#include "mainWindow.h"

SoundSweep::SoundSweep(std::string baseName, int minVal, int maxVal, int vs)
{
    int i = minVal;
    valStep = vs;
    while (i <= maxVal)
    {
        std::strstream str;
        str << baseName;
        str << i;
        str << ".wav";
        std::string filename;
        str >> filename;
        SoundSample *ss = new SoundSample(filename, i);
        if (ss->getSamples() == NULL)
        {
            QMessageBox::critical(theWindow, "Failed to read",
                                  filename.c_str(),
                                  QMessageBox::Ok,
                                  QMessageBox::Ok);
            return;
        }
        sounds.push_back(ss);
        i += valStep;
    }
}

SoundSweep::~SoundSweep()
{
    for (std::list<SoundSample *>::iterator iter = sounds.begin(); iter != sounds.end();)
    {
        delete *iter;
    }
}

void SoundSweep::getSamples(SoundSample **samples, float speed)
{
    samples[0] = *(sounds.begin());
    samples[1] = NULL;
    samples[2] = NULL;
    for (std::list<SoundSample *>::iterator iter = sounds.begin(); iter != sounds.end(); iter++)
    {
        if ((*iter)->getSpeed() + (valStep / 2.0) >= speed)
        {
            samples[1] = *iter;
            if (iter != sounds.begin())
            {
                std::list<SoundSample *>::iterator it;
                it = iter;
                it--;
                samples[0] = *(it);
            }
            iter++;
            if (iter != sounds.end())
            {
                samples[2] = *iter;
            }
            return;
        }
    }
    std::list<SoundSample *>::iterator it = sounds.end();
    it--;
    samples[1] = *it;
    samples[2] = *it;
}

void SoundSweep::setPitchVolumeScale(int soundIndex, float pitchOffset, float VolumeOffset)
{
    int i = 0;
    for (std::list<SoundSample *>::iterator iter = sounds.begin(); iter != sounds.end(); iter++)
    {
        if (i == soundIndex)
        {
            (*iter)->setLowPitchVolumeScale(pitchOffset, VolumeOffset);
            if (iter != sounds.begin())
            {
                iter--;
                (*iter)->setHighPitchVolumeScale(pitchOffset, VolumeOffset);
            }
            break;
        }
        i++;
    }
}