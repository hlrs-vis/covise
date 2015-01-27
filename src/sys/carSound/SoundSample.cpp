/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "SoundSample.h"
#include <iostream>
#include <strstream>
#include <QMessageBox>
#include "mainWindow.h"

SoundSample::SoundSample(std::string name, int rev)
{
    uMin = rev;
    samples = NULL;

#ifdef HAVE_AUDIOFILE

    lowPitchOffset = highPitchOffset = 1.0;
    lowVolumeOffset = highVolumeOffset = 1.0;
    lastDir = 0;
    lastValue = 0;
    file = afOpenFile(name.c_str(), "r", 0);
    if (file == AF_NULL_FILEHANDLE)
    {
        QMessageBox::critical(theWindow, "Error",
                              "SoundSample::afOpenFD() - failed!",
                              QMessageBox::Ok,
                              QMessageBox::Ok);
    }
    else
    {
        afSetVirtualSampleFormat(file, AF_DEFAULT_TRACK, AF_SAMPFMT_TWOSCOMP, 16);
        //afSetVirtualSampleFormat(file, AF_DEFAULT_TRACK, AF_SAMPFMT_FLOAT, 32);

        channels = afGetChannels(file, AF_DEFAULT_TRACK);
        samplesPerSec = (int)afGetRate(file, AF_DEFAULT_TRACK);
        numSamples = afGetFrameCount(file, AF_DEFAULT_TRACK);
        sampleBlocksize = (int)afGetVirtualFrameSize(file, AF_DEFAULT_TRACK, 1);
        bytesPerSample = (int)afGetVirtualFrameSize(file, AF_DEFAULT_TRACK, 1) / channels;
        if (channels != 2)
        {
            std::strstream buf;
            buf << "SoundSample::not a stereo sound file" << name;
            std::string s;
            buf >> s;
            QMessageBox::critical(theWindow, "Error",
                                  s.c_str(),
                                  QMessageBox::Ok,
                                  QMessageBox::Ok);
        }
        if (samplesPerSec != 48000)
        {
            std::strstream buf;
            buf << "SoundSample::48kHz required but " << name << " has " << samplesPerSec;
            std::string s;
            buf >> s;
            QMessageBox::critical(theWindow, "Error",
                                  s.c_str(),
                                  QMessageBox::Ok,
                                  QMessageBox::Ok);
        }
        samples = new short[numSamples * channels];
        if (afReadFrames(file, AF_DEFAULT_TRACK, samples, numSamples) != numSamples)
        {
            std::strstream buf;
            buf << "SoundSample::alread(): did not read " << numSamples << " frames as expected";
            std::string s;
            buf >> s;
            QMessageBox::critical(theWindow, "Error",
                                  s.c_str(),
                                  QMessageBox::Ok,
                                  QMessageBox::Ok);
        }
        afCloseFile(file);

        currPos = 0;
        skipPos = 0;
        skipSamples = 0;
        skipPos = 1;
    }
#endif
}

SoundSample::~SoundSample()
{
    delete[] samples;
}

void SoundSample::getNextSamples(short &left, short &right)
{
    if (skipSamples > 0 && (currPos == (int)skipPos || skipPos == 0.0))
    {
        skipPos += skipSamples;
        if (skipPos < currPos)
            skipPos = currPos + skipSamples;
        if (skipPos > (numSamples - 1))
        {
            skipPos -= (numSamples - 1);
        }
        if (faster)
            currPos++;
        else
            currPos--;
        if (currPos >= numSamples)
        {
            currPos = 0;
        }
    }
    left = samples[currPos * 2] * lowVolumeOffset;
    right = samples[(currPos * 2) + 1] * lowVolumeOffset;
    currPos++;
    if (currPos >= numSamples)
    {
        currPos = 0;
    }
    lastDir = lastValue < left ? 1 : 0;
    lastValue = left;
}

void SoundSample::shiftTo(SoundSample *ss)
{
    if (ss == NULL)
        return;
    int i;
    int shiftPos = currPos;
    short lastV = samples[shiftPos * 2];
    short currV;
    short minDist = 32000;
    int minDistPos = currPos;
    int minDistSkipPos = skipPos;
    int shiftSkipPos = skipPos;

    for (i = 0; i < 400; i++) // shift as much as 400 samples
    {
        shiftPos++;
        shiftSkipPos++;
        if (shiftPos >= numSamples)
        {
            shiftPos = 0;
        }
        if (shiftSkipPos >= numSamples)
        {
            shiftSkipPos = 0;
        }
        currV = samples[shiftPos * 2];
        if (lastV < currV) // going up;
        {
            if (ss->lastDir == 1 && (lastV < ss->lastValue && currV >= ss->lastValue))
            {
                currPos = shiftPos;
                skipPos = shiftSkipPos;
                return;
            }
        }
        else // going down
        {
            if (ss->lastDir == 1 && (lastV > ss->lastValue && currV <= ss->lastValue))
            {
                currPos = shiftPos;
                skipPos = shiftSkipPos;
                return;
            }
        }
        if (abs(ss->lastValue - currV) < minDist)
        {
            minDist = abs(ss->lastValue - currV);
            minDistPos = shiftPos;
            minDistSkipPos = shiftSkipPos;
        }
        lastV = currV;
    }
    currPos = minDistPos;
    skipPos = minDistSkipPos;
}

void SoundSample::setPitch(float pitch) // -0.5 - 0.5 half a step down/half a step up
{
    skipSamples = 0;
    if (pitch < 0)
    {
        faster = false;
        skipSamples = -20.0 / pitch * lowPitchOffset;
    }
    else if (pitch > 0)
    {
        faster = true;
        skipSamples = 20.0 / pitch * highPitchOffset;
    }
}

void SoundSample::setLowPitchVolumeScale(float po, float vo)
{
    lowPitchOffset = po;
    lowVolumeOffset = vo;
}

void SoundSample::setHighPitchVolumeScale(float po, float vo)
{
    highPitchOffset = po;
    highVolumeOffset = vo;
}
