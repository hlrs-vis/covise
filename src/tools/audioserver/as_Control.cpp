/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_control.cpp
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                           Date: 05/05/2002
 *
 *  Purpose : Non-GUI control
 *
 *********************************************************************
 */

#include "as_Control.h"

#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////
// Konstruktion/Destruktion
//////////////////////////////////////////////////////////////////////

as_Control::as_Control()
{

    this->numActiveHandles = 0;

    for (long i = 0; i < MAX_SOUNDS; i++)
    {
        this->handles[i].handle = -1;
        this->handles[i].sound = NULL;
        //		this->handles[i].color = 0x000000FF;
    }
    srand((unsigned)time(NULL));

    AddLogMsg("AudioSystem initialised.");
}

as_Control::~as_Control()
{
    for (long i = 0; i < MAX_SOUNDS; i++)
    {
        if (NULL != this->handles[i].sound)
            delete (this->handles[i].sound);
    }
}

as_Sound *as_Control::getSoundByHandle(long handle)
{
    if ((-1 == handle) || (MAX_SOUNDS < handle))
    {
        //		AddLogMsg("getSoundByHandle: invalid handle!");
        return NULL;
    }
    if (NULL == this->handles[handle].sound)
    {
        //		AddLogMsg("getSoundByHandle: no sound assigned!");
        return NULL;
    }
    return this->handles[handle].sound;
}

long as_Control::getHandleBySound(as_Sound *sound)
{
    long i;

    // no valid sound
    if (NULL == sound)
    {
        AddLogMsg("getHandleBySound: invalid sound!");
        return -1;
    }

    for (i = 0; i < MAX_SOUNDS; i++)
    {
        if (sound == (as_Sound *)(handles[i].sound))
        {
            // sound found, return handle
            return i;
        }
    }
    // no valid handle
    AddLogMsg("getHandleBySound: invalid handle!");
    return -1;
}

long as_Control::getNumberOfActiveHandles()
{
    return this->numActiveHandles;
}

long as_Control::newHandle(char *filename)
{
    as_Sound *psound = NULL;

    char msg[MSG_LEN];
    long handle = -1;
    long colorRGB;
    char *cacheFilename;

    // test for maximum number of sounds
    if (MAX_SOUNDS == this->numActiveHandles)
    {
        AddLogMsg("newHandle: Maximum usable number of sounds reached!");
        return -1;
    }

    // test if file already exists in defaults
    cacheFilename = AS_Cache->fileExistsDefault(filename);
    if (NULL == cacheFilename)
    {
#ifdef DEBUG
        sprintf(msg, "newHandle: File does not exist in cache (default sounds): %s", filename);
        AddLogMsg(msg);
#endif
        // test if file already exists in cache
        cacheFilename = AS_Cache->fileExists(filename);
        if (NULL == cacheFilename)
        {
            sprintf(msg, "newHandle: File does not exist in cache: %s", filename);
            AddLogMsg(msg);
            return -1;
        }
    }

    // create new sound
    psound = new as_Sound(cacheFilename);
    if (!psound)
    {
        AddLogMsg("newHandle: Unable to initialize new sound");
        return -1;
    }

    // search free entry
    for (handle = 0; handle < MAX_SOUNDS; handle++)
    {
        if (-1 == this->handles[handle].handle)
            break;
    }
    if ((MAX_SOUNDS == handle) && (-1 == this->handles[handle].handle))
    {
        sprintf(msg, "newHandle: handle = %d exceeds maximum number!", handle);
        AddLogMsg(msg);
        return -1;
    }

    // set sound position display color from list of predefined colors
    colorRGB = GetNextColor();

    // store handle data
    this->handles[handle].handle = handle;
    this->handles[handle].sound = psound;
    this->handles[handle].sound->SetColor(colorRGB);
    sprintf(this->handles[handle].filename, "%s\0", filename);

    this->numActiveHandles++;

#ifdef DEBUG
    sprintf(msg, "newHandle: handle = %d, color = %08lXh", handle, colorRGB);
    AddLogMsg(msg);
#endif

    // return current handle number
    return handle;
}

void as_Control::releaseHandle(long handle)
{
    delete (this->handles[handle].sound);
    this->handles[handle].sound = NULL;
    this->handles[handle].handle = -1;
    this->numActiveHandles--;

#ifdef DEBUG
    char msg[MSG_LEN];
    sprintf(msg, "releaseHandle %d", handle);
    AddLogMsg(msg);
#endif
}

void as_Control::setVolume(unsigned long volumeLeft, unsigned long volumeRight)
{
    char msg[MSG_LEN];
    // set volume of wave out device

    // ??? general volume and volume of other devices ???

    // currently using waveOutSetVolume for each Wave Out Device

    long volume = 0;
    int numDevs;
    HWAVEOUT hWaveOutDev;
    int i;
    int rc;
    WAVEFORMATEX wfex;
    WAVEOUTCAPS woc;

    if (0 > volumeLeft)
        volumeLeft = 0;
    if (65535 < volumeLeft)
        volumeLeft = 65535;
    if (0 > volumeRight)
        volumeRight = 0;
    if (65535 < volumeRight)
        volumeRight = 65535;
    volume = volumeRight + (volumeLeft << 16);

#ifdef DEBUG
    sprintf(msg, "cmdSetVolume, volume = %04lX, Left = %04lXL, Right = %04lX",
            (unsigned long)volume,
            (unsigned long)volumeLeft,
            (unsigned long)volumeRight);
    AddLogMsg(msg);
#endif

    // get number of devices
    numDevs = waveOutGetNumDevs();
    if (0 < numDevs)
    {
        for (i = 0; i < numDevs; i++)
        {

            rc = waveOutGetDevCaps(i, &woc, sizeof(WAVEOUTCAPS));
            if (MMSYSERR_NOERROR != rc)
            {
                sprintf(msg, "waveOutGetDevCaps error %d", rc);
                AddLogMsg(msg);
                return;
            }

            if (!(woc.dwSupport & WAVECAPS_VOLUME))
            {
                AddLogMsg("Device does not support volume change!");
                continue;
            }

            if ((woc.dwSupport & WAVECAPS_LRVOLUME))
            {
                AddLogMsg("Device does support L&R volume change!");
            }

            rc = waveOutOpen(&hWaveOutDev, i, &wfex, 0, 0, CALLBACK_NULL);

            rc = waveOutSetVolume(hWaveOutDev, volume);

            rc = waveOutClose(hWaveOutDev);
        }
    }
}

int as_Control::test(long paramVal)
{
    // play the different signals assigned to msgbox symbols

    char msg[MSG_LEN];

#ifdef DEBUG
    sprintf(msg, "Test cmd, param = %d", paramVal);
    AddLogMsg(msg);
#endif

    switch (paramVal)
    {
    default:
        sprintf(msg, "Test: unknown parameter %d!", paramVal);
        AddLogMsg(msg);
    case 0:
        // standard bell + speaker beep
        AddLogMsg("Test: standard bell + speaker beep");
        AddLogMsg(msg);
        if (0 > MessageBeep(MB_OK))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
        Beep(440, 100);
        Beep(660, 100);
        Beep(880, 100);
        break;

    case 1:
        // standard bell
        AddLogMsg("Test: standard bell");
        if (0 > MessageBeep(MB_OK))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
        break;

    case 2:
        // information
        AddLogMsg("Test: information sound");
        if (0 > MessageBeep(MB_ICONASTERISK))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
        break;

    case 3:
        // warning
        AddLogMsg("Test: warning sound");
        if (0 > MessageBeep(MB_ICONEXCLAMATION))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
        break;

    case 4:
        // critical
        AddLogMsg("Test: critical sound");
        if (0 > MessageBeep(MB_ICONHAND))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
        break;

    case 5:
        // question
        AddLogMsg("Test: question sound");
        if (0 > MessageBeep(MB_ICONQUESTION))
        {
            sprintf(msg, "MessageBeep failed, error %d", GetLastError());
            AddLogMsg(msg);
        }
    }
    return 0;
}

int as_Control::playFile(char *filename)
{
    // play file
    FILE *file;
    char msg[_MAX_PATH];

    if ((NULL == filename) || ("" == filename))
    {
        AddLogMsg("playFile error: empty filename!");
        return -1;
    }

    sprintf(msg, "playFile(%s)", filename);
    AddLogMsg(msg);

    file = fopen(filename, "r");
    if (NULL == file)
    {
        sprintf(msg, "playFile error: file '%s' not found!", filename);
        AddLogMsg(msg);
        return -1;
    }
    fclose(file);
    if (FALSE == PlaySound(filename, NULL, SND_ASYNC | SND_FILENAME))
    {
        sprintf(msg, "playFile error: Could not play sound '%s'", filename);
        AddLogMsg(msg);
        return -1;
    }

    return 0;
}

long as_Control::getHandleColor(long handle)
{
    return this->handles[handle].sound->GetColor();
}

void as_Control::Panic()
{
    as_Sound *psound = NULL;
    long i;
    long numHandles;

    //AddLogMsg("AudioSystem: Panic");

    numHandles = this->numActiveHandles;

    for (i = 0; i < numHandles; i++)
    {

        psound = this->getSoundByHandle(i);
        if (NULL != psound)
        {
            psound->Stop();
            releaseHandle(i);
        }
    }
}

as_Sound *as_Control::GetSoundBySegment(void *pSegment)
{
    long i;
    as_Sound *pSound;

    if (NULL == pSegment)
    {
        AddLogMsg("GetSoundBySegment: invalid segment!");
        return NULL;
    }
    for (i = 0; i < this->getNumberOfActiveHandles(); i++)
    {
        pSound = this->handles[i].sound;
        if (NULL == pSound)
            continue;
        if (pSegment == pSound->GetSegment())
        {
            return this->handles[i].sound;
        }
    }
    return NULL;
}

char *as_Control::GetSoundNameByHandle(long handle)
{
    return this->handles[handle].filename;
}
