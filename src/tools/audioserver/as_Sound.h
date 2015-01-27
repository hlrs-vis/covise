/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_Sound.h
 *
 *  Project : AudioServer
 *
 *  Package : AudioServer prototype
 *
 *  Author  : Marc Schreier                              Date: 05/05/2002
 *
 *  Purpose : Header file
 *
 *********************************************************************
 */

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#if !defined(AS_SOUND_H__)
#define AS_SOUND_H__

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <dmusici.h>
#include <dsound.h>

#define STATUS_NOTUSED 0
#define STATUS_INITIAL 1
#define STATUS_PLAYING 2
#define STATUS_LOOPING 3
#define STATUS_STOPPED 4

class as_Sound
{
private:
    IDirectMusicAudioPath8 *pAudioPath;
    IDirectMusicSegment8 *pSegment;
    IDirectMusicSegmentState *pState;
    IDirectSound3DBuffer *pDS3DBuffer;
    DMUS_OBJECTDESC dmod;
    float Position[3];
    float Direction[3];
    bool spatialized;
    bool playing;
    long status;
    bool looping;
    long starttime;
    long stoptime;
    long repeats;
    float gain;

    long oldGridPosX;
    long oldGridPosY;
    long color;

public:
    unsigned long GetColor(void);
    void SetColor(unsigned long color);
    void updateGridColored(void);
    void CommitDeferredSetings(void);
    void SetDirectionRelative(float x, float y, float z, long color);
    void SetDirectionRelative(float angle, long color);
    bool IsPlaying(void);
    void SetStatus(long newstatus);
    void *GetSegment(void);
    void GetDirection(D3DVECTOR *direction);
    void GetPosition(D3DVECTOR *position);
    long GetStatus(void);
    void SetVelocity(float x, float y, float z);
    void SetVelocity(float velocity);
    void SetLooping(bool enable);
    void SetDirectionRelative(float x, float y, float z);
    void SetDirectionRelative(float angle);
    void SetDirection(float angle);
    void SetDirection(float x, float y, float z);
    void SetDopplerFrequency(float freq);
    void SetGain(float gain);
    void SetPitch(float pitch);
    void SetPosition(float x, float y, float z);
    void PlayLooping(void);
    void Play(float starttime, float stoptime);
    void Play(float starttime);
    void Stop(void);
    void Play(void);
    as_Sound(char *fileName);
    ~as_Sound(void);
};
#endif
