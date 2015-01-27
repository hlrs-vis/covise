/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 *********************************************************************
 *  @file   : as_control.h
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
#if !defined(AS_CONTROL_H__)
#define AS_CONTROL_H__

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "as_cache.h"
#include "as_Sound.h"
#include <dmusici.h>

#define MAX_SOUNDS 256

class as_Control
{
    typedef struct
    {
        long handle;
        as_Sound *sound;
        long priority;
        //		long color;
        char filename[_MAX_PATH];
    } asHandle;

public:
    char *GetSoundNameByHandle(long handle);
    as_Sound *GetSoundBySegment(void *pSegment);
    void Panic();
    long getHandleColor(long handle);
    int playFile(char *filename);
    long newHandle(char *filename);
    void releaseHandle(long handle);
    int test(long);
    as_Control();
    virtual ~as_Control();

    long getNumberOfActiveHandles();
    as_Sound *getSoundByHandle(long handle);
    long getHandleBySound(as_Sound *sound);
    void setVolume(unsigned long volumeLeft, unsigned long volumeRight);

private:
    asHandle handles[MAX_SOUNDS];
    long numActiveHandles;
};

extern as_Control *AS_Control;
#endif // !defined(as_Control_H__)
