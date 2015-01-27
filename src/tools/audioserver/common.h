/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//////////////////////////////////////////////////////////////////////////////
//
// common.h
//
// Copyright Microsoft Corporation. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <windows.h>
#include <dmusics.h>
//#include <dmusici.h>
#include "loader.h"
#include <stdio.h>

// extern void InitPosition(HWND hDlg);
// extern LPCTSTR FindMediaDirectory();

#define MSG_LEN 3000
#define MAX_LOG_MSG 1000

#define NUM_COLORS 19

// Macro to release COM interface if needed.
//
#ifndef RELEASE
#define RELEASE(x)          \
    {                       \
        if (x)              \
            (x)->Release(); \
        x = NULL;           \
    }
#endif

// Return the number of elements in an array.
//
#ifndef _countof
#define _countof(x) (sizeof(x) / sizeof(x[0]))
#endif

extern HWND hWndMain;
extern HWND hWndStatusBar;
extern HWND hWndStatusList;
extern HWND hWndLogList;
extern HWND hWndProgressBar;
extern HWND hWndGridView;
extern HWND hWnd3DView;
extern BOOL enableLogList; // log messages
extern BOOL enableStatusList; // status list
extern BOOL enable3DView; // 3D view
extern long numLogListItems; // number of items currently present in list

extern C_DMLoader *g_Loader;
extern IDirectMusicPerformance8 *g_pPerformance;
extern IDirectMusic8 *g_pDirectMusic8;
extern IDirectMusic *g_pDirectMusic;
extern IReferenceClock *g_pReferenceClock;

extern PALETTEENTRY paletteEntries[];

extern FILE *logfile;

extern HANDLE g_hNotify;

long GetNextColor(void);
void SetStatusMsg(char *msg);
void AddLogMsg(char *msg);
void SetProgress(long value, long range);

//void updateGrid( float x, float y, float z );
//void updateGridColored( float x, float y, float z, long color );

//////////////////////////////////////////////////////////////////////////////
//
// MillisecToRefTime
//
// Given a delta time in milliseconds, convert it to reference time, which
// is in 100 nanosecond units.
//
inline REFERENCE_TIME MillisecToRefTime(DWORD ms)
{
    // Reference time is in 100 ns units
    //
    return ms * (REFERENCE_TIME)(10 * 1000);
}

inline MUSIC_TIME TimeToMusicTime(double time)
{
    DWORD ticks = DMUS_PPQ;
    return (MUSIC_TIME)(time * 2 * ticks);
}
