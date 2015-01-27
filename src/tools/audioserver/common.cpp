/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//////////////////////////////////////////////////////////////////////////////
//
// common.cpp
//
//
// Copyright (c) Microsoft Corporation. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "common.h"
#include "resource.h"
#include "COMMCTRL.H"
#include "loader.h"
#include "as_cache.h"
#include "as_comm.h"
#include "as_control.h"
#include "as_ServerSocket.h"

#include <stdio.h>
#include <tchar.h>
#include <atlbase.h>
#include <sys/timeb.h>
#include <time.h>

HWND hWndMain = NULL;
HWND m_hWnd3D = NULL;
HWND hWndStatusBar = NULL;
HWND hWndStatusList = NULL;
HWND hWndLogList = NULL;
HWND hWndProgressBar = NULL;
HWND hWndGridView = NULL;
HWND hWnd3DView = NULL;
BOOL enableLogList = false; // log messages
BOOL enableStatusList = false; // status list
BOOL enable3DView = false; // 3D view
long numLogListItems = 0; // number of items currently present in list
static long currentColorIndex = 0;
FILE *logfile;

C_DMLoader *g_Loader = NULL;
IDirectMusicPerformance8 *g_pPerformance = NULL;
IDirectMusic8 *g_pDirectMusic8 = NULL;
IDirectMusic *g_pDirectMusic = NULL;
IReferenceClock *g_pReferenceClock = NULL;

as_Comm *AS_Communication = NULL;
as_cache *AS_Cache = NULL;
as_Control *AS_Control = NULL;
as_ServerSocket *AS_Server = NULL;

// colors static
static PALETTEENTRY paletteEntries[NUM_COLORS] = {
    { 0xFF, 0, 0, 0 },
    { 0, 0xFF, 0, 0 },
    { 0xFF, 0xFF, 0, 0 },
    { 0, 0, 0xFF, 0 },
    { 0xFF, 0, 0xFF, 0 },
    { 0, 0xFF, 0xFF, 0 },
    { 0xFF, 0xFF, 0xFF, 0 },
    { 0x80, 0, 0, 0 },
    { 0, 0x80, 0, 0 },
    { 0x80, 0x80, 0, 0 },
    { 0, 0, 0x80, 0 },
    { 0x80, 0, 0x80, 0 },
    { 0, 0x80, 0x80, 0 },
    { 0xC0, 0xC0, 0xC0, 0 },
    { 192, 220, 192, 0 },
    { 166, 202, 240, 0 },
    { 255, 251, 240, 0 },
    { 160, 160, 164, 0 },
    { 0x80, 0x80, 0x80, 0 }
};

long GetNextColor(void)
{
    long color;

    currentColorIndex++;
    if (currentColorIndex > NUM_COLORS)
        currentColorIndex = 0;
    color = PALETTERGB(
        paletteEntries[currentColorIndex].peRed,
        paletteEntries[currentColorIndex].peGreen,
        paletteEntries[currentColorIndex].peBlue);

    return color;
}

void SetStatusMsg(char *msg)
{
    SendMessage(hWndStatusBar, SB_SETTEXT, 0, (LPARAM)msg);
}

void SetProgress(long value, long range)
{
    if (0 != range)
    {
        SendMessage(hWndProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, range));
    }
    SendMessage(hWndProgressBar, PBM_SETPOS, (WPARAM)value, 0);
}

void AddLogMsg(char *msg)
{

    /*
      struct _timeb timebuffer;
      char *timeline;

      if (NULL != logfile) {
         _ftime( &timebuffer );
         timeline = ctime( & ( timebuffer.time ) );
         fprintf(logfile, "[%.19s.%hu] %s\n", timeline, timebuffer.millitm, msg);
      }
   */
    if (enableLogList)
    {
        int rc;

        // add message
        rc = SendMessage(
            hWndLogList, // handle to dialog box
            LB_ADDSTRING, // message to send
            0, // first message parameter
            (LPARAM)(LPCTSTR)msg // second message parameter
            );
        numLogListItems++;

        // if count > MAX_LOG_MSG delete topmost message
        if (numLogListItems >= MAX_LOG_MSG)
        {
            // delete topmost message
            rc = SendMessage(
                hWndLogList, // handle to dialog box
                LB_DELETESTRING, // message to send
                0, // first message parameter
                0 // second message parameter
                );
            numLogListItems--;
        }

        // Set listbox index to most recent message
        rc = SendMessage(
            hWndLogList, // handle to dialog box
            LB_SETTOPINDEX, // message to send
            (WPARAM)numLogListItems - 1, // first message parameter
            0 // second message parameter
            );
    }
}

//////////////////////////////////////////////////////////////////////////////
//
// InitPosition
//
// Called from WM_INITDIALOG to place the dialog in the center of the screen
// or parent window.
//
void InitPosition(HWND hDlg)
{
    RECT rc;
    RECT rcParent;
    HWND hWndParent = GetParent(hDlg);

    GetWindowRect(hDlg, &rc);

    if (hWndParent == (HWND)NULL)
    {
        rcParent.left = 0;
        rcParent.right = GetSystemMetrics(SM_CXSCREEN);
        rcParent.top = 0;
        rcParent.bottom = GetSystemMetrics(SM_CYSCREEN);
    }
    else
    {
        GetWindowRect(hWndParent, &rcParent);
    }

    SetWindowPos(hDlg,
                 NULL,
                 rcParent.left + ((rcParent.right - rcParent.left) - (rc.right - rc.left)) / 2,
                 rcParent.top + ((rcParent.bottom - rcParent.top) - (rc.bottom - rc.top)) / 2,
                 0,
                 0,
                 SWP_NOZORDER | SWP_NOSIZE);
}

//////////////////////////////////////////////////////////////////////////////
//
// FindMediaDirectory
//
// Find the media directory by walking upwards towards the root
//
LPCTSTR FindMediaDirectory()
{
    static TCHAR tzMediaDir[MAX_PATH];

    TCHAR tzCurrDir[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, tzCurrDir);

    TCHAR tzSearch[MAX_PATH];

    for (;;)
    {
        _tmakepath(tzSearch, NULL, tzCurrDir, _T("media"), NULL);

        WIN32_FIND_DATA find;

        HANDLE h = FindFirstFile(tzSearch, &find);

        if (h != INVALID_HANDLE_VALUE)
        {
            FindClose(h);

            if (find.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            {
                _tcscpy(tzMediaDir, tzSearch);
                _tcscat(tzMediaDir, _T("\\"));
                return tzMediaDir;
            }
        }

        LPTSTR ptz = _tcsrchr(tzCurrDir, _T('\\'));
        if (ptz == NULL)
        {
            break;
        }

        *ptz = _T('\0');
    }

    _tcscpy(tzMediaDir, _T(".\\"));

    return tzMediaDir;
}

/**
 * Name: UpdateGrid()
 * Desc: Draws a red dot in the dialog's grid bitmap at the x,y coordinate.
 */
void updateGrid(float x, float y, float z)
{
    HWND hWndGrid;
    HDC hDC;
    RECT rc;
    static LONG s_lPixel[5] = { CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID };
    static LONG s_lX = 0;
    static LONG s_lY = 0;

    if (NULL == hWndGridView)
        return;

    hWndGrid = GetDlgItem(hWndGridView, IDC_GRID);
    hDC = GetDC(hWndGrid);

    // Don't update the grid if a WM_PAINT will be called soon
    BOOL bUpdateInProgress = GetUpdateRect(hWndGridView, NULL, FALSE);
    if (bUpdateInProgress)
    {
        return;
    }

    /************************************************************

      All pixels will be deleted when other windows overlay!!!

    ************************************************************/

    /*
       if( s_lPixel[0] != CLR_INVALID )
       {
           // Replace pixels from that were overdrawn last time
           SetPixel( hDC, s_lX-1, s_lY+0, s_lPixel[0] );
           SetPixel( hDC, s_lX+0, s_lY-1, s_lPixel[1] );
           SetPixel( hDC, s_lX+0, s_lY+0, s_lPixel[2] );
           SetPixel( hDC, s_lX+0, s_lY+1, s_lPixel[3] );
           SetPixel( hDC, s_lX+1, s_lY+0, s_lPixel[4] );
       }
   */

    // Draw a crosshair object in dark red pixels to reveal the trail
    SetPixel(hDC, s_lX - 1, s_lY + 0, 0x000000a0);
    SetPixel(hDC, s_lX + 0, s_lY - 1, 0x000000a0);
    SetPixel(hDC, s_lX + 0, s_lY + 0, 0x000000a0);
    SetPixel(hDC, s_lX + 0, s_lY + 1, 0x000000a0);
    SetPixel(hDC, s_lX + 1, s_lY + 0, 0x000000a0);

    // Convert the world space x,y coordinates to pixel coordinates
    GetClientRect(hWndGrid, &rc);

    s_lX = (LONG)(((rc.right - rc.left) / 2) + 50 * x);
    s_lY = (LONG)(((rc.bottom - rc.top) / 2) - 50 * z);

    // attention: x, z are horizontal, y is vertical axis !!!
    /*
       // Save the pixels before drawing the cross hair
       s_lPixel[0] = GetPixel( hDC, s_lX-1, s_lY+0 );
       s_lPixel[1] = GetPixel( hDC, s_lX+0, s_lY-1 );
       s_lPixel[2] = GetPixel( hDC, s_lX+0, s_lY+0 );
       s_lPixel[3] = GetPixel( hDC, s_lX+0, s_lY+1 );
       s_lPixel[4] = GetPixel( hDC, s_lX+1, s_lY+0 );
   */
    // Draw a crosshair object in light red pixels
    SetPixel(hDC, s_lX - 1, s_lY + 0, 0x000000ff);
    SetPixel(hDC, s_lX + 0, s_lY - 1, 0x000000ff);
    SetPixel(hDC, s_lX + 0, s_lY + 0, 0x000000ff);
    SetPixel(hDC, s_lX + 0, s_lY + 1, 0x000000ff);
    SetPixel(hDC, s_lX + 1, s_lY + 0, 0x000000ff);

    ReleaseDC(hWndGrid, hDC);
}

/**
 * Name: UpdateGridColored()
 * Desc: Draws a red dot in the dialog's grid bitmap at the x,y coordinate.
 *
void updateGridColored( float x, float y, float z, long color )
{
   HWND hWndGrid;
   HDC  hDC;
    RECT rc;
    static LONG s_lPixel[5] = { CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID, CLR_INVALID };
    static LONG s_lX = 0;
static LONG s_lY = 0;
long darkColor;

if (NULL == hWndGridView) return;

hWndGrid = GetDlgItem( hWndGridView, IDC_GRID );
hDC      = GetDC( hWndGrid );

// Don't update the grid if a WM_PAINT will be called soon
BOOL bUpdateInProgress = GetUpdateRect(hWndGridView,NULL,FALSE);
if( bUpdateInProgress ) {
return;
}

// All pixels will be deleted when other windows overlay!!!

#if 0
if( s_lPixel[0] != CLR_INVALID )
{
// Replace pixels from that were overdrawn last time
SetPixel( hDC, s_lX-1, s_lY+0, s_lPixel[0] );
SetPixel( hDC, s_lX+0, s_lY-1, s_lPixel[1] );
SetPixel( hDC, s_lX+0, s_lY+0, s_lPixel[2] );
SetPixel( hDC, s_lX+0, s_lY+1, s_lPixel[3] );
SetPixel( hDC, s_lX+1, s_lY+0, s_lPixel[4] );
}
#endif

darkColor = RGB(GetRValue(color)*0.5, GetGValue(color)*0.5, GetBValue(color)*0.5);

// Draw a crosshair object darker than normal the trail
SetPixel( hDC, s_lX-1, s_lY+0, darkColor );
SetPixel( hDC, s_lX+0, s_lY-1, darkColor );
SetPixel( hDC, s_lX+0, s_lY+0, darkColor );
SetPixel( hDC, s_lX+0, s_lY+1, darkColor );
SetPixel( hDC, s_lX+1, s_lY+0, darkColor );

// Convert the world space x,y coordinates to pixel coordinates
GetClientRect( hWndGrid, &rc );

s_lX = (LONG)(  (( rc.right - rc.left ) / 2 ) + 50*x );
s_lY = (LONG)(  (( rc.bottom - rc.top ) / 2 ) - 50*z );

// attention: x, z are horizontal, y is vertical axis !!!
#if 0
// Save the pixels before drawing the cross hair
s_lPixel[0] = GetPixel( hDC, s_lX-1, s_lY+0 );
s_lPixel[1] = GetPixel( hDC, s_lX+0, s_lY-1 );
s_lPixel[2] = GetPixel( hDC, s_lX+0, s_lY+0 );
s_lPixel[3] = GetPixel( hDC, s_lX+0, s_lY+1 );
s_lPixel[4] = GetPixel( hDC, s_lX+1, s_lY+0 );
#endif
// Draw a crosshair object in light red pixels
SetPixel( hDC, s_lX-1, s_lY+0, color );
SetPixel( hDC, s_lX+0, s_lY-1, color );
SetPixel( hDC, s_lX+0, s_lY+0, color );
SetPixel( hDC, s_lX+0, s_lY+1, color );
SetPixel( hDC, s_lX+1, s_lY+0, color );

ReleaseDC( hWndGrid, hDC );
}

*/
