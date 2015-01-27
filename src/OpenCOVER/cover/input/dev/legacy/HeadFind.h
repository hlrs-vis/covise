/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HEADFIND_H__
#define __HEADFIND_H__

/******************************************************************************\
*       This is a part of the Microsoft Source Code Samples. 
*       Copyright (C) 1993 - 1996 Microsoft Corp.
*       All rights reserved. 
*       This source code is only intended as a supplement to 
*       Microsoft Development Tools and/or WinHelp documentation.
*       See these sources for detailed information regarding the 
*       Microsoft samples programs.
\******************************************************************************/

#define USECOMM // yes, we need the COMM API

//#undef NO_STRICT    // be bold!

//#define HINSTANCE HANDLE

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#include <commdlg.h>
#include <string.h>
#include <io.h>
#include <memory.h>

// constant definitions

#define GWL_NPHEFDCOMINFO 0
#define HEFDCOMEXTRABYTES sizeof(LONG)

#define ABOUTDLG_USEBITMAP 1

#define ATOM_HEFDCOMINFO 0x100

// hard coded maximum number of ports for device under Win32

#define MAXPORTS 4
#define MaxCOMMANDS 12
#define CommandStringLen 40
// terminal size

#define MAXROWS 25
#define MAXCOLS 80

#define MAXBLOCK 80

#define MAXLEN_TEMPSTR 81

#define RXQUEUE 4096
#define TXQUEUE 4096

// cursor states

#define CS_HIDE 0x00
#define CS_SHOW 0x01

// Flow control flags

#define FC_DTRDSR 0x01
#define FC_RTSCTS 0x02
#define FC_XONXOFF 0x04

// ascii definitions

#define ASCII_BEL 0x07
#define ASCII_BS 0x08
#define ASCII_LF 0x0A
#define ASCII_CR 0x0D
#define ASCII_XON 0x11
#define ASCII_XOFF 0x13

typedef struct tagHEFDCOMINFO
{
    HANDLE idComDev;
    BYTE bPort;
    BOOL fConnected, fXonXoff, fLocalEcho, fNewLine, fAutoWrap,
        fUseCNReceive, fDisplayErrors;
    BYTE bByteSize, bFlowCtrl, bParity, bStopBits;
    DWORD dwBaudRate;
    DWORD rgbFGColor;
    HANDLE hPostEvent, hWatchThread, hWatchEvent;
    HWND hTermWnd;
    DWORD dwThreadID;
    OVERLAPPED osWrite, osRead;
} HEFDCOMINFO, NEAR *NPHEFDCOMINFO;

// macros ( for easier readability )

#define GETHINST(x) ((HINSTANCE)GetWindowLong(x, GWL_HINSTANCE))
#define GETNPHEFDCOMINFO(x) ((NPHEFDCOMINFO)GetWindowLong(x, GWL_NPHEFDCOMINFO))
#define SETNPHEFDCOMINFO(x, y) SetWindowLong(x, GWL_NPHEFDCOMINFO, (LONG)y)

#define COMDEV(x) (x->idComDev)
#define PORT(x) (x->bPort)
#define CONNECTED(x) (x->fConnected)
#define XONXOFF(x) (x->fXonXoff)
#define LOCALECHO(x) (x->fLocalEcho)
#define NEWLINE(x) (x->fNewLine)
#define AUTOWRAP(x) (x->fAutoWrap)
#define BYTESIZE(x) (x->bByteSize)
#define FLOWCTRL(x) (x->bFlowCtrl)
#define PARITY(x) (x->bParity)
#define STOPBITS(x) (x->bStopBits)
#define BAUDRATE(x) (x->dwBaudRate)
#define FGCOLOR(x) (x->rgbFGColor)
#define USECNRECEIVE(x) (x->fUseCNReceive)
#define DISPLAYERRORS(x) (x->fDisplayErrors)

#define POSTEVENT(x) (x->hPostEvent)
#define TERMWND(x) (x->hTermWnd)
#define HTHREAD(x) (x->hWatchThread)
#define THREADID(x) (x->dwThreadID)
#define WRITE_OS(x) (x->osWrite)
#define READ_OS(x) (x->osRead)

#define SET_PROP(x, y, z) SetProp(x, MAKEINTATOM(y), z)
#define GET_PROP(x, y) GetProp(x, MAKEINTATOM(y))
#define REMOVE_PROP(x, y) RemoveProp(x, MAKEINTATOM(y))

// CRT mappings to NT API

#define _fmemset memset
#define _fmemmove memmove

// function prototypes (private)

LRESULT NEAR CreateHEFDCOMInfo(HWND);
BOOL NEAR DestroyHEFDCOMInfo(HWND);
BOOL NEAR KillHEFDCOMFocus(HWND);
BOOL NEAR PaintHEFDCOM(HWND);
BOOL NEAR SetHEFDCOMFocus(HWND);
BOOL NEAR WriteHEFDCOMBlock(HWND, LPSTR, int);
int NEAR ReadCommBlock(HWND, LPSTR, int);
BOOL NEAR WriteCommBlock(HWND, LPSTR, DWORD);
BOOL NEAR OpenConnection(HWND);
BOOL NEAR SetupConnection(HWND);
BOOL NEAR CloseConnection(HWND);

VOID NEAR GoModalDialogBoxParam(HINSTANCE, LPCSTR, HWND, DLGPROC, LPARAM);

// function prototypes (public)

LRESULT FAR PASCAL HEFDCOMWndProc(HWND, UINT, WPARAM, LPARAM);
BOOL FAR PASCAL AboutDlgProc(HWND, UINT, WPARAM, LPARAM);
DWORD FAR PASCAL CommWatchProc(LPSTR);

int initHeadFinder(void *hMainWnd);
void useHF(bool value);
int closeHeadFinder(void *hMainWnd);
void getViewerPosition(int &x, int &y, int &z);
void SetHeadFinderPort(BYTE Port);
void SetBaudRate(DWORD ABaudRate);
void NextHeadFinderSetting();

//---------------------------------------------------------------------------
//  End of File: HEFDCOM.h
//---------------------------------------------------------------------------

#endif //__HEADFIND_H__