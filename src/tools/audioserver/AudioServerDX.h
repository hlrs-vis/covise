/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
// File: AudioServerDX.h
//
// Desc: Header file AudioServerDX sample app
//-----------------------------------------------------------------------------
#ifndef AFX_AUDIOSERVERDX_H__9C21AEA8_78D2_462A_AA75_A2E08239630A__INCLUDED_
#define AFX_AUDIOSERVERDX_H__9C21AEA8_78D2_462A_AA75_A2E08239630A__INCLUDED_

//-----------------------------------------------------------------------------
// Defines, and constants
//-----------------------------------------------------------------------------
// TODO: change "DirectX AppWizard Apps" to your name or the company name
#define DXAPP_KEY TEXT("Software\\AudioServerDX")

#include "commctrl.h"

//-----------------------------------------------------------------------------
// Name: class CMyApplication
// Desc: Application class.
//-----------------------------------------------------------------------------
class CMyApplication
{
    BOOL m_bLoadingApp; // TRUE, if the app is loading
    BOOL m_bHasFocus; // TRUE, if the app has focus
    TCHAR *m_strWindowTitle; // Title for the app's window
    HWND m_hWnd; // The main app window
    FLOAT m_fTime; // Current time in seconds
    FLOAT m_fElapsedTime; // Time elapsed since last frame

    DWORD m_dwCreationTop; // Top used to create window
    DWORD m_dwCreationLeft; // Left used to create window
    DWORD m_dwCreationWidth; // Width used to create window
    DWORD m_dwCreationHeight; // Height used to create window

    unsigned long m_SocketPort; // socket port

    FLOAT m_fWorldRotX; // World rotation state X-axis
    FLOAT m_fWorldRotY; // World rotation state Y-axis
    unsigned long m_CacheSize; // maximum cache utilisation disk space

    HIMAGELIST hImageListStatusSmallIcons; // handles to image lists for small icons
    HIMAGELIST hImageListStatusLargeIcons; // handles to image lists for large icons

protected:
    HRESULT OneTimeSceneInit();

    HRESULT InitAudio(HWND hWnd);

    VOID ReadSettings();
    VOID WriteSettings();

public:
    HRESULT CreateMainWindow(HINSTANCE hInstance);
    INT Run();
    LRESULT MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

    LRESULT resizeControls(HWND hWnd);
    VOID Pause(BOOL bPause);
    HWND CreateLogView(HWND hWndParent);
    HWND CreateStatusView(HWND hWndParent);
    VOID CheckStatusListState(HWND hWnd);
    VOID CheckLogState(HWND hWnd);

    CMyApplication();
};
#endif // !defined(AFX_AUDIOSERVERDX_H__9C21AEA8_78D2_462A_AA75_A2E08239630A__INCLUDED_)
