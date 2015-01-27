/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-----------------------------------------------------------------------------
// File: AudioServerDX.cpp
//
// Desc: DirectX window application created by the DirectX AppWizard
//-----------------------------------------------------------------------------
#define STRICT

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <windows.h>
#include <basetsd.h>
#include <math.h>
#include <stdio.h>
#include <DXErr8.h>
#include <tchar.h>
#include <process.h>
#include "DMUtil.h"
#include "DSUtil.h"
#include "DXUtil.h"
#include "resource.h"
#include "AudioServerDX.h"
#include "COMMCTRL.H"

#include "common.h"
#include "loader.h"
#include "as_cache.h"
#include "as_comm.h"
#include "as_control.h"
#include "as_ServerSocket.h"
#include "as_client_api.h"
#include "as_opengl.h"

//CMyApplication theApp;

//-----------------------------------------------------------------------------
// Function prototypes
//-----------------------------------------------------------------------------
LRESULT CALLBACK StaticMsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//-----------------------------------------------------------------------------
// Global access to the app (needed for the global WndProc())
//-----------------------------------------------------------------------------
CMyApplication *g_pApp = NULL;
HINSTANCE g_hInst = NULL;
HANDLE g_hNotify;

static int aWidths[2];
static bool showStatus = true;
DWORD lpThreadId;
int run = 0;

void WaitForEvent(LPVOID lpv)
{
    DWORD dwResult;
    DMUS_NOTIFICATION_PMSG *pPmsg;

    IDirectMusicSegment *pSegment;
    IDirectMusicSegmentState8 *pSegmentState;
    as_Sound *pSound;

    while (run)
    {
        dwResult = WaitForSingleObject(g_hNotify, 100);
        if (NULL == g_pPerformance)
            continue;
        while (S_OK == g_pPerformance->GetNotificationPMsg(&pPmsg))
        {
            // Check notification type and do something in response
            if (GUID_NOTIFICATION_SEGMENT == pPmsg->guidNotificationType)
            {

                pSegmentState = (IDirectMusicSegmentState8 *)pPmsg->punkUser;
                pSegmentState->GetSegment(&pSegment);
                pSound = AS_Control->GetSoundBySegment(pSegment);

                switch (pPmsg->dwNotificationOption)
                {
                case DMUS_NOTIFICATION_SEGSTART:
                    //						AddLogMsg("DMUS_NOTIFICATION_SEGSTART");
                    if (NULL != pSound)
                        pSound->SetStatus(STATUS_PLAYING);
                    break;

                case DMUS_NOTIFICATION_SEGLOOP:
                    //						g_pPerformance->StopEx(NULL,0,0);
                    //						AddLogMsg("DMUS_NOTIFICATION_SEGLOOP");
                    if (NULL != pSound)
                    {
                        if (pSound->IsPlaying())
                            pSound->SetStatus(STATUS_LOOPING);
                    }
                    break;

                case DMUS_NOTIFICATION_SEGALMOSTEND:
                    //						AddLogMsg("DMUS_NOTIFICATION_SEGALMOSTEND");
                    break;

                case DMUS_NOTIFICATION_SEGEND:
                    //						AddLogMsg("DMUS_NOTIFICATION_SEGEND");
                    if (NULL != pSound)
                        pSound->SetStatus(STATUS_STOPPED);
                    break;

                case DMUS_NOTIFICATION_SEGABORT:
                    //						AddLogMsg("DMUS_NOTIFICATION_SEGABORT");
                    if (NULL != pSound)
                        pSound->SetStatus(STATUS_STOPPED);
                    break;

                default:
                    break;
                }
            }

            g_pPerformance->FreePMsg((DMUS_PMSG *)pPmsg);
            //			Sleep(1);
        }
        //		Sleep(1);
    }
    g_pPerformance->SetNotificationHandle(0, 0);
    CloseHandle(g_hNotify);
    _endthread();
}

DWORD WINAPI statusThread(HWND hDlg)
{
    static int i;

    static char stateStr[32];
    static char posStr[32];
    static char dirStr[32];
    static char nameStr[_MAX_PATH];
    static int maxSounds;
    static long status;
    static D3DVECTOR direction;
    static D3DVECTOR position;

    as_Sound *pSound;

    // status list control properties
    LVCOLUMN lvColumn;
    LVITEM lvItem;
    int iCol;
    char itemStr[32];

    // using list control IDC_LIST_3D_SOURCES

    // init control
    iCol = 0;
    lvColumn.mask = LVCF_TEXT | LVCF_WIDTH;
    lvColumn.fmt = LVCFMT_LEFT;
    lvColumn.pszText = "";
    lvColumn.cx = 20;
    SendMessage(hWndStatusList, LVM_INSERTCOLUMN, (WPARAM)(int)iCol, (LPARAM)(const LPLVCOLUMN)&lvColumn);

    iCol = 1;
    lvColumn.mask = LVCF_TEXT | LVCF_WIDTH;
    lvColumn.fmt = LVCFMT_LEFT;
    lvColumn.pszText = "#";
    lvColumn.cx = 30;
    SendMessage(hWndStatusList, LVM_INSERTCOLUMN, (WPARAM)(int)iCol, (LPARAM)(const LPLVCOLUMN)&lvColumn);

    iCol = 2;
    lvColumn.mask = LVCF_TEXT | LVCF_WIDTH;
    lvColumn.fmt = LVCFMT_LEFT;
    lvColumn.pszText = "Position";
    lvColumn.cx = 120;
    SendMessage(hWndStatusList, LVM_INSERTCOLUMN, (WPARAM)(int)iCol, (LPARAM)(const LPLVCOLUMN)&lvColumn);

    iCol = 3;
    lvColumn.mask = LVCF_TEXT | LVCF_WIDTH;
    lvColumn.fmt = LVCFMT_LEFT;
    lvColumn.pszText = "Direction";
    lvColumn.cx = 120;
    SendMessage(hWndStatusList, LVM_INSERTCOLUMN, (WPARAM)(int)iCol, (LPARAM)(const LPLVCOLUMN)&lvColumn);
    /*
      iCol = 4;
      lvColumn.mask = LVCF_TEXT | LVCF_WIDTH;
      lvColumn.fmt = LVCFMT_LEFT;
      lvColumn.pszText = "Sound";
      lvColumn.cx = 240;
      SendMessage(hWndStatusList, LVM_INSERTCOLUMN, (WPARAM) (int) iCol, (LPARAM) (const LPLVCOLUMN) &lvColumn);
   */
    maxSounds = 32;

    // add items
    // column 1 (No.)
    for (i = 0; i < maxSounds; i++)
    {
        // insert line and set icon at column 1
        lvItem.mask = LVIF_IMAGE;
        lvItem.iItem = i;
        lvItem.iSubItem = 0;
        lvItem.iImage = status;
        SendMessage(hWndStatusList, LVM_INSERTITEM, 0, (LPARAM)(const LPLVITEM)&lvItem);
        // set number at column 2
        lvItem.mask = LVIF_TEXT;
        lvItem.iItem = i;
        lvItem.iSubItem = 1;
        sprintf(itemStr, "%03d", i);
        lvItem.pszText = itemStr;
        SendMessage(hWndStatusList, LVM_SETITEM, 0, (LPARAM)(const LPLVITEM)&lvItem);
    }

    while (showStatus)
    {
        for (i = 0; i < maxSounds; i++)
        {
            pSound = AS_Control->getSoundByHandle(i);
            if (NULL == pSound)
            {
                sprintf(stateStr, "NOT INITIALISED");
                sprintf(posStr, " \0");
                sprintf(dirStr, " \0");
                //				sprintf(nameStr, " \0");
                status = STATUS_NOTUSED;
            }
            else
            {
                status = pSound->GetStatus();
                switch (status)
                {
                case STATUS_INITIAL:
                    sprintf(stateStr, "INITIALISED");
                    break;
                case STATUS_PLAYING:
                    sprintf(stateStr, "PLAYING");
                    break;
                case STATUS_LOOPING:
                    sprintf(stateStr, "PLAYING LOOP");
                    break;
                case STATUS_STOPPED:
                    sprintf(stateStr, "STOPPED");
                    break;
                case STATUS_NOTUSED:
                    sprintf(stateStr, "NOT INITIALISED");
                    break;
                default:
                    //						sprintf(stateStr, "NOT INITIALISED");
                    sprintf(stateStr, "-DEFAULT-(0x%08lX)", status);
                    break;
                }

                pSound->GetDirection(&direction);
                pSound->GetPosition(&position);

                // build string
                sprintf(posStr, "% 06.2f % 06.2f % 06.2f\0",
                        position.x, position.y, position.z);

                sprintf(dirStr, "% 06.2f % 06.2f % 06.2f\0",
                        direction.x, direction.y, direction.z);
                /*
                        // get sound name
                        pSound = AS_Control->getSoundByHandle(i);
                        sprintf(nameStr, "%s\0", AS_Control->GetSoundNameByHandle(i));
            */
            }

            // Column 1 (Status)
            lvItem.mask = LVIF_IMAGE;
            lvItem.iItem = i;
            lvItem.iSubItem = 0;
            lvItem.iImage = status;

            SendMessage(hWndStatusList,
                        LVM_SETITEM,
                        0,
                        (LPARAM)(const LPLVITEM)&lvItem);

            // Column 2 (Sound number)

            // Column 3 (Position)
            lvItem.mask = LVIF_TEXT;
            lvItem.iItem = i;
            lvItem.iSubItem = 2;
            lvItem.pszText = posStr;

            SendMessage(hWndStatusList,
                        LVM_SETITEM,
                        0,
                        (LPARAM)(const LPLVITEM)&lvItem);

            // Column 4 (Direction)
            lvItem.mask = LVIF_TEXT;
            lvItem.iItem = i;
            lvItem.iSubItem = 3;
            lvItem.pszText = dirStr;

            SendMessage(hWndStatusList,
                        LVM_SETITEM,
                        0,
                        (LPARAM)(const LPLVITEM)&lvItem);
            /*
                  // Column 5 (sound name)
                  lvItem.mask = LVIF_TEXT;
                  lvItem.iItem = i;
                  lvItem.iSubItem = 4;
                  lvItem.pszText = nameStr;
                  SendMessage(hWndStatusList,
                     LVM_SETITEM,
                     0,
                     (LPARAM) (const LPLVITEM) &lvItem
                  );
         */
        }
        Sleep(100);
    }

    return TRUE;
}

VOID CMyApplication::CheckLogState(HWND hWnd)
{
    int rc;
    HMENU hMenu;

    hMenu = GetMenu(hWnd);
    rc = GetMenuState(
        hMenu, // handle to destination window
        IDM_LOG, // control ID
        0);
    if (FALSE == enableLogList)
    {
        DestroyWindow(hWndLogList);
        CheckMenuItem(hMenu, IDM_LOG, MF_UNCHECKED);
        EnableMenuItem(hMenu, IDM_DISPLAY_CLEARLOG, MF_GRAYED);
    }
    else
    {
        hWndLogList = CreateLogView(m_hWnd);
        CheckMenuItem(hMenu, IDM_LOG, MF_CHECKED);
        EnableMenuItem(hMenu, IDM_DISPLAY_CLEARLOG, MF_ENABLED);
    }
    resizeControls(hWnd);
}

VOID CMyApplication::CheckStatusListState(HWND hWnd)
{
    int rc;
    HMENU hMenu;
    void *hStatus;

    hMenu = GetMenu(hWnd);
    rc = GetMenuState(
        hMenu, // handle to destination window
        IDM_STATUS, // control ID
        0);
    if (FALSE == enableStatusList)
    {
        showStatus = false;
        DestroyWindow(hWndStatusList);
        CheckMenuItem(hMenu, IDM_STATUS, MF_UNCHECKED);
    }
    else
    {
        hWndStatusList = CreateStatusView(m_hWnd);
        CheckMenuItem(hMenu, IDM_STATUS, MF_CHECKED);
        // start status thread
        showStatus = true;
        hStatus = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)statusThread, hWnd, 0, &lpThreadId);
    }
    resizeControls(hWnd);
}

LRESULT CMyApplication::resizeControls(HWND hWnd)
{
    long windowBottom;
    long windowRight;
    RECT statusRect;
    RECT windowRect;
    HDC hDC;

    GetClientRect(hWnd, &windowRect);
    windowBottom = windowRect.bottom;
    windowRight = windowRect.right;

    // resize and move status bar
    hDC = GetDC(hWndStatusBar);
    if (NULL != hDC)
    {
        // Resize the status bar to fit along the bottom of the client area.
        MoveWindow(hWndStatusBar, 0, windowBottom - 10, windowRight, windowBottom, TRUE);
        // Set the rectangles for the multiple parts of the status bar.
        aWidths[0] = 2 * windowRight / 3;
        aWidths[1] = -1;
        SendMessage(hWndStatusBar, SB_SETPARTS, 1, (LPARAM)aWidths);
        GetClientRect(hWndStatusBar, &statusRect);
        ReleaseDC(hWnd, hDC);

        // resize and move progress bar
        hDC = GetDC(hWndProgressBar);
        if (NULL != hDC)
        {
            // Resize the status bar to fit along the bottom of the client area.
            MoveWindow(hWndProgressBar, aWidths[0] + 4, windowBottom - 18, (windowRight / 3) - 4, 18, TRUE);
            // Set the rectangles for the multiple parts of the status bar.
            ReleaseDC(hWnd, hDC);
        }

        windowBottom -= statusRect.bottom;
    }

    if ((enableLogList) && (enableStatusList))
    {

        // set status to 2/3 and log to 1/3 of window height

        // move and resize log list
        hDC = GetDC(hWndLogList);
        if (NULL != hDC)
        {
            // Resize the list control to fit along the client area.
            MoveWindow(hWndLogList, 0, 2 * windowBottom / 3, windowRight, (windowBottom / 3), TRUE);
            ReleaseDC(hWnd, hDC);
        }

        // move and resize log list
        hDC = GetDC(hWndStatusList);
        if (NULL != hDC)
        {
            // Resize the list control to fit along the client area.
            MoveWindow(hWndStatusList, 0, 0, windowRight, 2 * windowBottom / 3, TRUE);
            ReleaseDC(hWnd, hDC);
        }
    }
    else
    {

        // move and resize log list
        hDC = GetDC(hWndLogList);
        if (NULL != hDC)
        {
            // Resize the list control to fit along the client area.
            MoveWindow(hWndLogList, 0, 0, windowRight, windowBottom, TRUE);
            ReleaseDC(hWnd, hDC);
        }

        // move and resize log list
        hDC = GetDC(hWndStatusList);
        if (NULL != hDC)
        {
            // Resize the list control to fit along the client area.
            MoveWindow(hWndStatusList, 0, 0, windowRight, windowBottom, TRUE);
            ReleaseDC(hWnd, hDC);
        }
    }
    return TRUE;
}

//-----------------------------------------------------------------------------
// CreateStatusView
// Desc: creates list view window
//-----------------------------------------------------------------------------
HWND CMyApplication::CreateStatusView(HWND hWndParent)
{
    HWND hWndList; // handle to list view window
    RECT rcl; // rectangle for setting size of window
    HICON hIcon; // handle to an icon
    //	int index;		// index used in for loops

    LV_COLUMN lvC; // list view column structure
    char szText[MAX_PATH]; // place to store some text
    //	LV_ITEM lvI;	// list view item structure
    //	int iSubItem;	// index into column header string table

    HINSTANCE hInst;

    hInst = (HINSTANCE)GetWindowLong(hWndParent, GWL_HINSTANCE);

    // Get the size and position of the parent window.
    GetClientRect(hWndParent, &rcl);

    // Create the list view window that starts out in details view
    // and supports label editing.
    hWndList = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        WC_LISTVIEW, // list view class
        "", // no default text
        WS_CLIPSIBLINGS | WS_VISIBLE | WS_CHILD | WS_BORDER | LVS_REPORT | /* | LVS_EDITLABELS | */ WS_EX_CLIENTEDGE, // styles
        0, 0,
        rcl.right - rcl.left, 2 * (rcl.bottom - rcl.top) / 3,
        hWndParent,
        NULL, //(HMENU)ID_LISTVIEW,
        hInst,
        NULL);

    if (hWndList == NULL)
        return NULL;

    // First initialize the image lists you will need:
    // create image lists for the small and the large icons.

    hImageListStatusSmallIcons = ImageList_Create(16, 16, FALSE, 3, 0);
    hImageListStatusLargeIcons = ImageList_Create(32, 32, FALSE, 3, 0);

    // STATUS_NOTUSED
    hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_NOTINIT));
    ImageList_AddIcon(hImageListStatusSmallIcons, hIcon);
    ImageList_AddIcon(hImageListStatusLargeIcons, hIcon);

    // STATUS_INITIAL
    hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_INIT));
    ImageList_AddIcon(hImageListStatusSmallIcons, hIcon);
    ImageList_AddIcon(hImageListStatusLargeIcons, hIcon);

    // STATUS_PLAYING
    hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_PLAY));
    ImageList_AddIcon(hImageListStatusSmallIcons, hIcon);
    ImageList_AddIcon(hImageListStatusLargeIcons, hIcon);

    // STATUS_LOOPING
    hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_PLAYLOOP));
    ImageList_AddIcon(hImageListStatusSmallIcons, hIcon);
    ImageList_AddIcon(hImageListStatusLargeIcons, hIcon);

    // STATUS_STOPPED
    hIcon = LoadIcon(hInst, MAKEINTRESOURCE(IDI_ICON_STOP));
    ImageList_AddIcon(hImageListStatusSmallIcons, hIcon);
    ImageList_AddIcon(hImageListStatusLargeIcons, hIcon);

    // Associate the image lists with the list view control.
    ListView_SetImageList(hWndList, hImageListStatusSmallIcons, LVSIL_SMALL);
    ListView_SetImageList(hWndList, hImageListStatusLargeIcons, LVSIL_NORMAL);

    // Now initialize the columns you will need.
    // Initialize the LV_COLUMN structure.
    // The mask specifies that the fmt, width, pszText, and subitem members
    // of the structure are valid.
    lvC.mask = LVCF_FMT | LVCF_WIDTH | LVCF_TEXT | LVCF_SUBITEM;
    lvC.fmt = LVCFMT_LEFT; // left-align column
    lvC.cx = 75; // width of column in pixels
    lvC.pszText = szText;

    enableStatusList = true;

    return (hWndList);
}

//-----------------------------------------------------------------------------
// CreateLogView
// Desc: creates list view window
//-----------------------------------------------------------------------------
HWND CMyApplication::CreateLogView(HWND hWndParent)
{
    HWND hWndList; // handle to list view window
    RECT rcl; // rectangle for setting size of window

    HINSTANCE hInst;

    numLogListItems = 0;
    HFONT hfnt;

    hInst = (HINSTANCE)GetWindowLong(hWndParent, GWL_HINSTANCE);

    // Get the size and position of the parent window.
    GetClientRect(hWndParent, &rcl);

    // Create the list view window that starts out in details view
    // and supports label editing.
    hWndList = CreateWindowEx(
        WS_EX_CLIENTEDGE,
        "listbox", // list view class
        "", // no default text
        // styles
        WS_CHILDWINDOW | WS_VISIBLE | WS_VSCROLL | WS_HSCROLL | LBS_NOINTEGRALHEIGHT | LBS_USETABSTOPS,
        0, 2 * (rcl.bottom - rcl.top) / 3,
        rcl.right - rcl.left, (rcl.bottom - rcl.top) / 3,
        hWndParent,
        NULL,
        hInst,
        NULL);

    if (hWndList == NULL)
        return NULL;

    hfnt = (HFONT)GetStockObject(ANSI_VAR_FONT);

    SendMessage(
        hWndList, // handle to destination window
        WM_SETFONT,
        (WPARAM)hfnt, // handle of font
        MAKELPARAM(0, 0) // redraw flag
        );

    if (LB_ERRSPACE == SendMessage(
                           hWndList, // handle to destination window
                           LB_INITSTORAGE, // message to send
                           (WPARAM)MAX_LOG_MSG, // number of items
                           (LPARAM)sizeof(char) * MSG_LEN // amount of memory
                           ))
    {
        char msg[128];
        sprintf(msg, "Could not init list, error %d", GetLastError());
        MessageBox(NULL, msg, "", MB_OK);
    }

    enableLogList = false;

    return (hWndList);
}

//-----------------------------------------------------------------------------
// Name: ReadSettings()
// Desc: Read the app settings from the registry
//-----------------------------------------------------------------------------
VOID CMyApplication::ReadSettings()
{
    HKEY hkey;
    if (ERROR_SUCCESS == RegCreateKeyEx(HKEY_CURRENT_USER, DXAPP_KEY,
                                        0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hkey, NULL))
    {

        unsigned long maxHeight;
        unsigned long maxWidth;

        DXUtil_ReadIntRegKey(hkey, TEXT("Width"), &m_dwCreationWidth, 500);
        DXUtil_ReadIntRegKey(hkey, TEXT("Height"), &m_dwCreationHeight, 300);
        DXUtil_ReadIntRegKey(hkey, TEXT("Top"), &m_dwCreationTop, 10);
        DXUtil_ReadIntRegKey(hkey, TEXT("Left"), &m_dwCreationLeft, 10);
        DXUtil_ReadIntRegKey(hkey, TEXT("Socket Port"), &m_SocketPort, SOCKPORT_DEFAULT);
        DXUtil_ReadIntRegKey(hkey, TEXT("Cache max"), &m_CacheSize, 0);

        maxHeight = GetSystemMetrics(SM_CYSCREEN);
        maxWidth = GetSystemMetrics(SM_CXSCREEN);

        DXUtil_ReadBoolRegKey(hkey, TEXT("Log"), &enableLogList, true);
        DXUtil_ReadBoolRegKey(hkey, TEXT("Status"), &enableStatusList, true);

        RegCloseKey(hkey);
    }
}

//-----------------------------------------------------------------------------
// Name: WriteSettings()
// Desc: Write the app settings to the registry
//-----------------------------------------------------------------------------
VOID CMyApplication::WriteSettings()
{
    HKEY hkey;

    RECT rct;

    GetWindowRect(
        m_hWnd, // handle to window
        &rct // address of structure for window coordinates
        );

    if (ERROR_SUCCESS == RegCreateKeyEx(HKEY_CURRENT_USER, DXAPP_KEY,
                                        0, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hkey, NULL))
    {
        DXUtil_WriteIntRegKey(hkey, TEXT("Width"), rct.right - rct.left);
        DXUtil_WriteIntRegKey(hkey, TEXT("Height"), rct.bottom - rct.top);
        DXUtil_WriteIntRegKey(hkey, TEXT("Top"), rct.top);
        DXUtil_WriteIntRegKey(hkey, TEXT("Left"), rct.left);
        DXUtil_WriteIntRegKey(hkey, TEXT("Socket Port"), m_SocketPort);
        DXUtil_WriteIntRegKey(hkey, TEXT("Cache max"), AS_Cache->GetMaxCacheSize());

        DXUtil_WriteBoolRegKey(hkey, TEXT("Log"), enableLogList);
        DXUtil_WriteBoolRegKey(hkey, TEXT("Status"), enableStatusList);

        RegCloseKey(hkey);
    }
}

//-----------------------------------------------------------------------------
// Name: InitAudio()
// Desc: Init both DirectMusic and DirectSound
//-----------------------------------------------------------------------------
HRESULT CMyApplication::InitAudio(HWND hDlg)
{
    HRESULT hr;
    long volume = DMUS_VOLUME_MAX;

    CoInitialize(NULL);
    hr = CoCreateInstance(
        CLSID_DirectMusicPerformance,
        NULL,
        CLSCTX_INPROC_SERVER,
        IID_IDirectMusicPerformance8,
        (void **)&g_pPerformance);
    if (FAILED(hr))
    {
        AddLogMsg("Could not CoCreateInstance: IID_IDirectMusicPerformance8");
        return S_FALSE;
    }

    hr = g_pPerformance->InitAudio(
        &g_pDirectMusic,
        NULL,
        hDlg,
        NULL,
        64,
        DMUS_AUDIOF_ALL,
        NULL);
    if (FAILED(hr))
    {
        AddLogMsg("Could not init performance");
        return S_FALSE;
    }

    hr = g_pPerformance->SetGlobalParam(GUID_PerfMasterVolume, &volume, sizeof(volume));
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not set performance volume, error %08lXh", hr);
        AddLogMsg(msg);
        return S_FALSE;
    }

    g_pDirectMusic->QueryInterface(IID_IDirectMusic8, (void **)&g_pDirectMusic8);

    // get master clock for benchmark and stress test
    hr = g_pDirectMusic8->GetMasterClock(
        NULL,
        &g_pReferenceClock);
    if (FAILED(hr))
    {
        char msg[MSG_LEN];
        sprintf(msg, "Could not get master clock, error %08lXh", hr);
        AddLogMsg(msg);
        return S_FALSE;
    }

    g_pPerformance->SetNotificationHandle(g_hNotify, 0);
    run = 1;
    _beginthread(WaitForEvent, 0, NULL);

    AddLogMsg("DirectMusic initialised");

    return S_OK;
}

//-----------------------------------------------------------------------------
// Name: CMyApplication()
// Desc: Constructor
//-----------------------------------------------------------------------------
CMyApplication::CMyApplication()
{
    g_pApp = this;

    m_hWnd = NULL;
    m_strWindowTitle = TEXT("AudioServerDX");
    m_dwCreationWidth = 500;
    m_dwCreationHeight = 375;
    m_bLoadingApp = TRUE;

    // Read settings from registry
    ReadSettings();
}

//-----------------------------------------------------------------------------
// Name: StaticMsgProc()
// Desc: Static msg handler which passes messages to the application class.
//-----------------------------------------------------------------------------
LRESULT CALLBACK StaticMsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    return g_pApp->MsgProc(hWnd, uMsg, wParam, lParam);
}

//-----------------------------------------------------------------------------
// Name: DialogSettingsProc()
// Desc: Callback for Settings dialog Windows messages
//-----------------------------------------------------------------------------
BOOL CALLBACK DialogSettingsProc(
    HWND hwndDlg,
    UINT message,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (message)
    {
    case WM_INITDIALOG:
    {
        char numStr[16];

        // get port
        sprintf(numStr, "%d", AS_Server->GetPort());
        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_EDIT_SOCKPORT, // control ID
            WM_SETTEXT, // message to send
            (WPARAM)0,
            (LPARAM)numStr);

        // get cache used disk space
        sprintf(numStr, "%6.2f", (double)(AS_Cache->getUsedDiskSpacekB()) / 1024);
        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_STATIC_DISK_USED, // control ID
            WM_SETTEXT, // message to send
            (WPARAM)0,
            (LPARAM)numStr);

        // get cache free disk space
        sprintf(numStr, "%6.2f", (double)AS_Cache->GetFreeDiskSpaceMB());
        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_STATIC_DISK_FREE, // control ID
            WM_SETTEXT, // message to send
            (WPARAM)0,
            (LPARAM)numStr);

        // retrieve current cache size limitation
        sprintf(numStr, "%6.2f", (double)AS_Cache->GetMaxCacheSize());
        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_EDIT_CACHE, // control ID
            WM_SETTEXT, // message to send
            (WPARAM)0,
            (LPARAM)numStr);
    }
    break;

    case WM_PAINT:
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_BUTTON_SOCKDEF:
        {
            char numStr[16];

            // set default socket port number
            sprintf(numStr, "%d", SOCKPORT_DEFAULT);
            SendDlgItemMessage(
                hwndDlg, // handle to destination window
                IDC_EDIT_SOCKPORT, // control ID
                WM_SETTEXT, // message to send
                (WPARAM)0,
                (LPARAM)numStr);
        }

        break;

        case IDC_CLEAR_CACHE:
        {
            int rc;

            rc = MessageBox(
                hwndDlg,
                "Empty the cache directory?",
                "Clear cache",
                MB_YESNO | MB_ICONQUESTION);
            if (IDYES == rc)
            {
                AS_Cache->clear();
            };
        }

        break;

        case IDOK:
        {
            char numStr[16];
            unsigned long newPort;
            unsigned long cacheSize;

            SendDlgItemMessage(
                hwndDlg, // handle to destination window
                IDC_EDIT_SOCKPORT, // control ID
                WM_GETTEXT, // message to send
                (WPARAM)16,
                (LPARAM)numStr);
            newPort = atol(numStr);
            if (newPort != AS_Server->GetPort())
            {
                if ((newPort < 22) || (newPort > 65535))
                {
                    MessageBox(hwndDlg, "Invalid socket port!", "Error", MB_ICONHAND);
                    SendDlgItemMessage(
                        hwndDlg, // handle to destination window
                        IDC_EDIT_SOCKPORT, // control ID
                        WM_UNDO, // message to send
                        (WPARAM)0,
                        (LPARAM)0);
                    break;
                }
                MessageBox(hwndDlg, "Please close application and restart to apply changes!",
                           "Settings", MB_ICONINFORMATION);
                AS_Server->SetPort((unsigned short)newPort);
            }

            SendDlgItemMessage(
                hwndDlg, // handle to destination window
                IDC_EDIT_CACHE, // control ID
                WM_GETTEXT, // message to send
                (WPARAM)16,
                (LPARAM)numStr);
            cacheSize = atol(numStr);
            if (cacheSize > AS_Cache->GetFreeDiskSpaceMB())
            {
                MessageBox(hwndDlg, "Wrong cache size!", "Error", MB_ICONHAND);
                sprintf(numStr, "%6.2f", (double)AS_Cache->GetFreeDiskSpaceMB());
                SendDlgItemMessage(
                    hwndDlg, // handle to destination window
                    IDC_EDIT_CACHE, // control ID
                    WM_SETTEXT, // message to send
                    (WPARAM)0,
                    (LPARAM)numStr);
                break;
            }
            AS_Cache->SetMaxCacheSize(cacheSize);
        }
        // Fall through and close dialog

        case IDCANCEL:
            EndDialog(hwndDlg, wParam);
            return TRUE;
        }
    }
    return FALSE;
}

//-----------------------------------------------------------------------------
// Name: DialogAboutProc()
// Desc: Callback for About dialog Windows messages
//-----------------------------------------------------------------------------
BOOL CALLBACK DialogAboutProc(
    HWND hwndDlg,
    UINT message,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (message)
    {
    case WM_INITDIALOG:
    {
        HWND hWnd = (HWND)GetWindowLong(hwndDlg, GWL_HWNDPARENT);
        HINSTANCE hInst = (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE);
        HANDLE hImage; ///< image resource

        // Load the grid bitmap
        hImage = LoadImage(
            hInst, // handle of the instance containing the image
            MAKEINTRESOURCE(IDB_LOGO_MINI), // name or identifier of image
            IMAGE_BITMAP, // type of image
            0, // desired width
            0, // desired height
            LR_VGACOLOR // load flags
            );
        if (NULL == hImage)
        {
            char msg[128];
            sprintf(msg, "Could not load image, error %d", GetLastError());
            MessageBox(NULL, msg, "", MB_OK);
        }

        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_STATIC_BMP, // control ID
            STM_SETIMAGE, // message to send
            (WPARAM)IMAGE_BITMAP,
            (LPARAM)hImage);
    }
    break;

    case WM_PAINT:
        break;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDOK:
        // Fall through.

        case IDCANCEL:
            EndDialog(hwndDlg, wParam);
            return TRUE;
        }
    }
    return FALSE;
}

//-----------------------------------------------------------------------------
// Name: DialogGridProc()
// Desc: Callback for Grid dialog Windows messages
//-----------------------------------------------------------------------------
BOOL CALLBACK DialogGridProc(
    HWND hwndDlg,
    UINT message,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (message)
    {
    case WM_INITDIALOG:
    {
        HWND hWnd = (HWND)GetWindowLong(hwndDlg, GWL_HWNDPARENT);
        HINSTANCE hInst = (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE);
        HANDLE hGridImage; ///< grid image resource

        // Load the grid bitmap
        hGridImage = LoadImage(
            hInst, // handle of the instance containing the image
            MAKEINTRESOURCE(IDB_GRID), // name or identifier of image
            IMAGE_BITMAP, // type of image
            0, // desired width
            0, // desired height
            LR_VGACOLOR // load flags
            );
        if (NULL == hGridImage)
        {
            char msg[128];
            sprintf(msg, "Could not load image, error %d", GetLastError());
            MessageBox(NULL, msg, "", MB_OK);
        }

        SendDlgItemMessage(
            hwndDlg, // handle to destination window
            IDC_GRID, // control ID
            STM_SETIMAGE, // message to send
            (WPARAM)IMAGE_BITMAP,
            (LPARAM)hGridImage);
    }
    break;

    case WM_PAINT:
        break;
    /*
                 case WM_COMMAND:
                     switch (LOWORD(wParam))
                     {
                         case IDOK:
                             // Fall through.
                        break;
                         case IDCANCEL:
         //                    EndDialog(hwndDlg, wParam);
         //					hWndGridView = NULL;
                             return TRUE;
         default:
         break;
         }
         break;
         */
    case WM_CLOSE:
        EndDialog(hwndDlg, wParam);
        hWndGridView = NULL;
        return TRUE;
    }
    return FALSE;
}

//-----------------------------------------------------------------------------
// Name: Dialog3DProc()
// Desc: Callback for 3D dialog Window messages
//-----------------------------------------------------------------------------
BOOL CALLBACK Dialog3DProc(
    HWND hWndDlg,
    UINT msg,
    WPARAM wParam,
    LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        Window::displayCallback();
    }
    break;

    case WM_SIZE:
    {
        int fwSizeType, nWidth, nHeight;
        fwSizeType = wParam; // resizing flag
        nWidth = LOWORD(lParam); // width of client area
        nHeight = HIWORD(lParam); // height of client area

        Window::reshapeCallback(nWidth, nHeight);
    }
    break;

    case WM_CLOSE:
        delete (OpenGL);
        OpenGL = NULL;
        EndDialog(hWnd3DView, wParam);
        hWnd3DView = NULL;
        return TRUE;
    }
    return FALSE;
}

//-----------------------------------------------------------------------------
// Name: MainWndproc()
// Desc: Callback for all Windows messages
//-----------------------------------------------------------------------------
LRESULT CMyApplication::MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    // TODO: Repond to Windows messages as needed

    case WM_APP:
        // userdefined messages, here: socket events
        //			AS_Communication->eventHandler(wParam, lParam);
        break;

    case WM_APP + 1:
        // userdefined messages, here: socket events
        //			AS_Communication->eventHandler(wParam, lParam);
        break;

    //		case WM_NCACTIVATE:
    //			return FALSE;

    case WM_COMMAND:
    {
        switch (LOWORD(wParam))
        {
        case IDM_EXIT:
            PostQuitMessage(0);
            break;

        case IDM_HELP_ABOUT:
            DialogBox(
                NULL,
                MAKEINTRESOURCE(IDD_ABOUT),
                hWnd,
                (DLGPROC)DialogAboutProc);
            break;

        case IDM_DISPLAY_3D:
        {

            HINSTANCE hInst;
            if (NULL != hWnd3DView)
            {
                BringWindowToTop(hWnd3DView);
                break;
            }
            hInst = (HINSTANCE)GetWindowLong(hWndMain, GWL_HINSTANCE);

            hWnd3DView = CreateDialog(
                hInst,
                MAKEINTRESOURCE(IDD_3DVIEW),
                hWnd,
                (DLGPROC)Dialog3DProc);

            if (NULL == hWnd3DView)
            {
                char msg[MSG_LEN];
                sprintf(msg, "CreateDialog error %08lXh", GetLastError());
                MessageBox(hWndMain, msg, "Error", MB_ICONHAND);
            }
            ShowWindow(hWnd3DView, SW_SHOWNORMAL);
            OpenGL = new as_OpenGL(hWnd3DView);
        }
        break;

        case IDM_DISPLAY_GRID:
        {
            HINSTANCE hInst;
            if (NULL != hWndGridView)
            {
                BringWindowToTop(hWndGridView);
                break;
            }

            hInst = (HINSTANCE)GetWindowLong(hWndMain, GWL_HINSTANCE);
            hWndGridView = CreateDialog(
                hInst,
                MAKEINTRESOURCE(IDD_GRIDVIEW),
                hWnd,
                (DLGPROC)DialogGridProc);
            if (NULL == hWndGridView)
            {
                char msg[MSG_LEN];
                sprintf(msg, "CreateDialog error %08lXh", GetLastError());
                MessageBox(hWndMain, msg, "Error", MB_ICONHAND);
            }
            ShowWindow(hWndGridView, SW_SHOWNORMAL);
        }
        break;

        case IDM_SETTINGS:
            DialogBox(
                NULL,
                MAKEINTRESOURCE(IDD_SETTINGS),
                hWnd,
                (DLGPROC)DialogSettingsProc);
            break;

        case IDM_STATUS:
        {
            int rc;
            HMENU hMenu;

            hMenu = GetMenu(hWnd);
            rc = GetMenuState(
                hMenu, // handle to destination window
                IDM_STATUS, // control ID
                0);
            if (MF_CHECKED == (rc & MF_CHECKED))
            {
                enableStatusList = false;
            }
            else
            {
                enableStatusList = true;
            }
            CheckStatusListState(hWnd);
        }
        break;

        case IDM_LOG:
        {
            int rc;
            HMENU hMenu;

            hMenu = GetMenu(hWnd);
            rc = GetMenuState(
                hMenu, // handle to destination window
                IDM_LOG, // control ID
                0);
            if (MF_CHECKED == (rc & MF_CHECKED))
            {
                enableLogList = false;
            }
            else
            {
                enableLogList = true;
            }
            CheckLogState(hWnd);
        }
        break;

        case IDM_DISPLAY_CLEARLOG:
        {
            int rc;

            rc = MessageBox(
                hWndMain,
                "Empty the log window?",
                "Clear log",
                MB_YESNO | MB_ICONQUESTION);
            if (IDYES == rc)
            {
                SendMessage(hWndLogList,
                            LB_RESETCONTENT,
                            0,
                            0);
            };
        }

        break;

        case LVN_GETDISPINFO:
        {
            SetStatusMsg("LVN_GETDISPINFO");
        }
        break;
        }
        break;
    }

    case WM_ACTIVATEAPP:
        m_bHasFocus = wParam;
        break;

    case WM_PAINT:
        break;

    case WM_MOVE:
        break;
    case WM_SIZE:
        resizeControls(hWnd);
        break;

    case WM_DESTROY:
        m_SocketPort = AS_Server->GetPort();
        WriteSettings();
        PostQuitMessage(0);
        break;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

//-----------------------------------------------------------------------------
// Name: Create()
// Desc: Creates the window
//-----------------------------------------------------------------------------
HRESULT CMyApplication::CreateMainWindow(HINSTANCE hInstance)
{
    int rc;
    HICON hIcon; ///< icon resource

    // Register the window class
    WNDCLASS wndClass = {
        CS_DBLCLKS, StaticMsgProc, 0, 0, hInstance, NULL,
        LoadCursor(NULL, IDC_ARROW),
        (HBRUSH)GetSysColorBrush(COLOR_WINDOW),
        NULL, _T("AudioServerDX Class")
    };
    RegisterClass(&wndClass);

    // Ensure that the common control DLL is loaded.
    InitCommonControls();

    // Create our main window
    HMENU hMenu = LoadMenu(NULL, MAKEINTRESOURCE(IDR_MENU));
    m_hWnd = CreateWindowEx(0, _T("AudioServerDX Class"),
                            m_strWindowTitle,
                            WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_EX_APPWINDOW,
                            m_dwCreationLeft, m_dwCreationTop,
                            m_dwCreationWidth, m_dwCreationHeight,
                            NULL, hMenu, hInstance, NULL);
    if (NULL == m_hWnd)
        return E_FAIL;
    hWndMain = m_hWnd;

    // Load the icon
    hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MAIN_ICON));

    // Set the icon for this dialog.
    // Set big icon
    SendMessage(m_hWnd, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
    // Set small icon
    SendMessage(m_hWnd, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);

    // create status bar
    hWndStatusBar = CreateWindowEx(
        0L, // extended style
        STATUSCLASSNAME, // create status bar
        "", // window title
        WS_CHILD | WS_VISIBLE | SBS_SIZEGRIP, // window styles
        0, 0, 0, 0, // x, y, width, height
        m_hWnd, // parent window
        NULL, // ID
        g_hInst, // instance
        NULL // window data
        );

    if (hWndStatusBar == NULL)
    {
        MessageBox(NULL, "Status bar not created!", NULL, MB_OK);
    }

    // Break the status bar into parts.
    SendMessage(hWndStatusBar, SB_SETPARTS, 1, (LPARAM)aWidths);

    // create progress bar
    hWndProgressBar = CreateWindowEx(
        0L, // extended style
        PROGRESS_CLASS, // create status bar
        "", // window title
        WS_CHILD | WS_VISIBLE | PBS_SMOOTH, // window styles
        0, 0, 0, 0, // x, y, width, height
        m_hWnd, // parent window
        NULL, // ID
        g_hInst, // instance
        NULL // window data
        );

    if (hWndProgressBar == NULL)
    {
        MessageBox(NULL, "Progress bar not created!", NULL, MB_OK);
    }

    // set progress bar properties
    rc = SendMessage(hWndProgressBar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
    rc = SendMessage(hWndProgressBar, PBM_SETPOS, (WPARAM)0, 0);

    UpdateWindow(m_hWnd);

    // Initialize the application timer
    DXUtil_Timer(TIMER_START);

    // Initialize the app's custom scene stuff
    // Drawing loading status message until app finishes loading
    SendMessage(m_hWnd, WM_PAINT, 0, 0);

    CheckStatusListState(m_hWnd);
    CheckLogState(m_hWnd);

    m_bLoadingApp = FALSE;

    // Initialize audio
    InitAudio(m_hWnd);

    // init Control system
    AS_Control = new as_Control();

    // init sound file cache
    AS_Cache = new as_cache();
    if (false == AS_Cache->IsInitialised())
    {
        AddLogMsg("Cache not initialised! Exiting now...");
        return -ERROR_BAD_ENVIRONMENT;
    }
    if (m_CacheSize > AS_Cache->GetFreeDiskSpaceMB())
    {
        char msg[MSG_LEN];
        sprintf(msg, "Not enough free disk space for requested cache size!\nRequested: %d MB, available: %d MB",
                m_CacheSize, AS_Cache->GetFreeDiskSpaceMB());
        MessageBox(hWndMain, msg, "Cache", MB_ICONEXCLAMATION);
        m_CacheSize = AS_Cache->GetFreeDiskSpaceMB();
    }
    AS_Cache->SetMaxCacheSize(m_CacheSize);

    // init DirectMusic Loader
    g_Loader = new C_DMLoader();

    // init server socket
    AS_Server = new as_ServerSocket();

    // start socket server
    AS_Server->port = (unsigned short)m_SocketPort;
    AS_Server->create(hWndMain);

    // init communication
    AS_Communication = new as_Comm(hWndMain);

    return S_OK;
}

//-----------------------------------------------------------------------------
// Name: Run()
// Desc: Handles the message loop and calls FrameMove() and Render() when
//       idle.
//-----------------------------------------------------------------------------
INT CMyApplication::Run()
{
    MSG msg;
    HACCEL hAccel = LoadAccelerators(g_hInst, MAKEINTRESOURCE(IDR_MAIN_ACCEL));

    // Message loop to run the app
    while (TRUE)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE))
        {
            if (FALSE == GetMessage(&msg, NULL, 0, 0))
                break;

            if (0 == TranslateAccelerator(m_hWnd, hAccel, &msg))
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        else
        {
            Sleep(20);
        }
    }

    return (INT)msg.wParam;
}

//-----------------------------------------------------------------------------
// Name: WinMain()
// Desc: Application entry point
//-----------------------------------------------------------------------------
INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    CMyApplication app;
    int rc;
    HINSTANCE hD3D8DLL = NULL;

    g_pApp = &app;
    g_hInst = hInstance;

    // Simple test if D3D8.dll exists.
    hD3D8DLL = LoadLibrary("D3D8.DLL");
    if (hD3D8DLL == NULL)
    {
        MessageBox(NULL, "DirectX 8.1 must be installed to run the AudioServer!!!", "DirectX error", MB_ICONHAND);
        exit(1);
    }

    g_hNotify = CreateEvent(NULL, FALSE, FALSE, NULL);

    InitCommonControls();

#ifdef LOGFILE
    logfile = fopen("AudioServer.log", "w");
#endif

    if (FAILED(g_pApp->CreateMainWindow(hInstance)))
    {
        return 0;
    }

    rc = g_pApp->Run();

#ifdef LOGFILE
    if (NULL != logfile)
        fclose(logfile);
#endif

    return rc;
}
