/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CO_RAW_MOUSE_H
#define __CO_RAW_MOUSE_H

#ifdef WIN32

#if !defined(_WIN32_WINNT) && !defined(_CHICAGO_)
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#endif

#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#include <util/common.h>
#include <stdio.h>

#include "coRawMouse.h"
#include <iostream>
#include <sstream>
namespace opencover
{

//============================================================
//	Dynamically linked functions from rawinput
//============================================================
typedef WINUSERAPI INT(WINAPI *pGetRawInputDeviceList)(OUT PRAWINPUTDEVICELIST pRawInputDeviceList, IN OUT PINT puiNumDevices, IN UINT cbSize);
typedef WINUSERAPI INT(WINAPI *pGetRawInputData)(IN HRAWINPUT hRawInput, IN UINT uiCommand, OUT LPVOID pData, IN OUT PINT pcbSize, IN UINT cbSizeHeader);
typedef WINUSERAPI INT(WINAPI *pGetRawInputDeviceInfoA)(IN HANDLE hDevice, IN UINT uiCommand, OUT LPVOID pData, IN OUT PINT pcbSize);
typedef WINUSERAPI BOOL(WINAPI *pRegisterRawInputDevices)(IN PCRAWINPUTDEVICE pRawInputDevices, IN UINT uiNumDevices, IN UINT cbSize);

//============================================================
//	PARAMETERS
//============================================================

#define RAW_SYS_MOUSE 0 // The sys mouse combines all the other usb mice into one
#define MAX_RAW_MOUSE_BUTTONS 5

//============================================================
//	DATA TYPES
//============================================================

typedef struct STRUCT_RAW_MOUSE
{

    // Identifier for the mouse.  WM_INPUT passes the device HANDLE as lparam when registering a mousemove
    HANDLE device_handle;

    // The running tally of mouse moves received from WM_INPUT (mouse delta).
    ULONG x;
    ULONG y;
    ULONG z;

    // Used to determine if the HID is using absolute mode or relative mode
    //    The Act Labs PC USB Light Gun is absolute mode (returns screen coordinates)
    //    and mice are relative mode (returns delta)
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_absolute;
    // This indicates if the coordinates are coming from a multi-monitor setup
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_virtual_desktop;

    int buttonpressed[MAX_RAW_MOUSE_BUTTONS];

    // Identifying the name of the button may be useful in the future as a way to
    //   use a mousewheel as a button and other neat tricks (button name: "wheel up", "wheel down")
    //   -- not a bad way to look at it for a rotary joystick
    char *button_name[MAX_RAW_MOUSE_BUTTONS];

    char *mouseName;

} RAW_MOUSE, *PRAW_MOUSE;

//============================================================
//	LOCAL VARIABLES
//============================================================

class INPUT_LEGACY_EXPORT coRawMouse
{
private:
    int buttonNumber;

public:
    coRawMouse(int n);
    coRawMouse(const char *);
    ~coRawMouse();

    int getX();
    int getY();
    int getWheelCount();
    unsigned int getButtonBits();
    bool getButton(int i);
};

class coRawMouseManager
{
private:
    coRawMouseManager();
    static coRawMouseManager *inst;

    int excluded_sysmouse_devices_count;

    BOOL IncludeSysMouse, IncludeRemoteDeskTopMouse, IncludeIndividualMice;

    pGetRawInputDeviceList _GRIDL;
    pGetRawInputData _GRID;
    pGetRawInputDeviceInfoA _GRIDIA;
    pRegisterRawInputDevices _RRID;

    int nnumMice;

    HWND handle_;
    HGLRC context_;
    HINSTANCE instance_;
    LPBYTE lpb;
    int oldSize;

    BOOL is_rm_rdp_mouse(char cDeviceString[]);

    BOOL read_raw_input(PRAWINPUT);

    void setupDevices();

    // register to reviece WM_INPUT messages in WNDPROC
    BOOL register_raw_mouse(void);

public:
    ~coRawMouseManager();
    static coRawMouseManager *instance();
    void update();
    int numMice();

    // Pointer to our array of raw mice.  Created by called init_raw_mouse()
    PRAW_MOUSE rawMice;
    // Every time the WM_INPUT message is received, the lparam must be passed to this function to keep a running tally of
    BOOL processData(HANDLE); // device handle

    // Fetch the relative position of the mouse since the last time get_raw_mouse_x_delta() or get_raw_mouse_y_delta
    //    was called
    ULONG get_raw_mouse_x_delta(int);
    ULONG get_raw_mouse_y_delta(int);
    ULONG get_raw_mouse_z_delta(int);

    // pass the mousenumber, button number, returns 0 if the button is up, 1 if the button is down
    BOOL is_raw_mouse_button_pressed(int, int);
    char *get_raw_mouse_button_name(int, int);

    // Used to determine if the HID is using absolute mode or relative mode
    //    The Act Labs PC USB Light Gun is absolute mode (returns screen coordinates)
    //    and mice are relative mode (returns delta)
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_raw_mouse_absolute(int);

    // This indicates if the coordinates are coming from a multi-monitor setup
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_raw_mouse_virtual_desktop(int);
};
}
#endif
#endif /* ifndef __CO_RAW_MOUSE_H */
