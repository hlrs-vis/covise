/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CO_RAW_MOUSE_H
#define __CO_RAW_MOUSE_H

#ifdef WIN32

#if !defined(_WIN32_WINNT) && !defined(_CHICAGO_)
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawdevice from the user32.dll
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

#include "vvRawDevice.h"
#include <iostream>
#include <sstream>
namespace vive
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

#define RAW_SYS_MOUSE 0 // The sys device combines all the other usb mice into one
#define MAX_RAW_MOUSE_BUTTONS 20

//============================================================
//	DATA TYPES
//============================================================

typedef struct STRUCT_RAW_MOUSE
{

    // Identifier for the device.  WM_INPUT passes the device HANDLE as lparam when registering a devicemove
    HANDLE device_handle;

    // The running tally of device moves received from WM_INPUT (device delta).
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
    //   use a devicewheel as a button and other neat tricks (button name: "wheel up", "wheel down")
    //   -- not a bad way to look at it for a rotary joystick
    char *button_name[MAX_RAW_MOUSE_BUTTONS];

    char *deviceName;

    // type can be RIM_TYPEMOUSE RIM_TYPEKEYBOARD or RIM_TYPEHID
    int type;

} RAW_MOUSE, *PRAW_MOUSE;

//============================================================
//	LOCAL VARIABLES
//============================================================

class vvRawDevice
{
private:
    int buttonNumber;

public:
    vvRawDevice(int n);
    vvRawDevice(const char *);
    ~vvRawDevice();

    int getX();
    int getY();
    int getWheelCount();
    unsigned int getButtonBits();
    bool getButton(int i);
};

class vvRawDeviceManager
{
private:
    vvRawDeviceManager();
    static vvRawDeviceManager *inst;



    pGetRawInputDeviceList _GRIDL;
    pGetRawInputData _GRID;
    pGetRawInputDeviceInfoA _GRIDIA;
    pRegisterRawInputDevices _RRID;

    int nInputDevices;

    HWND handle_;
    HGLRC context_;
    HINSTANCE instance_;
    LPBYTE lpb;
    int oldSize;

    BOOL is_rm_rdp_device(char cDeviceString[]);

    BOOL read_raw_input(PRAWINPUT);

    void setupDevices();

    // register to reviece WM_INPUT messages in WNDPROC
    BOOL register_raw_device(void);
    
    void escape(std::string &data);

public:
    ~vvRawDeviceManager();
    static vvRawDeviceManager *instance();
    void update();
    int numDevices();

    // Pointer to our array of raw mice.  Created by called init_raw_device()
    PRAW_MOUSE rawDevices;
    // Every time the WM_INPUT message is received, the lparam must be passed to this function to keep a running tally of
    BOOL processData(HANDLE); // device handle

    // Fetch the relative position of the device since the last time get_raw_device_x_delta() or get_raw_device_y_delta
    //    was called
    ULONG get_raw_device_x_delta(int);
    ULONG get_raw_device_y_delta(int);
    ULONG get_raw_device_z_delta(int);

    // pass the devicenumber, button number, returns 0 if the button is up, 1 if the button is down
    BOOL is_raw_device_button_pressed(int, int);
    char *get_raw_device_button_name(int, int);

    // Used to determine if the HID is using absolute mode or relative mode
    //    The Act Labs PC USB Light Gun is absolute mode (returns screen coordinates)
    //    and mice are relative mode (returns delta)
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_raw_device_absolute(int);

    // This indicates if the coordinates are coming from a multi-monitor setup
    // NOTE: this value isn't updated until the device registers a WM_INPUT message
    BOOL is_raw_device_virtual_desktop(int);
};
}
#endif
#endif /* ifndef __CO_RAW_MOUSE_H */
