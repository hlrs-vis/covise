/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_MOUSEBUTTONS_H_
#define _CO_MOUSEBUTTONS_H_
/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			MouseButtons.cpp 				*
 *									*
 *	Description		read mouse button status
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/
#include <util/common.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
#undef USE_X11
#define USE_LINUX
#endif

#ifdef USE_X11
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/cursorfont.h>
#endif

#ifdef USE_LINUX
#define NUM_BUTTONS 3
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#undef USE_X11
#endif

#define EVENT_TYPE 1
#define PS2_TYPE 0
namespace opencover
{
class INPUT_LEGACY_EXPORT MouseButtons
{
private:
    bool poll();
    void mainLoop();
    void initialize();

    int getDeviceType(std::string name);
#ifdef USE_LINUX
    bool initPS2();
#endif

    bool useDedicatedButtonDevice;
#ifdef USE_X11
    int buttonPressEventType;
    int buttonReleaseEventType;
    Display *display;
    XEventClass eventClasses[50];
    int eventTypes[50];
    int buttonPressMask;
#endif

#ifdef USE_LINUX
    int oldwheelcounter;
#endif
    int wheelcounter;

    int device; // file descriptor for the opened device
#ifdef USE_LINUX
    int notify;
    int watch;
    std::string deviceName;
    std::string devicePath;
    char notifyBuf[4096];
#endif
    char buffer[4]; // [0] = buttonmask, [1] = dx, [2] = dy, [3] = dz (wheel)
    char buttonStatus;

public:
    MouseButtons(const char *buttonDeviceName = NULL);
    ~MouseButtons();
    void getButtons(int station, unsigned int *status);
    int getWheel(int station);
    void reset();
    int deviceType;
};
}
#endif
