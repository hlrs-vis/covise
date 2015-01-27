/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
 *	Description		MouseButtons optical tracking system interface class				*
 *									*
 *	Author			Uwe Woessner				*
 *									*
 *	Date			Jan 2004				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/common.h>
#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

#include "MouseButtons.h"
#ifndef WIN32
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef __linux__
#include <linux/input.h>
#endif
#else
#include <fcntl.h>
#endif

#ifdef USE_LINUX
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#if !defined CO_gecko && !defined CO_gcc3
#include <sys/inotify.h>
#endif
#include <sys/select.h>
#endif

using namespace opencover;
using vrui::vruiButtons;
//#define VERBOSE

#ifdef USE_LINUX
static char readps2(int fd)
{
    char ch;

    while (read(fd, &ch, 1) && (ch == (char)0xfa || ch == (char)0xaa))
    {
        fprintf(stderr, "<%02X>", ch & 0xff);
    }
    fprintf(stderr, "[%02X]", ch & 0xff);
    return (ch);
}
#endif

#ifdef USE_LINUX
bool MouseButtons::initPS2()
{
    std::string name = devicePath + deviceName;
    fprintf(stderr, "MouseButtons: trying to open device %s\n\n", name.c_str());
    memset(buffer, 0, 4);
    buttonStatus = 0;
    oldwheelcounter = 0;
    device = open(name.c_str(), O_RDWR | O_NONBLOCK);
    if (device == -1)
    {
        fprintf(stderr, "MouseButtons: could not open device %s\n\n", name.c_str());
        return false;
    }
    else
    {
        char ch;
        (void)ch;
        unsigned char getdevtype = 0xf2, disableps2 = 0xf5, imps2[6] = { 0xf3, 200, 0xf3, 100, 0xf3, 80 }, resetps2 = 0xff;

        fprintf(stderr, "write disable\n");
        ssize_t iret = write(device, &disableps2, 1);
        if (iret != 1)
            fprintf(stderr, "MouseButtons: error reading 'disableps2', wrong argument number\n");
        tcflush(device, TCIFLUSH);

        iret = write(device, &getdevtype, 1);
        if (iret != 1)
            fprintf(stderr, "MouseButtons: error reading 'disableps2', wrong argument number 2\n");
        usleep(1000);
        ch = readps2(device);

        iret = write(device, &resetps2, 1);
        if (iret != 1)
            fprintf(stderr, "MouseButtons: error reading 'resetps2', wrong argument number\n");
        usleep(1000);
        ch = readps2(device);

        tcflush(device, TCIFLUSH);
        iret = write(device, &getdevtype, 1);
        if (iret != 1)
            fprintf(stderr, "MouseButtons: error reading 'getdevtype', wrong argument number\n");
        usleep(1000);
        ch = readps2(device);

        iret = write(device, imps2, 6);
        if (iret != 6)
            fprintf(stderr, "MouseButtons: error reading 'imps2', wrong argument number\n");

        useDedicatedButtonDevice = true;
        wheelcounter = 0;

        return true;
    }
}
#endif

//-----------------------------------------------------------------------------

int MouseButtons::getDeviceType(std::string name)
{
#ifndef WIN32
    device = open(name.c_str(), O_RDONLY | O_NONBLOCK);
#else
    device = open(name.c_str(), O_RDONLY);
#endif
    if (device >= 0)
    {
        char name[256] = "Unknown";

#ifdef EVIOCGNAME
        if (ioctl(device, EVIOCGNAME(sizeof(name)), name) < 0)
        {
            std::cerr << "noEventDevice" << std::endl;
            close(device);
            return PS2_TYPE;
        }
#endif

        std::cerr << "EventDevice" << std::endl;
        useDedicatedButtonDevice = true;
        return EVENT_TYPE;
    }
    return -1;
}
MouseButtons::MouseButtons(const char *buttonDeviceName)
    : useDedicatedButtonDevice(false)
    , device(-1)
#ifdef USE_LINUX
    , notify(-1)
    , watch(-1)
#endif
{
    wheelcounter = 0;
    buttonStatus = 0;

#ifdef USE_LINUX
    oldwheelcounter = 0;
    deviceName = std::string(buttonDeviceName);
#if !defined CO_gecko && !defined CO_gcc3
    notify = inotify_init();
    if (notify < 0)
        perror("inotify_init");

    std::string name(buttonDeviceName);
    size_t last = name.rfind("/");
    devicePath = name.substr(0, last + 1);
    deviceName = name.substr(last + 1);

    watch = inotify_add_watch(notify, devicePath.c_str(), IN_DELETE | IN_CREATE);
#endif
    deviceType = getDeviceType(buttonDeviceName);
    if (deviceType == PS2_TYPE)
    {
        initPS2();
    }
#elif defined(USE_X11)
    if (buttonDeviceName)
    {
        display = XOpenDisplay(cover->inputDisplay);
        int numDevices = 0;
        XDeviceInfoPtr list = XListInputDevices(display, &numDevices);
        XDevice *buttonDevice = NULL;
        for (int i = 0; i < numDevices; i++)
        {
#ifdef VERBOSE
            fprintf(stderr, "found input device %s\n", list[i].name);
#endif
            if (!strcmp(list[i].name, buttonDeviceName))
            {
                buttonDevice = XOpenDevice(display, list[i].id);
                if (buttonDevice)
                {
#ifdef VERBOSE
                    fprintf(stderr, "using %s\n", buttonDeviceName);
#endif
                    break;
                }
            }
        }

        int numEventClasses = 0;
        uint32_t eventClass;
        if (buttonDevice)
        {
            Window window = DefaultRootWindow(display);
            XEventClass *oldEventClasses, *oldAllClasses;
            int allEvents;
            XGetSelectedExtensionEvents(display, window, &numEventClasses, &oldEventClasses, &allEvents, &oldAllClasses);
            int i;
            for (i = 0; i < numEventClasses; i++)
            {
                eventClasses[i] = oldEventClasses[i];
            }

            DeviceButtonPress(buttonDevice, buttonPressEventType, eventClass);
            eventClasses[numEventClasses] = eventClass;
            eventTypes[numEventClasses] = buttonPressEventType;
            numEventClasses++;

            DeviceButtonRelease(buttonDevice, buttonReleaseEventType, eventClass);
            eventClasses[numEventClasses] = eventClass;
            eventTypes[numEventClasses] = buttonReleaseEventType;
            numEventClasses++;

            XSelectExtensionEvent(display, window, eventClasses, numEventClasses);

            buttonPressMask = 0;
            wheelcounter = 0;

            useDedicatedButtonDevice = true;
        }
        else
        {
            fprintf(stderr, "MouseButtons: X Input device %s not found\n", buttonDeviceName);
        }
    }
#else
    (void)buttonDeviceName;
#endif
}

MouseButtons::~MouseButtons()
{
}

void MouseButtons::reset()
{
}

void
MouseButtons::getButtons(int /* station */, unsigned int *button)
{

    if (useDedicatedButtonDevice)
    {
#ifdef USE_LINUX
        if (device != -1)
        {
            int ret = 0;
            if (deviceType == PS2_TYPE)
            {
                while ((ret = read(device, buffer, 4)) == 4)
                {
                    buttonStatus = buffer[0];
                    buttonStatus &= ~(6);
                    buttonStatus |= ((buffer[0] & 2) << 1);
                    buttonStatus |= ((buffer[0] & 4) >> 1);
                    wheelcounter -= buffer[3];
                }
            }
            else
            {
                struct input_event ev;
                while ((ret = read(device, &ev, sizeof(ev))) == sizeof(ev))
                {
                    static int oldButtonState = 0;
                    if (ev.type == EV_KEY)
                    {
                        int b = -1;
                        if (ev.code == 104) // left
                            b = 0;
                        else if (ev.code == 109) // right
                            b = 1;
                        else if ((ev.code == 1) || (ev.code == 63) || ev.code == 425) // esc/F5
                            b = 2;
                        else if (ev.code == 52 || ev.code == 431) // blank
                            b = 3;
                        else if (ev.code == 114) // -
                            b = 4;
                        else if (ev.code == 115) // +
                            b = 5;
                        else if (ev.code >= BTN_0 && ev.code <= BTN_9)
                        {
                            b = ev.code - BTN_0;
                        }
                        else if (ev.code >= BTN_TRIGGER && ev.code <= BTN_DEAD)
                        {
                            b = ev.code - BTN_TRIGGER;
                        }
                        else if (ev.code >= BTN_A && ev.code <= BTN_THUMBR)
                        {
                            b = ev.code - BTN_A;
                        }
                        else if (ev.code >= BTN_LEFT && ev.code <= BTN_TASK)
                        {
                            b = ev.code - BTN_LEFT;
                        }
                        else
                        {
                            std::cerr << "unknown evdev code: " << ev.code << std::endl;
                        }
                        if (ev.value > 0)
                            oldButtonState |= 1 << b;
                        else
                            oldButtonState &= ~(1 << b);
                    }
                    else if (ev.type == EV_REL)
                    { // mouse position / trackpad
                        if (ev.code == REL_X)
                        {
                        }
                        else if (ev.code == REL_Y)
                        {
                            wheelcounter += ev.value;
                            std::cerr << "wheel: " << wheelcounter << endl;
                        }
                    }
                    buttonStatus = oldButtonState;
                }
            }

            *button = buttonStatus;
            if (oldwheelcounter != wheelcounter)
            {
                if (wheelcounter > 0)
                {
                    *button |= vruiButtons::WHEEL_UP;
                }
                else if (wheelcounter < 0)
                {
                    *button |= vruiButtons::WHEEL_DOWN;
                }
                oldwheelcounter = wheelcounter;
            }
        }
#if !defined CO_gecko && !defined CO_gcc3
        struct timeval time;
        time.tv_sec = 0;
        time.tv_usec = 0;
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(notify, &rfds);
        int ret = select(notify + 1, &rfds, NULL, NULL, &time);
        if (ret > 0 && FD_ISSET(notify, &rfds))
        {
            int len = read(notify, (struct inotify_event *)notifyBuf, 4096);
            if (len > 0)
            {

                int i = 0;
                while (i < len)
                {
                    struct inotify_event *e = (struct inotify_event *)&notifyBuf[i];

                    if (e->mask & IN_DELETE)
                    {
                        if (!strcmp(deviceName.c_str(), e->name))
                        {
                            fprintf(stderr, "MouseButtons: closing device [%s%s]\n", devicePath.c_str(), deviceName.c_str());
                            close(device);
                            device = -1;
                        }
                    }
                    if (e->mask & IN_CREATE)
                        if (!strcmp(deviceName.c_str(), e->name))
                            initPS2();

                    i += (sizeof(struct inotify_event) + e->len);
                }
            }
        }
#endif

#endif

#ifdef USE_X11
        XEvent event;
        while (XCheckTypedEvent(display, buttonPressEventType, &event))
        {
            XDeviceButtonEvent *bev = (XDeviceButtonEvent *)&event;
            if (bev->button == 4)
            {
                wheelcounter++;
            }
            else if (bev->button == 5)
            {
                wheelcounter--;
            }
            else
            {
                buttonPressMask |= 1 << (bev->button - 1);
            }
        }
        *button = buttonPressMask;
        while (XCheckTypedEvent(display, buttonReleaseEventType, &event))
        {
            XDeviceButtonEvent *bev = (XDeviceButtonEvent *)&event;
            if (bev->button != 4 && bev->button != 5)
            {
                buttonPressMask &= ~(1L << (bev->button - 1));
            }
        }
#endif
    }
    else
    {
        // just report mouse buttons
        *button = cover->getMouseButton()->getState();
    }
    //cerr << "Button: " << *button << endl;
}

int MouseButtons::getWheel(int /* station */)
{
    int ret = wheelcounter;
    wheelcounter = 0;
    return ret;
}
