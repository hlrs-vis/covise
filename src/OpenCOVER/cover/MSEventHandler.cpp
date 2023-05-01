/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include "VRViewer.h"
#include "VRWindow.h"
#include "OpenCOVER.h"
#include <OpenVRUI/osg/mathUtils.h>
#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include "coVRConfig.h"
#include "coCullVisitor.h"

#include <osg/LightSource>
#include <osg/ApplicationUsage>
#include <osg/StateSet>
#include <osg/BlendFunc>
#include <osg/Camera>
#include <osg/CameraView>
#include <osg/DeleteHandler>
#include <osgDB/DatabasePager>
#include <osg/CullStack>
#include <osgText/Text>
#include <osgUtil/Statistics>
#include <osgUtil/UpdateVisitor>
#include <osgUtil/Optimizer>
#include <osgUtil/GLObjectsVisitor>
#include <osgDB/Registry>

#include <osgGA/AnimationPathManipulator>
#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/StateSetManipulator>

#include "coVRMSController.h"
#include "MSEventHandler.h"

#ifndef _WIN32
#include <termios.h>
#include <unistd.h>

#define CONSOLEINPUT
#endif

#ifdef __linux__
#include <linux/input.h>
#include <sys/inotify.h>
#endif

#ifndef OSG_NOTICE
#define OSG_NOTICE std::cerr
#endif

using namespace opencover;
using namespace covise;

#ifdef CONSOLEINPUT
// non-blocking input from stdin -- see http://ubuntuforums.org/showthread.php?t=1396108
static int getch()
{
    struct termios old;
    struct termios tmp;

    if (tcgetattr(STDIN_FILENO, &old))
    {
        return -1;
    }

    memcpy(&tmp, &old, sizeof(old));

    tmp.c_lflag &= ~ICANON & ~ECHO;

    if (tcsetattr(STDIN_FILENO, TCSANOW, (const struct termios *)&tmp))
    {
        return -1;
    }

    int oflags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (oflags == -1)
        return -1;
    fcntl(STDIN_FILENO, F_SETFL, oflags | O_NONBLOCK);

    int ch = getchar();

    fcntl(STDIN_FILENO, F_SETFL, oflags);

    tcsetattr(STDIN_FILENO, TCSANOW, (const struct termios *)&old);

    return ch;
}
#endif

MSEventHandler::MSEventHandler()
: handleTerminal(true)
, keyboardFd(-1)
, notifyFd(-1)
, watchFd(-1)
, modifierState(0)
{
    if (coVRMSController::instance()->isMaster())
    {
        handleTerminal = covise::coCoviseConfig::isOn("terminal", "COVER.Input.Keyboard", true);

        bool useEvent = false;
#ifdef __linux__
        devicePathname = covise::coCoviseConfig::getEntry("evdev", "COVER.Input.Keyboard", &useEvent);
        if (useEvent && openEvdev())
        {
            notifyFd = inotify_init();                                                                                             
            if (notifyFd < 0)
                perror("inotify_init");

            size_t last = devicePathname.rfind("/");
            devicePath = devicePathname.substr(0, last + 1);
            deviceName = devicePathname.substr(last + 1);
            watchFd = inotify_add_watch(notifyFd, devicePath.c_str(), IN_DELETE | IN_CREATE);
        }
#endif
    }
}

MSEventHandler::~MSEventHandler()
{
    if (keyboardFd >= 0)
        close(keyboardFd);
    if (watchFd >= 0)
        close(watchFd);
    if (notifyFd >= 0)
        close(notifyFd);
}

bool MSEventHandler::openEvdev()
{
#ifdef __linux__
#ifdef EVIOCGNAME
    keyboardFd = open(devicePathname.c_str(), O_RDONLY | O_NONBLOCK);
    if (keyboardFd < 0)
    {
        perror("Keyboard: failed to open device");
        return false;
    }

    char name[256] = "Unknown";
    if (ioctl(keyboardFd, EVIOCGNAME(sizeof(name)), name) < 0)
    {
        perror("Keyboard: no event device");
        close(keyboardFd);
        keyboardFd = -1;
        return false;
    }

    std::cerr << "Keyboard: evdev keyboard \"" << name << "\"" << std::endl;
    return true;
#endif
#endif
    return false;
}

bool MSEventHandler::update()
{
    if (coVRMSController::instance()->isMaster())
    {
#ifdef CONSOLEINPUT
        if (handleTerminal)
        {
            bool escape = false;
            bool withKey = false;

            int key = -1;
            while ((key = getch()) != -1)
            {
                fprintf(stderr, "console: key code=%d\n", key);

                if (key == 27)
                {
                    // Alt was pressed
                    escape = true;
                    continue;
                }

                int mod = 0;
                if (escape)
                {
                    mod |= osgGA::GUIEventAdapter::MODKEY_ALT;
                    escape = false;
                }
                else
                {
                    withKey = true;
                }

                if (key == 10)
                {
                    key = osgGA::GUIEventAdapter::KEY_Return;
                }
                else if (key >= 1 && key <= 26)
                {
                    // letter together with Ctrl
                    mod |= osgGA::GUIEventAdapter::MODKEY_CTRL;
                    key += 96;
                }
                else if (key >= 65 && key <= 90)
                {
                    mod |= osgGA::GUIEventAdapter::MODKEY_SHIFT;
                }
		
		eventQueue.push_back(Event(osgGA::GUIEventAdapter::KEYDOWN, mod, key));
		eventQueue.push_back(Event(osgGA::GUIEventAdapter::KEYUP, mod, key));
            }

            if (escape && !withKey)
            {
                eventQueue.push_back(Event(osgGA::GUIEventAdapter::KEYDOWN, 0, osgGA::GUIEventAdapter::KEY_Escape));
                eventQueue.push_back(Event(osgGA::GUIEventAdapter::KEYUP, 0, osgGA::GUIEventAdapter::KEY_Escape));
            }
        }
#endif

#ifdef __linux__
        {
            struct timeval time;
            time.tv_sec = 0;
            time.tv_usec = 0;
            fd_set rfds;
            FD_ZERO(&rfds);
            if (notifyFd >= 0)
            {
                FD_SET(notifyFd, &rfds);
                const int ret = select(notifyFd + 1, &rfds, NULL, NULL, &time);
                if (ret > 0 && FD_ISSET(notifyFd, &rfds))
                {
                    char notifyBuf[4096];
                    const int len = read(notifyFd, (struct inotify_event *)notifyBuf, sizeof(notifyBuf));
                    int i = 0;
                    while (i < len)
                    {
                        struct inotify_event *e = (struct inotify_event *)&notifyBuf[i];

                        if (e->mask & IN_DELETE)
                        {
                            if (deviceName == e->name)
                            {
                                std::cerr << "Keyboard: closing device " << devicePathname << std::endl;
                                close(keyboardFd);
                                keyboardFd = -1;
                            }
                        }
                        if (e->mask & IN_CREATE)
                        {
                            if (deviceName == e->name && keyboardFd == -1)
                            {
                                openEvdev();
                            }
                        }

                        i += (sizeof(struct inotify_event) + e->len);
                    }
                }
            }
        }

#ifdef EVIOCGNAME
        if (keyboardFd >= 0)
        {
            struct input_event ev;
            int ret = 0;
            while ((ret = read(keyboardFd, &ev, sizeof(ev))) == sizeof(ev))
            {
		Event event;

                if (ev.type == EV_KEY)
                {
                    bool handled = false;
#define LETTER(x) \
                    case KEY_##x: \
                                  event.key = #x[0] - 'A' + 'a'; \
                    handled = true; \
                    break;
#define FUNC(x) \
                    case KEY_F##x: \
                                   event.key = 0xffbe /*osgGA::GUIEventAdapter::KEY_F1*/ + x - 1; \
                    handled = true; \
                    break;

#define NUM(x) \
                    case KEY_##x: \
                                  event.key = #x[0]; \
                    handled = true; \
                    break;

#define KP(x) \
                    case KEY_KP##x: \
                                    event.key = #x[0]; \
                    handled = true; \
                    break;

                    int modifierBit = 0;
                    switch (ev.code)
                    {
                        LETTER(A)
                            LETTER(B)
                            LETTER(C)
                            LETTER(D)
                            LETTER(E)
                            LETTER(F)
                            LETTER(G)
                            LETTER(H)
                            LETTER(I)
                            LETTER(J)
                            LETTER(K)
                            LETTER(L)
                            LETTER(M)
                            LETTER(N)
                            LETTER(O)
                            LETTER(P)
                            LETTER(Q)
                            LETTER(R)
                            LETTER(S)
                            LETTER(T)
                            LETTER(U)
                            LETTER(V)
                            LETTER(W)
                            LETTER(X)
                            LETTER(Y)
                            LETTER(Z)

                            NUM(0)
                            NUM(1)
                            NUM(2)
                            NUM(3)
                            NUM(4)
                            NUM(5)
                            NUM(6)
                            NUM(7)
                            NUM(8)
                            NUM(9)

                            KP(0)
                            KP(1)
                            KP(2)
                            KP(3)
                            KP(4)
                            KP(5)
                            KP(6)
                            KP(7)
                            KP(8)
                            KP(9)

                            FUNC(1)
                            FUNC(2)
                            FUNC(3)
                            FUNC(4)
                            FUNC(5)
                            FUNC(6)
                            FUNC(7)
                            FUNC(8)
                            FUNC(9)
                            FUNC(10)
                            FUNC(11)
                            FUNC(12)
                            FUNC(13)
                            FUNC(14)
                            FUNC(15)
                            FUNC(16)
                            FUNC(17)
                            FUNC(18)
                            FUNC(19)
                            FUNC(20)
                            FUNC(21)
                            FUNC(22)
                            FUNC(23)
                            FUNC(24)

                        case KEY_SPACE:
                            event.key = ' ';
                            handled = true;
                            break;

                        case KEY_COMMA:
                            event.key = ',';
                            handled = true;
                            break;

                        case KEY_DOT:
                            event.key = '.';
                            handled = true;
                            break;

                        case KEY_SLASH:
                            event.key = '/';
                            handled = true;
                            break;

                        case KEY_ESC:
                            event.key = 27;
                            handled = true;
                            break;

                        case KEY_SEMICOLON:
                            event.key = ';';
                            handled = true;
                            break;

                        case KEY_APOSTROPHE:
                            event.key = '\'';
                            handled = true;
                            break;

                        case KEY_LEFTSHIFT:
                        case KEY_RIGHTSHIFT:
                            modifierBit = osgGA::GUIEventAdapter::MODKEY_SHIFT;
                            //handled = true;
                            break;

                        case KEY_LEFTALT:
                        case KEY_RIGHTALT:
                            modifierBit = osgGA::GUIEventAdapter::MODKEY_ALT;
                            //handled = true;
                            break;

                        case KEY_LEFTCTRL:
                        case KEY_RIGHTCTRL:
                            modifierBit = osgGA::GUIEventAdapter::MODKEY_CTRL;
                            //handled = true;
                            break;
                    }

                    if (ev.value == 0)
                    {
                        event.event = osgGA::GUIEventAdapter::KEYUP;
                        if (modifierBit)
                            modifierState &= ~modifierBit;
                    }
                    else if (ev.value == 1)
                    {
                        event.event = osgGA::GUIEventAdapter::KEYDOWN;
                        if (modifierBit)
                            modifierState |= modifierBit;
                    }

                    if (modifierState & osgGA::GUIEventAdapter::MODKEY_SHIFT)
                    {
                        if (event.key >= 'a' && event.key <= 'z')
                            event.key += 'A' - 'a';
                    }

                    event.mod = modifierState;

                    if (handled)
                    {
                        eventQueue.push_back(event);
                    }
                }
            }
#undef LETTER
#undef FUNC
#undef NUM
#undef KP
        }
#endif
#endif
    }

    int numEventsToSync = eventQueue.size();
    coVRMSController::instance()->syncData(&numEventsToSync, sizeof(numEventsToSync));
    if (coVRMSController::instance()->isSlave())
    {
        eventQueue.resize(numEventsToSync);
    }
    if (numEventsToSync)
    {
        coVRMSController::instance()->syncData(&eventQueue[0], numEventsToSync * sizeof(eventQueue[0]));
    }

    bool event = false;
    size_t index = 0;
    while (index < eventQueue.size())
    {
        event = true;
        const Event &event = eventQueue[index];
        OpenCOVER::instance()->handleEvents(event.event, event.mod, event.key);
        index++;
        // Delay rest of the buffer in case of: push/release/doubleclick/keydown/keyup
        // Tablets sometimes send push and release in one frame which would not work properly overwise.
        // Idea: It might be better to delay before the push- and release-events (in case it is not the first event in the queue).
        //       Then a change in position () can fully be processed before the click event arrives.
        if (event.event==osgGA::GUIEventAdapter::PUSH
                || event.event==osgGA::GUIEventAdapter::RELEASE
                || event.event==osgGA::GUIEventAdapter::DOUBLECLICK
                || event.event==osgGA::GUIEventAdapter::KEYUP
                || event.event==osgGA::GUIEventAdapter::KEYDOWN)
        {
            break;
        }
    }
    eventQueue.erase(eventQueue.begin(), eventQueue.begin()+index);

    return event;
}

bool MSEventHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &)
{
    Event ev(ea.getEventType(), 0, 0);

    switch (ea.getEventType())
    {
    case (osgGA::GUIEventAdapter::SCROLL):
    {
        ev.mod = ea.getScrollingMotion();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::PUSH):
    {
        ev.mod = ea.getButtonMask();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::DRAG):
    {
        //modBuffer[numEventsToSync] = (int) (((ea.getXnormalized()+1.0)/2.0) * cover->windows[0].sx);
        //keyBuffer[numEventsToSync] = (int) (((ea.getYnormalized()+1.0)/2.0) * cover->windows[0].sy);
        ev.mod = (int)(ea.getX() - ea.getXmin());
        ev.key = (int)(ea.getY() - ea.getYmin());
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::MOVE):
    {
        ev.mod = (int)(ea.getX() - ea.getXmin());
        ev.key = (int)(ea.getY() - ea.getYmin());
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::RELEASE):
    {
        ev.mod = ea.getButtonMask();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::DOUBLECLICK):
    {
        ev.mod = ea.getButtonMask();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::KEYDOWN):
    {
        ev.mod = ea.getModKeyMask();
        ev.key = ea.getKey();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::KEYUP):
    {
        ev.mod = ea.getModKeyMask();
        ev.key = ea.getKey();
        eventQueue.push_back(ev);
        return true;
    }
    case (osgGA::GUIEventAdapter::USER):
    {
        ev.mod = ea.getModKeyMask();
        eventQueue.push_back(ev);
        return true;
    }
    default:
    {
        return false;
    }
    }
}

