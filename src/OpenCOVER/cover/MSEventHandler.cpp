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
#include "ARToolKit.h"

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
#endif

#ifndef OSG_NOTICE
#define OSG_NOTICE std::cerr
#endif

using namespace opencover;
using namespace covise;

#ifndef _WIN32
// non-blocking input from stdin -- see http://ubuntuforums.org/showthread.php?t=1396108
static int getch()
{
    int ch;
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

    ch = getchar();

    fcntl(STDIN_FILENO, F_SETFL, oflags);

    tcsetattr(STDIN_FILENO, TCSANOW, (const struct termios *)&old);

    return ch;
}

#endif

void MSEventHandler::update()
{
    if (!(coVRMSController::instance()->isSlave()))
    {
#ifndef _WIN32
#ifndef DONT
        bool escape = false;

        int key = -1;
        while ((key = getch()) != -1)
        {
            fprintf(stderr, "key: %d\n", key);

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

            if (key >= 1 && key <= 26)
            {
                // letter together with Ctrl
                mod |= osgGA::GUIEventAdapter::MODKEY_CTRL;
                key += 64;
            }

            if (key >= 65 && key <= 90)
            {
                mod |= osgGA::GUIEventAdapter::MODKEY_SHIFT;
            }

            eventBuffer[numEventsToSync] = osgGA::GUIEventAdapter::KEYDOWN;
            modBuffer[numEventsToSync] = mod;
            keyBuffer[numEventsToSync] = key;
            ++numEventsToSync;

            eventBuffer[numEventsToSync] = osgGA::GUIEventAdapter::KEYUP;
            modBuffer[numEventsToSync] = mod;
            keyBuffer[numEventsToSync] = key;
            ++numEventsToSync;
        }
#endif
#endif

        coVRMSController::instance()->sendSlaves((char *)&numEventsToSync, sizeof(int));
        if (numEventsToSync)
        {
            coVRMSController::instance()->sendSlaves((char *)&eventBuffer, numEventsToSync * sizeof(int));
            coVRMSController::instance()->sendSlaves((char *)&keyBuffer, numEventsToSync * sizeof(int));
            coVRMSController::instance()->sendSlaves((char *)&modBuffer, numEventsToSync * sizeof(int));
        }
    }
    else
    {
        if (coVRMSController::instance()->readMaster((char *)&numEventsToSync, sizeof(int)) < 0)
        {
            cerr << "numEventsToSync not read message from Master" << endl;
            exit(1);
        }
        if (numEventsToSync)
        {
            if (coVRMSController::instance()->readMaster((char *)&eventBuffer, numEventsToSync * sizeof(int)) < 0)
            {
                cerr << "numEventsToSync not read message from Master" << endl;
                exit(1);
            }
            if (coVRMSController::instance()->readMaster((char *)&keyBuffer, numEventsToSync * sizeof(int)) < 0)
            {
                cerr << "numEventsToSync not read message from Master" << endl;
                exit(1);
            }
            if (coVRMSController::instance()->readMaster((char *)&modBuffer, numEventsToSync * sizeof(int)) < 0)
            {
                cerr << "numEventsToSync not read message from Master" << endl;
                exit(1);
            }
        }
    }

    int index = 0;
    while (index < numEventsToSync)
    {
        int currentEvent = eventBuffer[index];
        OpenCOVER::instance()->handleEvents(currentEvent, modBuffer[index], keyBuffer[index]);
        index++;
        // Delay rest of the buffer in case of: push/release/doubleclick/keydown/keyup
        // Tablets sometimes send push and release in one frame which would not work properly overwise.
        // Idea: It might be better to delay before the push- and release-events (in case it is not the first event in the queue).
        //       Then a change in position () can fully be processed before the click event arrives.
        if ((currentEvent == 1) || (currentEvent == 2) || (currentEvent == 4) || (currentEvent == 32) || (currentEvent == 64))
        {
            for (int i = index; i < numEventsToSync; i++)
            {
                eventBuffer[i - index] = eventBuffer[i];
                modBuffer[i - index] = modBuffer[i];
                keyBuffer[i - index] = keyBuffer[i];
            }
            break;
        }
    }
    numEventsToSync -= index;
}

bool MSEventHandler::handle(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &)
{
    eventBuffer[numEventsToSync] = ea.getEventType();
    switch (ea.getEventType())
    {
    case (osgGA::GUIEventAdapter::SCROLL):
    {
        modBuffer[numEventsToSync] = ea.getScrollingMotion();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::PUSH):
    {
        modBuffer[numEventsToSync] = ea.getButtonMask();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::DRAG):
    {
        //modBuffer[numEventsToSync] = (int) (((ea.getXnormalized()+1.0)/2.0) * cover->windows[0].sx);
        //keyBuffer[numEventsToSync] = (int) (((ea.getYnormalized()+1.0)/2.0) * cover->windows[0].sy);
        modBuffer[numEventsToSync] = (int)(ea.getX() - ea.getXmin());
        keyBuffer[numEventsToSync] = (int)(ea.getY() - ea.getYmin());
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::MOVE):
    {
        modBuffer[numEventsToSync] = (int)(ea.getX() - ea.getXmin());
        keyBuffer[numEventsToSync] = (int)(ea.getY() - ea.getYmin());
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::RELEASE):
    {
        modBuffer[numEventsToSync] = ea.getButtonMask();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::DOUBLECLICK):
    {
        modBuffer[numEventsToSync] = ea.getButtonMask();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::KEYDOWN):
    {
        modBuffer[numEventsToSync] = ea.getModKeyMask();
        keyBuffer[numEventsToSync] = ea.getKey();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::KEYUP):
    {
        modBuffer[numEventsToSync] = ea.getModKeyMask();
        keyBuffer[numEventsToSync] = ea.getKey();
        numEventsToSync++;
        return true;
    }
    case (osgGA::GUIEventAdapter::USER):
    {
        modBuffer[numEventsToSync] = ea.getModKeyMask();
        numEventsToSync++;
        return true;
    }
    default:
    {
        return false;
    }
    }
}

