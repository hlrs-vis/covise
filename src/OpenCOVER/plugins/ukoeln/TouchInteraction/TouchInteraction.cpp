/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TouchInteraction.cpp

#include "TouchInteraction.h"
#include "RRXEvent.h"

#include <cassert>
#include <cstdio>

#include <PluginUtil/PluginMessageTypes.h>

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/input/input.h>
#include <cover/input/coMousePointer.h>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

//
// If non-zero, touch move events are merged.
// I.e. RREV_TOUCHMOVE events succeeding each other will be collected and only a single
// onTouchesMoved() call will be issued.
//
#define MERGE_TOUCH_MOVE_EVENTS 1

#define MERGE_MOUSE_MOVE_EVENTS 0 // Not implemented

#define QUEUE_MOUSE_EVENTS true

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
TouchInteraction::TouchInteraction()
    : vrui::coInteraction(ButtonA, "TouchInteraction", NavigationHigh)
    , mouseID(-1)
    , touchCancel(false)
{
#ifdef _WIN32
    timeBeginPeriod(1);
#endif

    setTIState(TIS_Idle);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
TouchInteraction::~TouchInteraction()
{
#ifdef _WIN32
    timeEndPeriod(1);
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
double TouchInteraction::time() const
{
#ifdef _WIN32
    return 1.0e-3 * (double)timeGetTime();
#else
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onInit()
{
    vrui::coInteractionManager::the()->registerInteraction(this);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onDestroy()
{
    vrui::coInteractionManager::the()->unregisterInteraction(this);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onPreFrame()
{
    switch (getTIState())
    {
    case TIS_Activating:
        break;
    case TIS_ActiveMouse:
        sendPendingMouseEvents();
        break;
    default:
        sendPendingTouchEvents();
        break;
    }

    // Reset plugin state if there are no more touch points
    if (getTIState() == TIS_Inactive && getTouches().empty())
    {
        setTIState(TIS_Idle);
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onMessage(int type, int /*length*/, void const *data)
{
    if (type == opencover::PluginMessageTypes::RRZK_rrxevent)
    {
        rrxevent const *e = (rrxevent const *)data;

        switch (e->type)
        {
        case RREV_TOUCHPRESS:
            onTouchPressMessage(Touch(e->d1, time(), e->x, e->y, Touch::Pressed));
            break;
        case RREV_TOUCHMOVE:
            onTouchMoveMessage(Touch(e->d1, time(), e->x, e->y, Touch::Moved));
            break;
        case RREV_TOUCHRELEASE:
            onTouchReleaseMessage(Touch(e->d1, time(), e->x, e->y, Touch::Released));
            break;
        default:
            return;
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onTouchPressMessage(Touch const &touch)
{
    if (getTIState() == TIS_Idle)
    {
        assert(mouseID == -1);

        // First touch point recognized
        // Send a mouse-press event at the correct location to start the activation process
        sendMousePressEvent(touch.id, touch.x, touch.y);

        // Update the plugin state
        setTIState(TIS_Activating);

        // Save the mouse id
        mouseID = touch.id;
    }

    touchEvents.push_back(touch);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onTouchMoveMessage(Touch const &touch)
{
    touchEvents.push_back(touch);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onTouchReleaseMessage(Touch const &touch)
{
    touchEvents.push_back(touch);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::onTouchCancelMessage()
{
    // Balance the mouse release event if currently activating
    if (getTIState() == TIS_Activating)
    {
        sendMouseReleaseEvent(mouseID);
        mouseID = -1;
    }

    // Clear the list of touch points
    getTouches().clear();

    // and pending touch events
    touchEvents.clear();

    // Reset the plugin state
    setTIState(TIS_Idle);

    // Notify the parent class
    onTouchesCanceled();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::sendMousePressEvent(int /*id*/, float x, float y)
{
    osg::GraphicsContext::Traits const *traits = opencover::coVRConfig::instance()->windows[0].window->getTraits();

    int ix = int(x * traits->width + 0.5f);
    int iy = int(y * traits->height + 0.5f);

    // Move to current position
    opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::MOVE, ix, iy, QUEUE_MOUSE_EVENTS);

    // Send a mouse press event
    opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::PUSH, osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON, 0, QUEUE_MOUSE_EVENTS);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::sendMouseMoveEvent(int /*id*/, float x, float y)
{
    osg::GraphicsContext::Traits const *traits = opencover::coVRConfig::instance()->windows[0].window->getTraits();

    int ix = int(x * traits->width + 0.5f);
    int iy = int(y * traits->height + 0.5f);

    opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::MOVE, ix, iy, QUEUE_MOUSE_EVENTS);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::sendMouseReleaseEvent(int /*id*/)
{
    opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0, QUEUE_MOUSE_EVENTS);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::sendPendingTouchEvents()
{
#if MERGE_TOUCH_MOVE_EVENTS

    bool touchesMoved = false;

    while (!touchEvents.empty())
    {
        Touch const &touch = touchEvents.front();

        switch (touch.state)
        {
        case Touch::Pressed:
            if (touchesMoved)
            {
                onTouchesMoved(getTouches());
                touchesMoved = false;
            }
            onTouchesBeingPressed(getTouches());
            addTouch(touch);
            onTouchPressed(getTouches(), touch);
            break;

        case Touch::Moved:
            updateTouch(touch);
            touchesMoved = true;
            break;

        case Touch::Released:
            if (touchesMoved)
            {
                onTouchesMoved(getTouches());
                touchesMoved = false;
            }
            onTouchesBeingReleased(getTouches());
            removeTouch(touch);
            onTouchReleased(getTouches(), touch);
            break;

        case Touch::Undefined:
        default:
            break;
        }

        touchEvents.pop_front();
    }

    if (touchesMoved)
    {
        onTouchesMoved(getTouches());
    }

#else

    while (!touchEvents.empty())
    {
        Touch const &touch = touchEvents.front();

        switch (touch.state)
        {
        case Touch::Pressed:
            onTouchesBeingPressed(getTouches());
            addTouch(touch);
            onTouchPressed(getTouches(), touch);
            break;

        case Touch::Moved:
            updateTouch(touch);
            onTouchesMoved(getTouches());
            break;

        case Touch::Released:
            onTouchesBeingReleased(getTouches());
            removeTouch(touch);
            onTouchReleased(getTouches(), touch);
            break;

        case Touch::Undefined:
        default:
            break;
        }

        touchEvents.pop_front();
    }

#endif

    // Reset the plugin state if the last touch point was removed
    if (getTIState() == TIS_Active && getTouches().empty())
    {
        setTIState(TIS_Inactive);
    }

    //  touchEvents.clear();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::sendPendingMouseEvents()
{
    for (TouchEvents::iterator it = touchEvents.begin(); it != touchEvents.end(); ++it)
    {
        Touch &touch = *it;

        switch (touch.state)
        {
        case Touch::Pressed:
            // Just update list of touch points
            // The mouse press event has already been sent in onTouchPressMessage()
            addTouch(touch);
            break;

        case Touch::Moved:
            updateTouch(touch);
            if (touch.id == mouseID)
            {
                sendMouseMoveEvent(touch.id, touch.x, touch.y);
            }
            break;

        case Touch::Released:
            removeTouch(touch);
            if (touch.id == mouseID)
            {
                // Send the mouse release event
                sendMouseReleaseEvent(touch.id);
                // and update the plugin state
                setTIState(TIS_Inactive);
                mouseID = -1;
            }
            break;

        case Touch::Undefined:
        default:
            break;
        }
    }

    touchEvents.clear();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TouchInteraction::update()
{
    if (getTIState() == TIS_Activating)
    {
        if (activate())
        {
            // Balance the mouse press event
            sendMouseReleaseEvent(mouseID);
            // Active, ie. not acting as the mouse, ie. send touch events
            setTIState(TIS_Active);
            // Invalidate mouse id
            mouseID = -1;
        }
        else
        {
            // Activation failed, ie. a menu interaction was started, ie. send mouse events
            setTIState(TIS_ActiveMouse);
        }
    }
}
