/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                              (C)2010 Anton Baumesberger  **
 **                                                                          **
 ** Description: Utouch3D Plugin                                             **
 **                                                                          **
 **                                                                          **
 ** Author: Anton Baumesberger	                                             **
 **                                                                          **
\****************************************************************************/

#include "Utouch3DPlugin.h"
#include "RREvent.h"
#include "TouchCursor.h"

#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

using namespace opencover;

OpenThreads::Mutex Utouch3DPlugin::mutex;

Utouch3DPlugin::Utouch3DPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, covise::coInteraction(covise::coInteraction::ButtonA, "Utouch3D", covise::coInteraction::NavigationHigh)
, touchPointsVisible(0)
, needsActivation(false)
, needsDeactivation(false)
, isActivating(false)
, actingAsMouse(false)
, mouseID(-1)
, checkForMenuEventsHandled(0)
{
}

// this is called if the plugin is removed at runtime
Utouch3DPlugin::~Utouch3DPlugin()
{
    covise::coInteractionManager::the()->unregisterInteraction(this);
}

bool Utouch3DPlugin::init()
{
    std::cout << "Utouch3D2Plugin... init()" << std::endl;

    covise::coInteractionManager::the()->registerInteraction(this);

    intMan = new InteractionManager(this, cover);

    std::cout << "Utouch3D2Plugin... init() done" << std::endl;

    return true;
}

bool Utouch3DPlugin::destroy()
{
    delete intMan;
    intMan = 0;

    return true;
}

void Utouch3DPlugin::startTouchInteraction()
{
}

void Utouch3DPlugin::stopTouchInteraction()
{
}

bool Utouch3DPlugin::isTouchInteractionRunning()
{
    return state == Active;
}

/**
 * dequeues @code fakedMouseEvents and processes one event per call
 * using a mutex
 */
void Utouch3DPlugin::preFrame()
{
    //mutex.lock();
    {
        // Start interaction?
        if (!actingAsMouse)
        {
            if (state == Idle && checkForMenuEventsHandled == 2)
            {
                needsActivation = true;
                needsDeactivation = false;

                checkForMenuEventsHandled = 0;
            }
        }

        // Synchronize touch point list with TouchInteraction
        syncTouchPoints();

        // Stop interaction?
        if (touchPointsVisible == 0)
        {
            if (state == Active)
            {
                needsDeactivation = true;
            }
            else if (actingAsMouse)
            {
                mouseID = -1;

                needsActivation = true;
                needsDeactivation = false;

                actingAsMouse = false;

                checkForMenuEventsHandled = 0;
            }
        }
    }
    //mutex.unlock();
}

int Utouch3DPlugin::addTouchPoint(int id, float x, float y)
{
    int result = 0;

    if (0 <= id && id < kMaxTouchPoints)
    {
        mutex.lock();
        {
            touchEvents.push(TouchPressed | id);

            touchPoints[id].push(TouchPoint(TouchCursor(id, x, 1.0f - y), TouchPressed));

            touchPointsVisible++;

            result = touchPointsVisible;

            //staticCursors[id]->update(TuioTime::getSessionTime(), x, 1.0f - y);
        }
        mutex.unlock();
    }
    else
    {
        cout << "addTouchPoint: invalid id:" << id << endl;
    }

    return result;
}

void Utouch3DPlugin::updateTouchPoint(int id, float x, float y)
{
    if (0 <= id && id < kMaxTouchPoints)
    {
        mutex.lock();
        {
            touchEvents.push(TouchMoved | id);

            //
            // Move events can be merged
            // ...
            //

            TouchPoint &p = touchPoints[id].back();

            if (touchPoints[id].size() == 0 || p.action != TouchMoved)
            {
                touchPoints[id].push(TouchPoint(TouchCursor(id, x, 1.0f - y), TouchMoved));
            }
            else
            {
                p.cursor.x = x;
                p.cursor.y = 1.0f - y;
            }

            //staticCursors[id]->update(TuioTime::getSessionTime(), x, 1.0f - y);
        }
        mutex.unlock();
    }
    else
    {
        cout << "updateTouchPoint: invalid id:" << id << endl;
    }
}

void Utouch3DPlugin::removeTouchPoint(int id, float x, float y)
{
    if (0 <= id && id < kMaxTouchPoints)
    {
        mutex.lock();
        {
            touchEvents.push(TouchReleased | id);

            touchPoints[id].push(TouchPoint(TouchCursor(id, x, 1.0f - y), TouchReleased));

            touchPointsVisible--;

            //staticCursors[id]->update(TuioTime::getSessionTime(), x, 1.0f - y);
        }
        mutex.unlock();
    }
    else
    {
        cout << "removeTouchPoint: invalid id:" << id << endl;
    }
}

void Utouch3DPlugin::syncTouchPoints()
{
    if (needsActivation || needsDeactivation)
    {
        //
        // Wait until coInteraction has made its attempt to activate itself
        // ...
        //

        return;
    }

    /*
    if (state != Active && !actingAsMouse)
    {
        return;
    }
    */

    //static unsigned int counter = 0; counter++;

    mutex.lock();
    {
        while (!touchEvents.empty())
        {
            int action = (touchEvents.front() & 0xffff0000);
            int id = (touchEvents.front() & 0x0000ffff);

            if (!touchPoints[id].empty())
            {
                TouchPoint &p = touchPoints[id].front();

                if (action == p.action) // if (action != p.action) then already handled!!!
                {
                    if (TouchPressed == action)
                    {
                        if (actingAsMouse)
                        {
                            //
                            // Nothing to do here
                            // ...
                            //
                        }
                        else if (state != Idle)
                        {
                            staticCursors[id].x = p.cursor.x;
                            staticCursors[id].y = p.cursor.y;

                            //
                            // Update interaction manager
                            //
                            intMan->addBlob(&staticCursors[id]);
                        }

                        touchPoints[id].pop();
                    }
                    else if (TouchMoved == action)
                    {
                        if (actingAsMouse)
                        {
                            if (p.cursor.id == mouseID)
                            {
                                const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();

                                int x = int((p.cursor.x) * traits->width + 0.5f);
                                int y = int((1.0f - p.cursor.y) * traits->height + 0.5f);

                                cover->handleMouseEvent(osgGA::GUIEventAdapter::DRAG, x, y); //, false);
                            }
                        }
                        else if (state != Idle)
                        {
                            staticCursors[id].x = p.cursor.x;
                            staticCursors[id].y = p.cursor.y;

                            //
                            // Update interaction manager
                            //
                            intMan->updateBlob(&staticCursors[id]);
                        }

                        touchPoints[id].pop();
                    }
                    else if (TouchReleased == action)
                    {
                        if (actingAsMouse)
                        {
                            if (p.cursor.id == mouseID)
                            {
                                cover->handleMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0); //, false);

                                //
                                // invalidate mouseID, but keep actingAsMouse!
                                //
                                mouseID = -1;
                            }
                        }
                        else if (state != Idle)
                        {
                            staticCursors[id].x = p.cursor.x;
                            staticCursors[id].y = p.cursor.y;

                            //
                            // Update interaction manager
                            //
                            intMan->removeBlob(&staticCursors[id]);
                        }

                        touchPoints[id].pop();
                    }
                }
            }

            touchEvents.pop();
        }
    }
    mutex.unlock();
}

void Utouch3DPlugin::handleTouchPressed(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    if (1 == addTouchPoint(rrev->d1, rrev->x, rrev->y))
    {
        const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();

        int x = int(rrev->x * traits->width + 0.5f);
        int y = int(rrev->y * traits->height + 0.5f);

        cover->handleMouseEvent(osgGA::GUIEventAdapter::MOVE, x, y); //, false);

        mutex.lock();
        {
            mouseID = rrev->d1;

            checkForMenuEventsHandled = 1;
        }
        mutex.unlock();
    }
}

void Utouch3DPlugin::handleTouchReleased(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    removeTouchPoint(rrev->d1, rrev->x, rrev->y);
}

void Utouch3DPlugin::handleTouchMoved(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    updateTouchPoint(rrev->d1, rrev->x, rrev->y);
}

void Utouch3DPlugin::message(int /*type*/, int /*length*/, const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    switch (rrev->type)
    {
    case RREV_TOUCHPRESS:
        handleTouchPressed(data);
        break;

    case RREV_TOUCHRELEASE:
        handleTouchReleased(data);
        break;

    case RREV_TOUCHMOVE:
        handleTouchMoved(data);
        break;
    }
}

void Utouch3DPlugin::update()
{
    if (state == Idle)
    {
        if (checkForMenuEventsHandled == 1)
        {
            cover->handleMouseEvent(osgGA::GUIEventAdapter::PUSH, osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON, 0, false);

            mutex.lock();
            {
                checkForMenuEventsHandled = 2;
            }
            mutex.unlock();
        }
        else if (needsActivation)
        {
            if (activate())
            {
                // Succeeded, ie. no menu interaction was started
                // Balance the left-button press event
                cover->handleMouseEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0); //, false);

                actingAsMouse = false;
            }
            else
            {
                actingAsMouse = true;
            }

            needsActivation = false;
        }
    }
    else if (state == Active || state == Paused || state == ActiveNotify)
    {
        if (needsDeactivation)
        {
            mutex.lock();
            {
                mouseID = -1;

                needsActivation = false;
                needsDeactivation = false;

                actingAsMouse = false;

                checkForMenuEventsHandled = 0;

                state = Idle;
            }
            mutex.unlock();
        }
    }
}

COVERPLUGIN(Utouch3DPlugin)
