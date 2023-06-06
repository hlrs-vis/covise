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
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

#include <cover/coVRConfig.h>
#include <cover/input/coMousePointer.h>
#include <cover/input/input.h>

using namespace TUIO;
using namespace opencover;

OpenThreads::Mutex Utouch3DPlugin::mutex;

#ifndef USE_TUIOCLIENT

enum rrxevent_types
{
    RREV_NONE = 0,
    RREV_KEYPRESS,
    RREV_KEYRELEASE,
    RREV_BTNPRESS,
    RREV_BTNRELEASE,
    RREV_MOTION,
    RREV_WHEEL,
    RREV_RESIZE,
    RREV_TOUCHPRESS,
    RREV_TOUCHRELEASE,
    RREV_TOUCHMOVE,
};

struct rrxevent
{
    int type;
    float x;
    float y;
    int d1;
    int d2;
};

#endif

Utouch3DPlugin::Utouch3DPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
#ifndef USE_TUIOCLIENT
, coInteraction(coInteraction::ButtonA, "Utouch3D", coInteraction::NavigationHigh)
, needsActivation(false)
, needsDeactivation(false)
, actingAsMouse(false)
, mouseID(-1)
, checkForMenuEventsHandled(0)
#endif
{
    //fprintf(stderr,"Utouch3DPlugin::Utouch3DPlugin\n");;
}

// this is called if the plugin is removed at runtime
Utouch3DPlugin::~Utouch3DPlugin()
{
    fprintf(stderr, "Utouch3DPlugin::~Utouch3DPlugin\n");

#ifndef USE_TUIOCLIENT
    vrui::coInteractionManager::the()->unregisterInteraction(this);
#endif
}

bool Utouch3DPlugin::init()
{
    std::cout << "Utouch3DPlugin... init()" << std::endl;

#ifdef USE_TUIOCLIENT
    int port = 50096;
    tuioClient = new TuioClient(port);

    tuioClient->addTuioListener(this);

    // call connect() without arguments, running in background! ( connect(true) is blocking mode )
    tuioClient->connect();

    touchInteraction = new TouchInteraction(vrui::coInteraction::ButtonA, "uTouch3D", vrui::coInteraction::NavigationHigh);
    vrui::coInteractionManager::the()->registerInteraction(touchInteraction);

    haveToStartInteraction = false;
    haveToStopInteraction = false;
#else
    vrui::coInteractionManager::the()->registerInteraction(this);
#endif

    intMan = new InteractionManager(this, cover);

    return true;
}

bool Utouch3DPlugin::destroy()
{
#ifdef USE_TUIOCLIENT
    tuioClient->disconnect();
    tuioClient->removeTuioListener(this);
    delete tuioClient;

    delete touchInteraction;
#endif

    delete intMan;

    return true;
}

void Utouch3DPlugin::insertFakedMouseEvent(FakedMouseEvent *fme)
{
#ifdef USE_TUIOCLIENT
    mutex.lock();
    fakedMouseEvents.push(fme);
    mutex.unlock();
#endif
}

void Utouch3DPlugin::startTouchInteraction()
{
#ifdef USE_TUIOCLIENT
    mutex.lock();
    haveToStartInteraction = true;
    mutex.unlock();
#endif
}

void Utouch3DPlugin::stopTouchInteraction()
{
#ifdef USE_TUIOCLIENT
    mutex.lock();
    haveToStopInteraction = true;
    mutex.unlock();
#endif
}

bool Utouch3DPlugin::isTouchInteractionRunning()
{
#ifdef USE_TUIOCLIENT
    bool result;

    mutex.lock();
    result = !touchInteraction->isIdle();
    mutex.unlock();

    return result;
#else
    return false;
#endif
}

/**
 * dequeues @code fakedMouseEvents and processes one event per call
 * using a mutex
 */
void Utouch3DPlugin::preFrame()
{
#ifndef USE_TUIOCLIENT

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
        if (touchPoints.empty())
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

#else

    mutex.lock();

    if (/*fakedMouseEvents.empty() && */ haveToStartInteraction)
    {
        touchInteraction->requestActivation();
        haveToStartInteraction = false;
        //            cout << "preFrame().touchInteraction->isRunning(): " << touchInteraction->isRunning() << endl;
    }

    if (!fakedMouseEvents.empty())
    {
        FakedMouseEvent *e = dynamic_cast<FakedMouseEvent *>(fakedMouseEvents.front());
        fakedMouseEvents.pop();
        opencover::Input::instance()->mouse()->handleEvent(e->getEventType(), e->getXPos(), e->getYPos());
        //            cout << "preFrame() processed FME.type " << e->getEventType() << " .x: " << e->getXPos() << " .y: " << e->getYPos() << endl;

        delete e;
    }

    if (haveToStopInteraction)
    {
        touchInteraction->requestDeactivation();
        haveToStopInteraction = false;
    }

    mutex.unlock();

//std::cout << "Utouch3DPlugin... current navigationMode: " << coVRNavigationManager::instance()->getMode() << std::endl;
#endif
}

#ifdef USE_TUIOCLIENT

void Utouch3DPlugin::addTuioObject(TuioObject *tobj)
{

    std::cout << "set obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ") " << tobj->getX() << " " << tobj->getY() << " " << tobj->getAngle()
              << " " << tobj->getMotionSpeed() << " " << tobj->getRotationSpeed() << " " << tobj->getMotionAccel() << " " << tobj->getRotationAccel() << std::endl;
}

void Utouch3DPlugin::updateTuioObject(TuioObject *tobj)
{

    std::cout << "set obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ") " << tobj->getX() << " " << tobj->getY() << " " << tobj->getAngle()
              << " " << tobj->getMotionSpeed() << " " << tobj->getRotationSpeed() << " " << tobj->getMotionAccel() << " " << tobj->getRotationAccel() << std::endl;
}

void Utouch3DPlugin::removeTuioObject(TuioObject *tobj)
{

    std::cout << "del obj " << tobj->getSymbolID() << " (" << tobj->getSessionID() << ")" << std::endl;
}

void Utouch3DPlugin::addTuioCursor(TuioCursor *tcur)
{
    //std::cout << "Utouch3DPlugin::addTuioCursor" << std::endl;
    //std::cout << "add cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ") " << tcur->getX() << " " << tcur->getY() << std::endl;

    intMan->addBlob(tcur);
}

void Utouch3DPlugin::updateTuioCursor(TuioCursor *tcur)
{
    //std::cout << "Utouch3DPlugin::updateTuioCursor" << std::endl;
    //std::cout << "set cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ") " << tcur->getX() << " " << tcur->getY()
    //          << " " << tcur->getMotionSpeed() << " " << tcur->getMotionAccel() << " " << std::endl;

    intMan->updateBlob(tcur);
}

void Utouch3DPlugin::removeTuioCursor(TuioCursor *tcur)
{
    // std::cout << "Utouch3DPlugin::removeTuioCursor" << std::endl;
    //std::cout << "del cur " << tcur->getCursorID() << " (" <<  tcur->getSessionID() << ")" << std::endl;

    intMan->removeBlob(tcur);
}

void Utouch3DPlugin::refresh(TuioTime frameTime)
{
}

#else

void Utouch3DPlugin::addTouchPoint(int id, float x, float y)
{
    //mutex.lock();
    {
        TouchPoints::iterator pos = touchPoints.find(id);

        if (pos == touchPoints.end())
        {
            touchPoints[id] = TouchPoint(TuioCursor(TuioTime::getSessionTime(), 0, id, x, y), TouchPoint::Pressed);
        }
    }
    //mutex.unlock();
}

void Utouch3DPlugin::updateTouchPoint(int id, float x, float y)
{
    //mutex.lock();
    {
        TouchPoints::iterator pos = touchPoints.find(id);

        if (pos != touchPoints.end())
        {
            TouchPoint &p = pos->second;

            p.cursor.update(TuioTime::getSessionTime(), x, y);
            p.flags |= TouchPoint::Moved;
        }
    }
    //mutex.unlock();
}

void Utouch3DPlugin::removeTouchPoint(int id, float x, float y)
{
    //mutex.lock();
    {
        TouchPoints::iterator pos = touchPoints.find(id);

        if (pos != touchPoints.end())
        {
            TouchPoint &p = pos->second;

            p.cursor.update(TuioTime::getSessionTime(), x, y);
            p.flags |= TouchPoint::Released; // mark as released; will be removed later
        }
    }
    //mutex.unlock();
}

void Utouch3DPlugin::syncTouchPoints()
{
    //mutex.lock();
    {
        for (TouchPoints::iterator it = touchPoints.begin(); it != touchPoints.end();)
        {
            TouchPoint &p = it->second;

            if (p.flags & TouchPoint::Pressed)
            {
                if (state != Idle && !actingAsMouse)
                {
                    intMan->addBlob(&p.cursor);

                    p.flags &= ~TouchPoint::Pressed;
                }
                else if (actingAsMouse)
                {
                    p.flags &= ~TouchPoint::Pressed;
                }

                ++it;
            }
            else if (p.flags & TouchPoint::Moved)
            {
                if (state != Idle && !actingAsMouse)
                {
                    intMan->updateBlob(&p.cursor);
                }
                else if (actingAsMouse)
                {
                    if (p.cursor.getCursorID() == mouseID)
                    {
                        const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();

                        int x = int((p.cursor.getX()) * traits->width + 0.5f);
                        int y = int((1.0f - p.cursor.getY()) * traits->height + 0.5f);

                        opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::DRAG, x, y, false);
                    }
                }

                p.flags &= ~TouchPoint::Moved;

                ++it;
            }
            else if (p.flags & TouchPoint::Released)
            {
                if (state != Idle && !actingAsMouse)
                {
                    intMan->removeBlob(&p.cursor);

                    //p.flags &= ~TouchPoint::Released;
                }
                else if (actingAsMouse)
                {
                    if (p.cursor.getCursorID() == mouseID)
                    {
                        opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0, false);

                        mouseID = -1; // invalidate mouseID, but keep actingAsMouse!
                    }

                    //p.flags &= ~TouchPoint::Released;
                }

                //if ((p.flags & TouchPoint::Released) == 0)
                {
                    touchPoints.erase(it++); // remove touch point from list and advance
                }
                //else
                //{
                //    ++it;
                //}
            }
            else
            {
                ++it;
            }
        }
    }
    //mutex.unlock();
}

void Utouch3DPlugin::handleTouchPressed(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    if (touchPoints.empty())
    {
        mouseID = rrev->d1;

        const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();

        int x = int(rrev->x * traits->width + 0.5f);
        int y = int(rrev->y * traits->height + 0.5f);

        opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::MOVE, x, y, false);

        checkForMenuEventsHandled = 1;
    }

    addTouchPoint(rrev->d1, rrev->x, 1.f - rrev->y);
}

void Utouch3DPlugin::handleTouchReleased(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    removeTouchPoint(rrev->d1, rrev->x, 1.f - rrev->y);
}

void Utouch3DPlugin::handleTouchMoved(const void *data)
{
    const struct rrxevent *rrev = (const struct rrxevent *)data;

    updateTouchPoint(rrev->d1, rrev->x, 1.f - rrev->y);
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

#ifndef USE_TUIOCLIENT
void Utouch3DPlugin::update()
{
    if (state == Idle)
    {
        if (checkForMenuEventsHandled == 1)
        {
            opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::PUSH, osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON, 0, false);

            checkForMenuEventsHandled = 2;
        }
        else if (needsActivation)
        {
            if (activate())
            {
                // Succeeded, ie. no menu interaction was started
                // First balance the left-button press event
                opencover::Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0, false);

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
            mouseID = -1;

            needsActivation = false;
            needsDeactivation = false;

            actingAsMouse = false;

            checkForMenuEventsHandled = 0;

            state = Idle;
        }
    }
}
#endif

#endif

COVERPLUGIN(Utouch3DPlugin)
