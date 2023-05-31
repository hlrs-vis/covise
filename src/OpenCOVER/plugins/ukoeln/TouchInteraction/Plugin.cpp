/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Plugin.cpp

#include "Plugin.h"
#include "ScreenSpaceHandler.h"
#include "Utouch3DHandler.h"
#include "TapHandler.h"

#include <config/CoviseConfig.h>

#include <cstdio>
#include <vector>
#include <list>
#include <memory>
#include <map>
#include <queue>
#include <sstream>

#define MULTIPLE_HANDLERS 1

#define UNUSED(var) ((void)(var))

//
// Adding a new handler
//
// 1. Derive from TouchHandler and implement the following routines
//          onTouchPressed
//          onTouchesMoved
//          onTouchReleased
//          onUpdate
// 2. Add a new if-statement in createHandler() below
// 3. Modify the config an start the plugin
//
// Examples: TapHandler.h/cpp and ScreenHandler.h/cpp
//

typedef std::map<std::string /*name*/, TouchHandler *> HandlerList;
typedef std::map<int /*number of touches*/, TouchHandler *> HandlerMap;

static HandlerList handlerList;
static HandlerMap handlerMap;
static TouchHandler *currentHandler = 0;

inline std::string int_to_string(int n)
{
    std::ostringstream out;
    out << n;
    return out.str();
}

inline int string_to_int(std::string const &s)
{
    std::stringstream str(s);
    int out = 0;
    str >> out;
    return out;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
inline TouchHandler *createHandler(std::string const &name)
{
    //  fprintf(stderr, "Creating touch handler: %s\n", name.c_str());

    if (name == "ScreenSpace")
        return new ScreenSpaceHandler;
    if (name == "Utouch3D")
        return new Utouch3DHandler;

    return 0;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
inline bool addHandler(int numTouches, std::string const &name, bool force = false)
{
    TouchHandler *handler = 0;

    HandlerList::iterator itList = handlerList.find(name);
    if (itList != handlerList.end())
    {
        handler = itList->second;
    }
    else
    {
        handler = createHandler(name);
        if (handler)
        {
            handler->init();
            handlerList[name] = handler;
        }
    }

    if (handler == 0 && !force)
    {
        fprintf(stderr, "TouchInteraction plugin: could not create handler \"%s\"\n", name.c_str());
        return false;
    }

    // NOTE:
    // handler might be NULL!
    handlerMap[numTouches] = handler;

    return true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
Plugin::Plugin()
: opencover::coVRPlugin(COVER_PLUGIN_NAME)
, TouchInteraction()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
Plugin::~Plugin()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::init()
{
    std::string config("COVER.Plugin.TouchInteraction");

#if MULTIPLE_HANDLERS

    std::string strMaxHandlers = covise::coCoviseConfig::getEntry("maxHandlers", config, "10");
    std::string strHandler0 = covise::coCoviseConfig::getEntry("handler0", config, "");

    // Always add a handler for 0 touches (default)
    // Might be NULL
    addHandler(0, strHandler0, true);

    for (int n = 1; n < string_to_int(strMaxHandlers); ++n)
    {
        std::string name = std::string("handler") + int_to_string(n);
        std::string entry = covise::coCoviseConfig::getEntry(name, config, "");

        if (entry != "")
        {
            addHandler(n, entry, false);
        }
    }

#else

    currentHandler = new ScreenSpaceHandler();
    currentHandler->init();

#endif

    TouchInteraction::onInit();

    return true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::destroy()
{
    TouchInteraction::onDestroy();

#if MULTIPLE_HANDLERS

    // Delete all touch handlers
    for (HandlerList::iterator it = handlerList.begin(); it != handlerList.end(); ++it)
    {
        TouchHandler *p = it->second;
        if (p != 0)
        {
            p->finish(); // Clean up!
            delete p;
        }
    }

#else

    currentHandler->finish();
    delete currentHandler;

#endif

    return true;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::preFrame()
{
    TouchInteraction::onPreFrame();

    if (currentHandler && (currentHandler->getState() == TouchHandler::Recognized || currentHandler->getState() == TouchHandler::Changed))
    {
        currentHandler->onUpdate();
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::message(int toWhom, int type, int length, void const *data)
{
    TouchInteraction::onMessage(type, length, data);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::onTouchesBeingPressed(Touches const &touches)
{
#if MULTIPLE_HANDLERS

    TouchHandler *nextHandler = 0;

    HandlerMap::iterator it = handlerMap.find(int(touches.size()) + 1);
    if (it != handlerMap.end())
    {
        nextHandler = it->second;
    }
    else
    {
        nextHandler = handlerMap[0];
    }

    if (currentHandler != nextHandler)
    {
        // Send release events to the current handler -- if any
        sendReleaseEvents(touches);

        currentHandler = nextHandler;

        // Send press events to the next handler -- if any
        sendPressEvents(touches);
    }

#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::onTouchPressed(Touches const &touches, Touch const &reason)
{
    if (currentHandler)
    {
        currentHandler->onTouchPressed(touches, reason);
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::onTouchesMoved(Touches const &touches)
{
    if (currentHandler)
    {
        currentHandler->onTouchesMoved(touches);
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::onTouchesBeingReleased(Touches const &touches)
{
#if MULTIPLE_HANDLERS

    TouchHandler *nextHandler = 0;

    if (touches.size() > 1)
    {
        HandlerMap::iterator it = handlerMap.find(int(touches.size()) - 1);
        if (it != handlerMap.end())
        {
            nextHandler = it->second;
        }
    }

    if (nextHandler == 0)
    {
        nextHandler = handlerMap[0];
    }

    if (currentHandler != nextHandler)
    {
        // Send release events to the current handler -- if any
        sendReleaseEvents(touches);

        currentHandler = nextHandler;

        // Send press events to the next handler -- if any
        sendPressEvents(touches);
    }

#endif
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::onTouchReleased(Touches const &touches, Touch const &reason)
{
    if (currentHandler)
    {
        currentHandler->onTouchReleased(touches, reason);
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
bool Plugin::onTouchesCanceled()
{
    if (currentHandler)
    {
        currentHandler->onTouchesCanceled();
        return true;
    }

    return false;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::sendPressEvents(Touches const &touches)
{
    if (currentHandler != 0)
    {
        Touches list;

        for (Touches::const_iterator it = touches.begin(); it != touches.end(); ++it)
        {
            Touch const &reason = it->second;

            list[reason.id] = reason;

            currentHandler->onTouchPressed(list, reason);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void Plugin::sendReleaseEvents(Touches const &touches)
{
    if (currentHandler != 0)
    {
        Touches list(touches);

        while (!list.empty())
        {
            Touches::iterator it = list.begin();

            Touch reason = it->second;

            list.erase(it);

            currentHandler->onTouchReleased(list, reason);
        }
    }
}

COVERPLUGIN(Plugin)
