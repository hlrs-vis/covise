/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _UTOUCH3D_PLUGIN_H
#define _UTOUCH3D_PLUGIN_H

/****************************************************************************\ 
 **                                              (C)2010 Anton Baumesberger  **
 **                                                                          **
 ** Description: Utouch3D Plugin                                             **
 **                                                                          **
 **                                                                          **
 ** Author: Anton Baumesberger	                                             **
 **                                                                          **
\****************************************************************************/

#include <queue>

#include "InteractionManager.h"
#include "TouchCursor.h"

#include <OpenThreads/Mutex>

#include <cover/coVRPlugin.h>

#include "OpenVRUI/coButtonInteraction.h"
#include "OpenVRUI/coInteraction.h"
#include "OpenVRUI/coInteractionManager.h"

class Utouch3DPlugin
    : public opencover::coVRPlugin,
      public covise::coInteraction
{
public:
    // faking mouse events, queuing them and dequeuing and processing them in preFrame
    static OpenThreads::Mutex mutex;

    Utouch3DPlugin();
    ~Utouch3DPlugin();

    // this will be called in PreFrame
    void preFrame();

    // override
    bool init();

    // override
    bool destroy();

    //void insertFakedMouseEvent(FakedMouseEvent* fme);

    void startTouchInteraction();
    void stopTouchInteraction();

    bool isTouchInteractionRunning();

private:
    static const unsigned int kMaxTouchPoints = 16;

    enum /*Action*/
    {
        TouchPressed = 0x00010000,
        TouchReleased = 0x00020000,
        TouchMoved = 0x00040000,
    };

    struct TouchPoint
    {
        TouchCursor cursor;
        int action; /*Action*/

        TouchPoint()
            : cursor(TouchCursor())
            , action(0)
        {
        }

        TouchPoint(const TouchCursor &cursor, int action)
            : cursor(cursor)
            , action(action)
        {
        }
    };

    int addTouchPoint(int id, float x, float y);

    void updateTouchPoint(int id, float x, float y);

    void removeTouchPoint(int id, float x, float y);

    void syncTouchPoints();

    void handleTouchPressed(const void *data);

    void handleTouchReleased(const void *data);

    void handleTouchMoved(const void *data);

    virtual void message(int type, int length, const void *data); // sendMessage...

    virtual void update(); // coInteraction interface

private:
    InteractionManager *intMan;

    std::queue<int /* Action (high word) OR'ed Touch point id (low word) */> touchEvents;

    std::queue<TouchPoint> touchPoints[kMaxTouchPoints];

    TouchCursor staticCursors[kMaxTouchPoints];

    unsigned int touchPointsVisible;

    bool needsActivation;
    bool needsDeactivation;

    bool isActivating;

    bool actingAsMouse;

    int mouseID;

    int checkForMenuEventsHandled;
};

#endif
