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
#include <cover/coVRPlugin.h>
#ifdef USE_TUIOCLIENT
#include "TuioClient.h"
#include "TuioListener.h"
#include "TuioObject.h"
#endif
#include "TuioCursor.h"
#include "TuioTime.h"
#include "InteractionManager.h"
#include <OpenThreads/Mutex>
#include <queue>
#include "FakedMouseEvent.h"
#include "OpenVRUI/coButtonInteraction.h"
#include "OpenVRUI/coInteraction.h"
#include "OpenVRUI/coInteractionManager.h"
#include "TouchInteraction.h"

class Utouch3DPlugin : public opencover::coVRPlugin
#ifdef USE_TUIOCLIENT
                       ,
                       public TUIO::TuioListener
#else
                       ,
                       public coInteraction
#endif
{
public:
    Utouch3DPlugin();
    ~Utouch3DPlugin();

    // this will be called in PreFrame
    void preFrame();

    // override
    bool init();

    // override
    bool destroy();

    // faking mouse events, queuing them and dequeuing and processing them in preFrame
    static OpenThreads::Mutex mutex;

    void insertFakedMouseEvent(FakedMouseEvent *fme);

    void startTouchInteraction();
    void stopTouchInteraction();

    bool isTouchInteractionRunning();

private:
    InteractionManager *intMan;
#ifdef USE_TUIOCLIENT
    TUIO::TuioClient *tuioClient;

    TouchInteraction *touchInteraction;
    bool haveToStartInteraction, haveToStopInteraction;

    // queue for faked mouse events, handling selection
    std::queue<FakedMouseEvent *> fakedMouseEvents;

    // TuioListener interface
    void addTuioObject(TUIO::TuioObject *tobj);
    void updateTuioObject(TUIO::TuioObject *tobj);
    void removeTuioObject(TUIO::TuioObject *tobj);

    void addTuioCursor(TUIO::TuioCursor *tcur);
    void updateTuioCursor(TUIO::TuioCursor *tcur);
    void removeTuioCursor(TUIO::TuioCursor *tcur);

    void refresh(TUIO::TuioTime frameTime);
#else

    struct TouchPoint
    {
        enum
        {
            Pressed = 0x1,
            Released = 0x2,
            Moved = 0x4,
        };

        TUIO::TuioCursor cursor;
        unsigned int flags;

        TouchPoint()
            : cursor(TUIO::TuioCursor(TUIO::TuioTime::getSessionTime(), 0, -1, 0.0f, 0.0f))
            , flags(0)
        {
        }

        TouchPoint(const TUIO::TuioCursor &cursor, unsigned int flags)
            : cursor(cursor)
            , flags(flags)
        {
        }
    };

    typedef std::map<int, TouchPoint> TouchPoints;

    TouchPoints touchPoints;

    void addTouchPoint(int id, float x, float y);

    void updateTouchPoint(int id, float x, float y);

    void removeTouchPoint(int id, float x, float y);

    void syncTouchPoints();

    void handleTouchPressed(const void *data);

    void handleTouchReleased(const void *data);

    void handleTouchMoved(const void *data);

    virtual void message(int toWhom, int type, int length, const void *data); // sendMessage...

#ifndef USE_TUIOCLIENT
    virtual void update(); // coInteraction interface
#endif

    bool needsActivation;
    bool needsDeactivation;

    bool actingAsMouse;

    int mouseID;

    int checkForMenuEventsHandled;

#endif
};
#endif
