/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TouchInteraction.h

#pragma once

#include "Touch.h"

#include "OpenVRUI/coInteraction.h"
#include "OpenVRUI/coInteractionManager.h"

#include <cstdio>
#include <deque>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

#define DEBUG_TOUCH_INTERACTION 0

//
// This class separates touch events interacting with the menu from touch events which are passed
// to the different handlers via the Plugin
//
class TouchInteraction
    : public vrui::coInteraction
{
public:
    TouchInteraction();

    virtual ~TouchInteraction();

protected:
    // Call this method in the plugin's init() method
    void onInit();

    // Call this method in the plugin's destroy() method
    void onDestroy();

    // Call this method in the plugin's preFrame() method
    void onPreFrame();

    // Call this method in the plugin's message() method
    void onMessage(int type, int length, void const *data);

    //
    // To be implemented...
    //

    //
    // Called when a new touch is recognized
    // Returns whether the message has been processed
    //
    virtual bool onTouchPressed(Touches const &touches, Touch const &reason) = 0;

    //
    // Called when a touch has moved
    // Returns whether the message has been processed
    //
    virtual bool onTouchesMoved(Touches const &touches) = 0;

    //
    // Called when a touch is released
    // Returns whether the message has been processed
    //
    // NOTE: id is actually invalid and touches might be empty!
    //
    virtual bool onTouchReleased(Touches const &touches, Touch const &reason) = 0;

    //
    // Called when the touch interaction has been canceled
    // Returns whether the message has been processed
    //
    virtual bool onTouchesCanceled() = 0;

    virtual void onTouchesBeingPressed(Touches const &touches) = 0;

    virtual void onTouchesBeingReleased(Touches const &touches) = 0;

private:
    enum TIState
    {
        TIS_Idle,
        TIS_Activating,
        TIS_Active,
        TIS_ActiveMouse,
        TIS_Inactive
    };

    // Returns the system time in seconds
    double time() const;

    void onTouchPressMessage(Touch const &touch);

    void onTouchMoveMessage(Touch const &touch);

    void onTouchReleaseMessage(Touch const &touch);

    void onTouchCancelMessage();

    void sendMousePressEvent(int /*id*/, float x, float y);

    void sendMouseMoveEvent(int /*id*/, float x, float y);

    void sendMouseReleaseEvent(int /*id*/);

    void sendPendingTouchEvents();

    void sendPendingMouseEvents();

    TIState getTIState() const;

    void setTIState(TIState tiState);

    Touches const &getTouches() const;

    Touches &getTouches();

    void addTouch(Touch const &touch);

    void updateTouch(Touch const &touch);

    void removeTouch(Touch const &touch);

    //
    // from coInteraction
    //

    virtual void update();

private:
    typedef std::deque<Touch> TouchEvents;

    // Current interaction state
    TIState tiState_;
    // ID of touch used for mouse input
    int mouseID;
    // Up-to-date list of touch points
    Touches touches_;
    // List of incoming touch events
    TouchEvents touchEvents;
    // Whether an unhandled touch cancel message was received
    bool touchCancel;
};

inline TouchInteraction::TIState TouchInteraction::getTIState() const
{
    return tiState_;
}

inline void TouchInteraction::setTIState(TouchInteraction::TIState tiState)
{
    if (getTIState() == TIS_Active)
    {
        //      std::cout << "TouchInteraction::cancel!!!" << std::endl;
        cancelInteraction(); // coInteraction is active: cancel!
    }

#if 0 && DEBUG_TOUCH_INTERACTION
    switch (tiState)
    {
    case TIS_Idle:
        fprintf(stderr, "New interaction state: IDLE\n");
        break;
    case TIS_Activating:
        fprintf(stderr, "New interaction state: ACTIVATING\n");
        break;
    case TIS_Active:
        fprintf(stderr, "New interaction state: ACTIVE\n");
        break;
    case TIS_ActiveMouse:
        fprintf(stderr, "New interaction state: ACTIVE-MOUSE\n");
        break;
    case TIS_Inactive:
        fprintf(stderr, "New interaction state: INACTIVE\n");
        break;
    default:
        fprintf(stderr, "New interaction state: <<< UNKNOWN >>>\n");
        break;
    }
#endif

    tiState_ = tiState;
}

inline Touches const &TouchInteraction::getTouches() const
{
    return touches_;
}

inline Touches &TouchInteraction::getTouches()
{
    return touches_;
}

inline void TouchInteraction::addTouch(Touch const &touch)
{
#if DEBUG_TOUCH_INTERACTION
    Touches::iterator it = touches_.find(touch.id);

    if (it != touches_.end())
    {
        fprintf(stderr, "TouchInteraction::addTouch: touch point [%d] already in list!\n", touch.id);
    }
    else
#endif
    {
        touches_[touch.id] = touch;
    }
}

inline void TouchInteraction::updateTouch(Touch const &touch)
{
#if DEBUG_TOUCH_INTERACTION
    Touches::iterator it = touches_.find(touch.id);

    if (it == touches_.end())
    {
        fprintf(stderr, "TouchInteraction::updateTouch: touch point [%d] not in list!\n", touch.id);
    }
    else
#endif
    {
        touches_[touch.id] = touch; //it->second = touch;
    }
}

inline void TouchInteraction::removeTouch(Touch const &touch)
{
#if DEBUG_TOUCH_INTERACTION
    Touches::iterator it = touches_.find(touch.id);

    if (it == touches_.end())
    {
        fprintf(stderr, "TouchInteraction::updateTouch: touch point [%d] not in list!\n", touch.id);
    }
    else
#endif
    {
        touches_.erase(touch.id); //touches_.erase(it);
    }
}
