/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TouchHandler.h

#pragma once

#include "Touch.h"

//
// TouchHandler
//
// Derived from this class to implement your own touch handler, managed by the plugin.
// An active touch handler received touch press, move and release events from the plugin
// and can take actions such as updating the OpenCOVER view matrix.
//
class TouchHandler
{
public:
    enum State
    {
        // The touch handler is ready (for recognizing a gesture).
        // Initially every touch handler is in this state.
        Possible,
        // The (continuous) touch handler has recognized a gesture.
        // E.g. a swipe gesture might get into this state if there is a single touch point currently
        // pressed
        Began,
        // Signals the plugin that the onUpdate() method of the touch handler
        // should be called the next time.
        Changed,
        // The continuous gesture has ended
        Ended,
        // The touch handler has failed to recognize a gesture (e.g. the handler only
        // handles 2 touch points but 3 are already recognized)
        Failed,
        // Discrete gestures (see e.g. TapHandler) do not change its state from
        // Began -> Changed -> Began -> Changed -> ... -> Changed -> Ended/Possible -> ...
        // but instead from Possible -> Recognized -> Possible -> ...
        Recognized = Ended
    };

    TouchHandler()
    {
        setState(Possible);
    }

    virtual ~TouchHandler()
    {
    }

    virtual void init()
    {
    }

    virtual void finish()
    {
    }

    //
    // This is called when a new touch point is recognized
    //
    // <touches>    The list of all currently recognized touch points
    // <reason>     The point causing the function call (ie. the new touch point)
    //              NOTE: This point is included in <touches>
    //
    virtual void onTouchPressed(Touches const &touches, Touch const &reason) = 0;

    //
    // This is called when one or more touch points have been moved
    //
    // <touches>    The list of all currently recognized touch points
    //
    virtual void onTouchesMoved(Touches const &touches) = 0;

    //
    // This is called when a touch point is released
    //
    // <touches>    The list of all currently recognized touch points
    // <reason>     The point causing the function call (ie. the touch point which has been released)
    //              NOTE: This point is NOT included in <touches>
    //
    virtual void onTouchReleased(Touches const &touches, Touch const &reason) = 0;

    //
    // Called when the touch interaction process should be canceled
    // XXX: Currently not implemented, therefore a default implemention
    //
    virtual void onTouchesCanceled()
    {
    }

    //
    // Called whenever the handler needs to update anything not part of the plugin.
    // This is only called by the plugin if the handler is in state <Changed> or <Recognized>
    //
    virtual void onUpdate() = 0;

    //
    // Returns the current state of the touch handler
    //
    State getState() const;

    //
    // Sets the state of the touch handler
    //
    void setState(State state);

private:
    State state_;
};

inline TouchHandler::State TouchHandler::getState() const
{
    return state_;
}

inline void TouchHandler::setState(TouchHandler::State state)
{
    state_ = state;
}
