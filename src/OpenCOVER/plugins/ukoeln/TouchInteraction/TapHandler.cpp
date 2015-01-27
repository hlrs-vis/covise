/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TapHandler.cpp

#include "TapHandler.h"

#include <iostream>

#define DELTA 0.3 // seconds

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::init()
{
    setState(Possible);
    setRequiredTapCount(2);
    reset();
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::finish()
{
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::onTouchPressed(Touches const &touches, Touch const &reason)
{
    if (getState() == Possible)
    {
        // Save timestamp
        lastPressTime = reason.time;

        if (touches.size() > 1 || (lastReleaseTime != 0.0 && lastPressTime - lastReleaseTime > DELTA))
        {
            setState(Failed);
            reset();
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::onTouchesMoved(Touches const &touches)
{
    //
    // Actually move messages are ignored by the tap handler
    // But move messages arrive quite often and we might check here if the time-out has
    // expired and change state to Failed in this case.
    //

    if (getState() == Possible)
    {
        Touch const &touch = touches.begin()->second; // There is at least one touch point in the list

        if (lastPressTime != 0.0f && (touch.time - lastPressTime > DELTA))
        {
            setState(Failed);
            reset();
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::onTouchReleased(Touches const &touches, Touch const &reason)
{
    if (getState() == Possible)
    {
        // Save timestamp
        lastReleaseTime = reason.time;
        // Increment tap counter
        currentTapCount++;

        // First check if the time-out has expired
        // Set the state to <Failed> in this case, ie. the gesture was not recognized
        if (lastReleaseTime - lastPressTime > DELTA)
        {
            setState(Failed);
            reset();
        }

        // Ok, time-out has not expired yet.
        // If the required number of taps has been recognized, set the state to <Recognized>
        // ie. the onUpdate() method will be called by the plugin
        else if (currentTapCount == requiredTapCount)
        {
            setState(Recognized);
            reset();
        }
    }

    // Reset the handler state if there are no more touches active
    if (getState() != Recognized && touches.empty())
    {
        setState(Possible);
        reset();
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::onUpdate()
{
    // This is called by the plugin if the handler's state is <Recognized>
    // Set the state to <Possible> to signal the plugin, the tap was handled.
    setState(Possible);
    reset();

    //
    // TODO:
    // Something useful...
    //

    static unsigned int counter = 0;
    counter++;
    std::cout << counter << ": Tap!" << std::endl;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void TapHandler::reset()
{
    lastPressTime = 0.0;
    lastReleaseTime = 0.0;
    currentTapCount = 0;
}
