/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// TapHandler.h
// Simple discrete gesture handler to detect taps

#pragma once

#include "TouchHandler.h"

class TapHandler
    : public TouchHandler
{
public:
    TapHandler()
    {
    }

    virtual ~TapHandler()
    {
    }

    virtual void init();

    virtual void finish();

    virtual void onTouchPressed(Touches const &touches, Touch const &reason);

    virtual void onTouchesMoved(Touches const &touches);

    virtual void onTouchReleased(Touches const &touches, Touch const &reason);

    virtual void onUpdate();

private:
    void setRequiredTapCount(int count);

    void reset();

private:
    double lastPressTime;
    double lastReleaseTime;
    int currentTapCount;
    int requiredTapCount;
};

inline void TapHandler::setRequiredTapCount(int count)
{
    requiredTapCount = count;
}
