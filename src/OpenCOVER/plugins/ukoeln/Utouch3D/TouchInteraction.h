/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TOUCHINTERACTION_H
#define TOUCHINTERACTION_H

#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

using namespace vrui;

class TouchInteraction
    : public coInteraction
{
public:
    enum RunningState
    {
        StateStarted = 0,
        StateRunning,
        StateStopped,
        StateNotRunning
    };

    TouchInteraction(InteractionType type, const std::string &name, InteractionPriority priority = NavigationHigh);
    virtual ~TouchInteraction();

    virtual void update();
    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();
    virtual void cancelInteraction();

    void requestActivation();
    void requestDeactivation();

    bool wasStarted() const
    {
        return (runningState == StateStarted);
    }
    bool isRunning() const
    {
        return (runningState == StateRunning);
    }
    bool wasStopped() const
    {
        return (runningState == StateStopped);
    }
    bool isIdle() const
    {
        return (runningState == StateNotRunning);
    }

protected:
    RunningState runningState;
    InteractionState oldState;
    bool activationRequested, deactivationRequested;
};
#endif
