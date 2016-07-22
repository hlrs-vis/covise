/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BUTTONINTERACTION_H
#define CO_BUTTONINTERACTION_H

#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

namespace vrui
{

class vruiButtons;

class OPENVRUIEXPORT coButtonInteraction
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

    coButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Medium);
    virtual ~coButtonInteraction();

    virtual void update();
    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();
    virtual void cancelInteraction();
    virtual void resetState();

    inline bool wasStarted() const
    {
        return (runningState == StateStarted);
    }
    inline bool isRunning() const
    {
        return (runningState == StateRunning);
    }
    inline bool wasStopped() const
    {
        return (runningState == StateStopped);
    }
    inline bool isIdle() const
    {
        return (runningState == StateNotRunning);
    }

    int getWheelCount()
    {
        return wheelCount;
    }

protected:
    void updateState(vruiButtons *button);

    RunningState runningState;
    InteractionState oldState;
    int wheelCount;
    unsigned buttonmask;

    vruiButtons *button;
};
}
#endif
