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
    coButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Medium);
    virtual ~coButtonInteraction();

    virtual void update();
    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();
    virtual void cancelInteraction();
    virtual void resetState();

    int getWheelCount() const;

protected:
    virtual bool conditionMet() const;
    virtual bool conditionBecameMet() const;

    void updateState(vruiButtons *button);

    unsigned buttonmask = 0;
    int wheelCount = 0;

    vruiButtons *button = nullptr;
};
}
#endif
