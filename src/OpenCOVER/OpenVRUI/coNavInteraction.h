/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_NAVINTERACTION_H
#define CO_NAVINTERACTION_H

#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

namespace vrui
{

class OPENVRUIEXPORT coNavInteraction
    : public coInteraction
{
public:
    coNavInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Navigation);
    virtual ~coNavInteraction();

    virtual void update();
    virtual void startInteraction();
    virtual void stopInteraction();
    virtual void doInteraction();
    virtual void cancelInteraction();

protected:
    InteractionState oldState;
};
}
#endif
