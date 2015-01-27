/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MOUSEBUTTONINTERACTION_H
#define CO_MOUSEBUTTONINTERACTION_H

#include <OpenVRUI/coButtonInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

namespace vrui
{

class vruiButtons;

class OPENVRUIEXPORT coMouseButtonInteraction
    : public coButtonInteraction
{
public:
    coMouseButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Low);
    virtual ~coMouseButtonInteraction();

protected:
    virtual void update();
};
}
#endif
