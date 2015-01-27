/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_TRACKERBUTTONINTERACTION_H
#define CO_TRACKERBUTTONINTERACTION_H

#include <OpenVRUI/coButtonInteraction.h>
#include <OpenVRUI/coInteractionManager.h>

namespace vrui
{

class vruiButtons;

class OPENVRUIEXPORT coTrackerButtonInteraction
    : public coButtonInteraction
{
public:
    coTrackerButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Medium);
    virtual ~coTrackerButtonInteraction();

protected:
    virtual void update();
};
}
#endif
