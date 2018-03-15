/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_RELATIVEBUTTONINTERACTION_H
#define CO_RELATIVEBUTTONINTERACTION_H

#include <OpenVRUI/coButtonInteraction.h>

namespace vrui
{

class vruiButtons;

class OPENVRUIEXPORT coRelativeButtonInteraction
    : public coButtonInteraction
{
public:
    coRelativeButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Low);
    virtual ~coRelativeButtonInteraction();

protected:
    virtual void update();
};
}
#endif
