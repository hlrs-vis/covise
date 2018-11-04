/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRelativeButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

namespace vrui
{

coRelativeButtonInteraction::coRelativeButtonInteraction(InteractionType type,
                                                   const std::string &name, InteractionPriority priority)
    : coButtonInteraction(type, name, priority)
{
}

coRelativeButtonInteraction::~coRelativeButtonInteraction()
{
}

void coRelativeButtonInteraction::update()
{
    button = vruiRendererInterface::the()->getRelativeButtons();
    coButtonInteraction::update();
}
}
