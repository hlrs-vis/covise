/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

namespace vrui
{

coMouseButtonInteraction::coMouseButtonInteraction(InteractionType type,
                                                   const std::string &name, InteractionPriority priority)
    : coButtonInteraction(type, name, priority)
{
}

coMouseButtonInteraction::~coMouseButtonInteraction()
{
}

void coMouseButtonInteraction::update()
{
    if (!button)
        button = vruiRendererInterface::the()->getMouseButtons();

    coButtonInteraction::update();
}
}
