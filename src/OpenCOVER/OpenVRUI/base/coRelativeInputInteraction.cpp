/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRelativeInputInteraction.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtons.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

coRelativeInputInteraction::coRelativeInputInteraction(const string &name, InteractionType type, InteractionPriority priority)
    : coRelativeButtonInteraction(type, name, priority)
{
}

coRelativeInputInteraction::~coRelativeInputInteraction()
{
    //fprintf(stderr,"coRelativeInputInteraction::~coRelativeInputInteraction\n");
    coInteractionManager::the()->unregisterInteraction(this);
}

void coRelativeInputInteraction::update()
{
    m_matWasIdentity = m_matIsIdentity;
    m_matIsIdentity = vruiRendererInterface::the()->getRelativeMatrix()->isIdentity();

    coRelativeButtonInteraction::update();
}

bool coRelativeInputInteraction::conditionMet() const
{
    if (m_matIsIdentity)
        return false;
    if (type == NoButton)
        return true;
    return coRelativeButtonInteraction::conditionMet();
}

bool coRelativeInputInteraction::conditionBecameMet() const
{
    if (!conditionMet())
        return false;
    if (!m_matWasIdentity)
        return false;
    if (type == NoButton)
        return true;
    return coRelativeButtonInteraction::conditionBecameMet();
}

}
