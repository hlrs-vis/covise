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
    : coButtonInteraction(type, name, priority)
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

    std::cerr << "coRelativeInputInteraction::update(): is identity=" << m_matIsIdentity << ", was identity=" << m_matWasIdentity << std::endl;

    button = vruiRendererInterface::the()->getRelativeButtons();
    coButtonInteraction::update();
}

bool coRelativeInputInteraction::conditionMet() const
{
    if (m_matIsIdentity)
        return false;
    if (type == NoButton)
        return true;
    return coButtonInteraction::conditionMet();
}

bool coRelativeInputInteraction::conditionWasMet() const
{
    if (!conditionMet())
        return false;
    if (!m_matWasIdentity)
        return false;
    if (type == NoButton)
        return true;
    return coButtonInteraction::conditionWasMet();
}

}
