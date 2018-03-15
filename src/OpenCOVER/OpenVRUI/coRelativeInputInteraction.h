/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_RELATIVEINPUTINTERACTION_H
#define CO_RELATIVEINPUTINTERACTION_H

#include <OpenVRUI/coRelativeButtonInteraction.h>

namespace vrui
{

class OPENVRUIEXPORT coRelativeInputInteraction
    : public coRelativeButtonInteraction
{
public:
    coRelativeInputInteraction(const std::string &name, InteractionType type = NoButton, InteractionPriority priority = Medium);
    virtual ~coRelativeInputInteraction();

    virtual void update() override;

protected:
    bool conditionMet() const override;
    bool conditionBecameMet() const override;

    bool m_matIsIdentity = false, m_matWasIdentity = false;
};

}
#endif
