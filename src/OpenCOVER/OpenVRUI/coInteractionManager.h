/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_INTERACTIONMANAGER
#define CO_INTERACTIONMANAGER

#include <OpenVRUI/coInteraction.h>
#include <list>

namespace vrui
{

class OPENVRUIEXPORT coInteractionManager
{
public:
    coInteractionManager();
    virtual ~coInteractionManager();

    bool update();
    void registerInteraction(coInteraction *);
    void unregisterInteraction(coInteraction *);
    bool isOneActive(coInteraction::InteractionType type);
    bool isOneActive(coInteraction::InteractionGroup group);

    static coInteractionManager *the();

private:
    // list of registered interactions
    std::list<coInteraction *> interactionStack[coInteraction::NumInteractorTypes];
    // list of active but unregistered interactions
    std::list<coInteraction *> activeUnregisteredInteractions[coInteraction::NumInteractorTypes];

protected:
    static coInteractionManager *im;
};
}
#endif
