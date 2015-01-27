/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COMBINEDBUTTONINTERACTION_H
#define CO_COMBINEDBUTTONINTERACTION_H

#include "coButtonInteraction.h"
#include "coInteractionManager.h"

namespace vrui
{

class vruiButtons;
class vruiMatrix;

class OPENVRUIEXPORT coCombinedButtonInteraction
    : public coButtonInteraction
{
public:
    coCombinedButtonInteraction(InteractionType type, const std::string &name, InteractionPriority priority = Medium);
    virtual ~coCombinedButtonInteraction();
    virtual void setHitByMouse(bool);
    virtual bool isMouse() const;
    virtual vruiMatrix *getHandMatrix() const;
    virtual bool is2D() const;

protected:
    virtual void update();
    vruiButtons *mousebutton;
    bool mouse;
};
}
#endif
