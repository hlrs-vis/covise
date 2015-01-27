/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRotButton.h>

namespace vrui
{

/** Constructor.
  @param geom,actor  coButtonGeometry and coActor are passed on to the coRotButton class
*/
coRotToggleButton::coRotToggleButton(coButtonGeometry *geometry, coRotButtonActor *actor)
    : coRotButton(geometry, actor)
{
}

/// Destructor.
coRotToggleButton::~coRotToggleButton()
{
}

// see superclass for comment
void coRotToggleButton::onPress()
{
    pressState = !pressState;
    myActor->buttonEvent(this);
}

const char *coRotToggleButton::getClassName() const
{
    return "coRotToggleButton";
}

bool coRotToggleButton::isOfClassName(const char *classname) const
{
    // paranoia makes us mistrust the string library and check for NULL.
    if (classname && getClassName())
    {
        // check for identity
        if (!strcmp(classname, getClassName()))
        { // we are the one
            return true;
        }
        else
        { // we are not the wanted one. Branch up to parent class
            return coRotButton::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
