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
coRotPushButton::coRotPushButton(coButtonGeometry *geometry, coRotButtonActor *actor)
    : coRotButton(geometry, actor)
{
}

/// Destructor.
coRotPushButton::~coRotPushButton()
{
}

// see superclass for comment
void coRotPushButton::miss()
{
    selectionState = false;
    pressState = false;
    updateSwitch();
}

// see superclass for comment
void coRotPushButton::onRelease()
{
    pressState = false;
}

// see superclass for comment
void coRotPushButton::onPress()
{
    pressState = true;
    myActor->buttonEvent(this);
}

const char *coRotPushButton::getClassName() const
{
    return "coRotPushButton";
}

bool coRotPushButton::isOfClassName(const char *classname) const
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
