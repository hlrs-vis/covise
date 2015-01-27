/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButton.h>

#include <string.h>

namespace vrui
{

// ------------------------------------------------------------------
/** Constructor.
  @param geom,actor  coButtonGeometry and coActor are passed on to the coButton class
*/
coToggleButton::coToggleButton(coButtonGeometry *geom, coButtonActor *actor)
    : coButton(geom, actor)
{
    wasReleased = true;
}

/// Destructor.
coToggleButton::~coToggleButton()
{
}

const char *coToggleButton::getClassName() const
{
    return "coToggleButton";
}

bool coToggleButton::isOfClassName(const char *classname) const
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
            return coButton::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}

// see superclass for comment
void coToggleButton::onRelease()
{
    if (!wasReleased)
    {
        pressState = !pressState;
        myActor->buttonEvent(this);
    }
    wasReleased = true;
}

// see superclass for comment
void coToggleButton::onPress()
{
    if (!wasReleased)
        return;
    wasReleased = false;
}
}
