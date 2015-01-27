/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coButton.h>

#include <string.h>

namespace vrui
{

/** Constructor.
  @param geom,actor  coButtonGeometry and coActor are passed on to the coButton class
*/
coPushButton::coPushButton(coButtonGeometry *geom, coButtonActor *actor)
    : coButton(geom, actor)
{
}

/// Destructor.
coPushButton::~coPushButton()
{
}

// see superclass for comment
void coPushButton::miss()
{
    pressState = false;
    coButton::miss();
}

const char *coPushButton::getClassName() const
{
    return "coPushButton";
}

bool coPushButton::isOfClassName(const char *classname) const
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
void coPushButton::onRelease()
{
    pressState = false;
    myActor->buttonEvent(this);
}

// see superclass for comment
void coPushButton::onPress()
{
    pressState = true;
    myActor->buttonEvent(this);
}
}
