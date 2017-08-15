/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream>
#include <ostream>

#include "coPartnerMenuItem.h"
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenuContainer.h>
#include "coVRPluginSupport.h"
#include <OpenVRUI/coColoredBackground.h>
//#include "coAction.h"

#include <OpenVRUI/coCheckboxGroup.h>

using namespace vrui;
using namespace opencover;

coPartnerMenuItem::coPartnerMenuItem(const char *n, bool s, coCheckboxGroup *g)
    : coCheckboxMenuItem(n, s, g)
{
    //viewpoint = new
    viewpoint = new coToggleButton(new coFlatButtonGeometry("UI/eye"), this);
    viewpoint->setSize(LEFTMARGIN);
    container->addElement(viewpoint);
}

/// Destructor.
coPartnerMenuItem::~coPartnerMenuItem()
{
    delete viewpoint;
}

/** This method is called on intersections of the input device with the
  checkbox menu item.
  @return ACTION_CALL_ON_MISS
*/
int coPartnerMenuItem::hit(osg::Vec3 &, osgUtil::Hit *)
{
    container->setHighlighted(true);
    //return ACTION_CALL_ON_MISS;
    return 1;
}

/// Called when input device leaves the element.
void coPartnerMenuItem::miss()
{
    coCheckboxMenuItem::miss();
}

void coPartnerMenuItem::buttonEvent(coButton *button)
{
    if (button == viewpoint)
    {
        std::cerr << "viewpoint pressed " << viewpoint->getState() << std::endl;
    }
    else
    {
        if (cover->getPointerButton()->getState())
        {
            std::cerr << "checkbox pressed" << std::endl;
            if (group)
            {
                group->toggleCheckbox(this);
            }
            else
            {
                checkBox->setState(!checkBox->getState());
            }
            if (listener)
                listener->menuEvent(this);
        }
        else
        {
            std::cerr << "checkbox released" << std::endl;
        }
    }
}

char *coPartnerMenuItem::getClassName()
{
    return (char *)"coPartnerMenuItem";
}

bool coPartnerMenuItem::isOfClassName(char *classname)
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
            return coCheckboxMenuItem::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
