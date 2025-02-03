/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream>
#include <ostream>

#include "vvVruiPartnerMenuItem.h"
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenuContainer.h>
#include "vvPluginSupport.h"
#include <OpenVRUI/coColoredBackground.h>
//#include "coAction.h"

#include <OpenVRUI/coCheckboxGroup.h>

using namespace vrui;
using namespace vive;

vvVruiPartnerMenuItem::vvVruiPartnerMenuItem(const std::string &name, bool on, coCheckboxGroup *g)
    : coCheckboxMenuItem(name, on, g)
{
    //viewpoint = new
    viewpoint = new coToggleButton(new coFlatButtonGeometry("UI/eye"), this);
    viewpoint->setSize(LEFTMARGIN);
    container->addElement(viewpoint);
}

/// Destructor.
vvVruiPartnerMenuItem::~vvVruiPartnerMenuItem()
{
    delete viewpoint;
}

/** This method is called on intersections of the input device with the
  checkbox menu item.
  @return ACTION_CALL_ON_MISS
*/
int vvVruiPartnerMenuItem::hit(vruiHit *hit)
{
    container->setHighlighted(true);
    //return ACTION_CALL_ON_MISS;
    return 1;
}

/// Called when input device leaves the element.
void vvVruiPartnerMenuItem::miss()
{
    coCheckboxMenuItem::miss();
}

void vvVruiPartnerMenuItem::buttonEvent(coButton *button)
{
    if (button == viewpoint)
    {
        std::cerr << "viewpoint pressed " << viewpoint->getState() << std::endl;
    }
    else
    {
        if (vv->getPointerButton()->getState())
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

const char *vvVruiPartnerMenuItem::getClassName() const
{
    return "coPartnerMenuItem";
}

bool vvVruiPartnerMenuItem::isOfClassName(const char *classname) const
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
