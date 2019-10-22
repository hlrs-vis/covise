/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream>
#include <ostream>

#include "VruiPartnerMenuItem.h"
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coMenuContainer.h>
#include "coVRPluginSupport.h"
#include <OpenVRUI/coColoredBackground.h>
//#include "coAction.h"

#include <OpenVRUI/coCheckboxGroup.h>

using namespace vrui;
using namespace opencover;

VruiPartnerMenuItem::VruiPartnerMenuItem(const std::string &name, bool on, coCheckboxGroup *g)
    : coCheckboxMenuItem(name, on, g)
{
    //viewpoint = new
    viewpoint = new coToggleButton(new coFlatButtonGeometry("UI/eye"), this);
    viewpoint->setSize(LEFTMARGIN);
    container->addElement(viewpoint);
}

/// Destructor.
VruiPartnerMenuItem::~VruiPartnerMenuItem()
{
    delete viewpoint;
}

/** This method is called on intersections of the input device with the
  checkbox menu item.
  @return ACTION_CALL_ON_MISS
*/
int VruiPartnerMenuItem::hit(vruiHit *hit)
{
    container->setHighlighted(true);
    //return ACTION_CALL_ON_MISS;
    return 1;
}

/// Called when input device leaves the element.
void VruiPartnerMenuItem::miss()
{
    coCheckboxMenuItem::miss();
}

void VruiPartnerMenuItem::buttonEvent(coButton *button)
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

const char *VruiPartnerMenuItem::getClassName() const
{
    return "coPartnerMenuItem";
}

bool VruiPartnerMenuItem::isOfClassName(const char *classname) const
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
