/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coFlatPanelGeometry.h>

namespace vrui
{

coFlatPanelGeometry::coFlatPanelGeometry(coUIElement::Material c)
{
    backgroundMaterial = c;
}

coFlatPanelGeometry::~coFlatPanelGeometry()
{
}

coUIElement::Material coFlatPanelGeometry::getBackgroundMaterial() const
{
    return backgroundMaterial;
}

const char *coFlatPanelGeometry::getClassName() const
{
    return "coFlatPanelGeometry";
}

bool coFlatPanelGeometry::isOfClassName(const char *classname) const
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
            return coPanelGeometry::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
