/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coRectButtonGeometry.h>
#include <cstring>

namespace vrui
{

// w/h = button width/height
coRectButtonGeometry::coRectButtonGeometry(float w, float h, const std::string &name)
    : coButtonGeometry(name)
{
    width = w;
    height = h;
}

coRectButtonGeometry::~coRectButtonGeometry()
{
}

const char *coRectButtonGeometry::getClassName() const
{
    return "coRectButtonGeometry";
}

bool coRectButtonGeometry::isOfClassName(const char *classname) const
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
            return coButtonGeometry::isOfClassName(classname);
        }
    }

    // nobody is NULL
    return false;
}
}
