/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <OpenVRUI/coSquareButtonGeometry.h>

using namespace std;

namespace vrui
{

/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb.
*/
coSquareButtonGeometry::coSquareButtonGeometry(const string &name)
    : coButtonGeometry(name)
{
}

coSquareButtonGeometry::~coSquareButtonGeometry()
{
}

const char *coSquareButtonGeometry::getClassName() const
{
    return "coSquareButtonGeometry";
}

bool coSquareButtonGeometry::isOfClassName(const char *classname) const
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
