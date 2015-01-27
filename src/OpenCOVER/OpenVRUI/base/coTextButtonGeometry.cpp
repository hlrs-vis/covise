/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <OpenVRUI/sginterface/vruiHit.h>
#include <OpenVRUI/coTextButtonGeometry.h>

namespace vrui
{

// w/h = button width/height
coTextButtonGeometry::coTextButtonGeometry(float w, float h, const std::string &name)
    : coButtonGeometry(name)
{
    width = w;
    height = h;
    c1r = c1g = c1b = 0.9f;
    c2r = c2g = c2b = 0.1f;
    c1a = c2a = 1.0;
}

coTextButtonGeometry::~coTextButtonGeometry()
{
}

void coTextButtonGeometry::setColors(float r1, float g1, float b1, float a1, float r2, float g2, float b2, float a2)
{
    c1r = r1;
    c1g = g1;
    c1b = b1;
    c1a = a1;
    c2r = r2;
    c2g = g2;
    c2b = b2;
    c2a = a2;
}

const char *coTextButtonGeometry::getClassName() const
{
    return "coTextButtonGeometry";
}

bool coTextButtonGeometry::isOfClassName(const char *classname) const
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
