/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coDefaultButtonGeometry.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <OpenVRUI/util/vruiLog.h>

namespace vrui
{

coDefaultButtonGeometry::coDefaultButtonGeometry(const std::string &name)
    : coButtonGeometry(name)
{
}

coDefaultButtonGeometry::~coDefaultButtonGeometry()
{
}

const char *coDefaultButtonGeometry::getClassName() const
{
    return "coDefaultButtonGeometry";
}

bool coDefaultButtonGeometry::isOfClassName(const char *classname) const
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
