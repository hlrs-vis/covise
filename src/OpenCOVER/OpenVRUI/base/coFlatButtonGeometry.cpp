/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coUIElement.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace std;

namespace vrui
{

#define STYLE_IN 1
#define STYLE_OUT 2
/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
coFlatButtonGeometry::coFlatButtonGeometry(const string &basename)
    : coButtonGeometry(basename)
{
}

/// Destructor.
coFlatButtonGeometry::~coFlatButtonGeometry()
{
}

const char *coFlatButtonGeometry::getClassName() const
{
    return "coFlatButtonGeometry";
}

bool coFlatButtonGeometry::isOfClassName(const char *classname) const
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
