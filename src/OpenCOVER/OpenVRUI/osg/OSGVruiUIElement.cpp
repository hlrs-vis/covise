/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiUIElement.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <OpenVRUI/coUIElement.h>

namespace vrui
{

OSGVruiUIElement::OSGVruiUIElement(coUIElement *element)
    : vruiUIElementProvider(element)
    , myDCS(0)
{
}

OSGVruiUIElement::~OSGVruiUIElement()
{
    myDCS->removeAllParents();
    myDCS->removeAllChildren();
    delete myDCS;
    myDCS = 0;
}

vruiTransformNode *OSGVruiUIElement::getDCS()
{
    createGeometry();
    return myDCS;
}
}
