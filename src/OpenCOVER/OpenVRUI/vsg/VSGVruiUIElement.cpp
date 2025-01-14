/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiUIElement.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include <OpenVRUI/coUIElement.h>

namespace vrui
{

VSGVruiUIElement::VSGVruiUIElement(coUIElement *element)
    : vruiUIElementProvider(element)
    , myDCS(0)
{
}

VSGVruiUIElement::~VSGVruiUIElement()
{
    myDCS->removeAllParents();
    myDCS->removeAllChildren();
    delete myDCS;
    myDCS = 0;
}

vruiTransformNode *VSGVruiUIElement::getDCS()
{
    createGeometry();
    return myDCS;
}
}
