/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiUIElement.h>
#include <OpenVRUI/opensg/SGVruiMatrix.h>

#include <OpenVRUI/coUIElement.h>

OSG_USING_NAMESPACE

SGVruiUIElement::SGVruiUIElement(coUIElement *element)
    : vruiUIElementProvider(element)
    , myDCS(0)
{
}

SGVruiUIElement::~SGVruiUIElement()
{
    myDCS->removeAllParents();
    myDCS->removeAllChildren();
    delete myDCS;
    myDCS = 0;
}

vruiTransformNode *SGVruiUIElement::getDCS()
{
    createGeometry();
    return myDCS;
}
