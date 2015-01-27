/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiNull.h>

#include <OpenVRUI/opensg/SGVruiTransformNode.h>

#include <OpenSG/OSGComponentTransform.h>
#include <OpenSG/OSGNode.h>

OSG_USING_NAMESPACE

SGVruiNull::SGVruiNull(coUIElement *element)
    : SGVruiUIElement(element)
{
}

/** Destructor
 */
SGVruiNull::~SGVruiNull()
{
}

/// recalculate and set new geometry coordinates
void SGVruiNull::resizeGeometry()
{
}

/// greate the  geometry node
void SGVruiNull::createGeometry()
{

    if (myDCS)
        return;

    NodePtr transform = makeCoredNode<ComponentTransform>();
    myDCS = new SGVruiTransformNode(transform);
}
