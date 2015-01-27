/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiNullButton.h>

#include <OpenVRUI/opensg/SGVruiTransformNode.h>

#include <OpenSG/OSGComponentTransform.h>

OSG_USING_NAMESPACE

SGVruiNullButton::SGVruiNullButton(coButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(0)
{
}

/** Destructor
 */
SGVruiNullButton::~SGVruiNullButton()
{
}

/// recalculate and set new geometry coordinates
void SGVruiNullButton::resizeGeometry()
{
}

/// greate the  geometry node
void SGVruiNullButton::createGeometry()
{

    if (myDCS)
        return;

    NodePtr transform = makeCoredNode<ComponentTransform>();
    myDCS = new SGVruiTransformNode(transform);
}

vruiTransformNode *SGVruiNullButton::getDCS()
{
    createGeometry();
    return myDCS;
}
