/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiNull.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>

#include <vsg/nodes/MatrixTransform.h>

namespace vrui
{

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
VSGVruiNull::VSGVruiNull(coUIElement *element)
    : VSGVruiUIElement(element)
{
}

/** Destructor
 */
VSGVruiNull::~VSGVruiNull()
{
}

/// recalculate and set new geometry coordinates
void VSGVruiNull::resizeGeometry()
{
}

/// greate the  geometry node
void VSGVruiNull::createGeometry()
{

    if (myDCS)
        return;

    myDCS = new VSGVruiTransformNode(vsg::MatrixTransform::create());
}
}
