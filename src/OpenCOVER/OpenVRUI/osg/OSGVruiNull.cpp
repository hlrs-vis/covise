/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiNull.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <osg/MatrixTransform>

namespace vrui
{

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
OSGVruiNull::OSGVruiNull(coUIElement *element)
    : OSGVruiUIElement(element)
{
}

/** Destructor
 */
OSGVruiNull::~OSGVruiNull()
{
}

/// recalculate and set new geometry coordinates
void OSGVruiNull::resizeGeometry()
{
}

/// greate the  geometry node
void OSGVruiNull::createGeometry()
{

    if (myDCS)
        return;

    osg::MatrixTransform *transform = new osg::MatrixTransform();
    myDCS = new OSGVruiTransformNode(transform);
}
}
