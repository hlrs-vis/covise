/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_NULL_H
#define OSG_VRUI_NULL_H

#include <util/coTypes.h>

#include <OpenVRUI/osg/OSGVruiUIElement.h>

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple children inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/
namespace vrui
{

class OSGVRUIEXPORT OSGVruiNull : public OSGVruiUIElement
{

public:
    OSGVruiNull(coUIElement *element);
    virtual ~OSGVruiNull();

    void createGeometry();

protected:
    virtual void resizeGeometry();
};
}
#endif
