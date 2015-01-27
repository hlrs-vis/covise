/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FRAME_H
#define OSG_VRUI_FRAME_H

#include <util/coTypes.h>

#include <OpenVRUI/osg/OSGVruiUIContainer.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Vec3>

#include <string>

#ifdef _WIN32
typedef unsigned short ushort;
#endif

namespace vrui
{

class coFrame;

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple children inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/

class OSGVRUIEXPORT OSGVruiFrame : public OSGVruiUIContainer
{

public:
    OSGVruiFrame(coFrame *frame, const std::string &textureName = "UI/Frame");
    virtual ~OSGVruiFrame();

    void createGeometry();

protected:
    virtual void resizeGeometry();
    void realign();

    coFrame *frame;

private:
    //shared coord and color list
    void createSharedLists();

    osg::ref_ptr<osg::Vec3Array> coord;

    static osg::ref_ptr<osg::Vec4Array> color;
    static osg::ref_ptr<osg::Vec3Array> normal;
    static osg::ref_ptr<osg::Vec2Array> texCoord;
    static osg::ref_ptr<osg::DrawElementsUShort> coordIndex;

    osg::ref_ptr<osg::Texture2D> texture;

    osg::ref_ptr<osg::Geode> geometryNode;
    osg::ref_ptr<osg::Geometry> geometry;
    osg::ref_ptr<osg::StateSet> stateSet;
};
}
#endif
