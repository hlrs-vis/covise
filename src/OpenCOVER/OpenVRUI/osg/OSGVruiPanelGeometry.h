/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_PANEL_GEOMETRY_H
#define OSG_VRUI_PANEL_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiPanelGeometryProvider.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osg/Vec3>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiPanelGeometry : public virtual vruiPanelGeometryProvider
{
public:
    OSGVruiPanelGeometry(coPanelGeometry *geometry);
    virtual ~OSGVruiPanelGeometry();
    virtual void attachGeode(vruiTransformNode *node);
    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

    virtual void createSharedLists();

private:
    coPanelGeometry *geometry;

    static float A;
    static float B;
    static float C;

    static osg::ref_ptr<osg::Vec3Array> coord;
    static osg::ref_ptr<osg::Vec3Array> normal;
    static osg::ref_ptr<osg::Vec4Array> color;
    static osg::ref_ptr<osg::Vec2Array> texcoord;

    static osg::ref_ptr<osg::DrawElementsUShort> coordIndex;

    static osg::ref_ptr<osg::UShortArray> normalIndex;

    static osg::ref_ptr<osg::Material> textureMaterial;

    osg::ref_ptr<osg::Texture2D> texture;
};
}
#endif
