/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FLAT_PANEL_GEOMETRY_H
#define OSG_VRUI_FLAT_PANEL_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiPanelGeometryProvider.h>

#include <OpenVRUI/coUIElement.h>

#include <osg/Array>

namespace vrui
{

class coFlatPanelGeometry;

class OSGVRUIEXPORT OSGVruiFlatPanelGeometry : public vruiPanelGeometryProvider
{
public:
    OSGVruiFlatPanelGeometry(coFlatPanelGeometry *geometry);
    virtual ~OSGVruiFlatPanelGeometry();

    virtual void attachGeode(vruiTransformNode *node);
    virtual float getWidth() const;
    virtual float getHeight() const;
    virtual float getDepth() const;

protected:
    virtual void createSharedLists();

private:
    //shared coord and color list
    static float A;
    static float B;
    //static osg::ref_ptr<osg::Vec4Array> color;
    static osg::ref_ptr<osg::Vec3Array> coord;
    static osg::ref_ptr<osg::Vec3Array> normal;

    coUIElement::Material backgroundMaterial;
    coFlatPanelGeometry *panelGeometry;
};
}
#endif
