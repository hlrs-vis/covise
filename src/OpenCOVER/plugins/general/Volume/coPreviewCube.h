/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_PREVIEW_CUBE_H
#define CO_PREVIEW_CUBE_H

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coButtonGeometry.h>
#include <OpenVRUI/coUIElement.h>

#include <osg/Array>
#include <osg/Geometry>
#include <osg/Switch>

namespace vrui
{
class vruiHit;
class OSGVruiTransformNode;
}

class coPreviewCube : public vrui::coUIElement
{
public:
    coPreviewCube();
    virtual ~coPreviewCube();

    osg::ref_ptr<osg::Node> createNode();
    vrui::vruiTransformNode *getDCS();
    void setPos(float x, float y, float z = 0);
    void setSize(float xs, float ys, float zs);
    void setSize(float);
    void setOrientation(float);
    virtual float getWidth() const
    {
        return 2.5f * scale;
    }
    virtual float getHeight() const
    {
        return 2.5f * scale;
    }
    virtual float getXpos() const
    {
        return myX;
    }
    virtual float getYpos() const
    {
        return myY;
    }
    void setHSVA(float, float, float, float);
    void setHS(float, float);
    void setAlpha(float);
    void setBrightness(float);
    void update();

protected:
    vrui::OSGVruiTransformNode *myDCS;
    float myX, myY;
    float scale;
    float h, s, v, a;
    osg::ref_ptr<osg::StateSet> normalGeostate;
    double start;
    float angle; // current cube rotational angle (degrees)

    void createLists();

private:
    osg::ref_ptr<osg::Vec4Array> color;
    osg::ref_ptr<osg::Vec3Array> coord;
    osg::ref_ptr<osg::Vec3Array> normal;

    osg::ref_ptr<osg::DrawElementsUShort> vertices;

    osg::ref_ptr<osg::Geometry> cubeGeo;
};
#endif
