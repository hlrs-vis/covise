/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef OSG_VRUI_RECT_BUTTON_GEOMETRY_H
#define OSG_VRUI_RECT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Vec3>

#include <string>

namespace vrui
{

class coRectButtonGeometry;
class OSGVruiTransformNode;

/**
    this class implements a rect, textured button
*/
class OSGVRUIEXPORT OSGVruiRectButtonGeometry : public vruiButtonProvider
{
public:
    OSGVruiRectButtonGeometry(coRectButtonGeometry *button);
    virtual ~OSGVruiRectButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual float getWidth() const
    {
        return 2.0 * (A + B) + C;
    }
    virtual float getHeight() const
    {
        return 2.0 * (A + B) + D;
    }

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

protected:
    void createSharedLists(); ///< creates shared coordinate arrays
    ///< creates the base button polygon
    osg::ref_ptr<osg::Node> createGeode(const std::string &textureName, int style);

    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode;

    osg::ref_ptr<osg::Switch> switchNode;

    coRectButtonGeometry *button;

    OSGVruiTransformNode *myDCS;

private:
    static float A;
    static float B;
    float C;
    float D;

    static osg::ref_ptr<osg::Vec4Array> color;
    static osg::ref_ptr<osg::Vec3Array> coord1; ///< coordinates
    static osg::ref_ptr<osg::Vec3Array> coord2; ///< coordinates
    static osg::ref_ptr<osg::Vec3Array> coordt1;
    static osg::ref_ptr<osg::Vec3Array> coordt2;
    static osg::ref_ptr<osg::Vec3Array> normal1; ///< normals
    static osg::ref_ptr<osg::Vec3Array> normal2; ///< normals
    static osg::ref_ptr<osg::Vec3Array> normalt;
    ///< texture coordinates
    static osg::ref_ptr<osg::Vec2Array> texCoord;

    static osg::ref_ptr<osg::DrawElementsUShort> coordIndex;

    static osg::ref_ptr<osg::Material> textureMat;
    static osg::ref_ptr<osg::StateSet> normalStateSet;

    osg::ref_ptr<osg::Geode> geode1; ///< base geometry
    osg::ref_ptr<osg::Geode> geode2; ///< overlay geometry
};
}
#endif
