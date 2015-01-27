/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FLAT_BUTTON_GEOMETRY_H
#define OSG_VRUI_FLAT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Vec3>
#include <osg/Texture2D>

#include <string>

namespace vrui
{

class coFlatButtonGeometry;
class OSGVruiTransformNode;

/**
    this class implements a flat, textured button
*/
class OSGVRUIEXPORT OSGVruiFlatButtonGeometry : public vruiButtonProvider
{
public:
    OSGVruiFlatButtonGeometry(coFlatButtonGeometry *button);
    virtual ~OSGVruiFlatButtonGeometry();
    virtual float getWidth() const
    {
        return A;
    }
    virtual float getHeight() const
    {
        return A;
    }

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

protected:
    ///< creates the base button polygon
    osg::ref_ptr<osg::Geode> createBox(const std::string &textureName);
    ///< creates the overlay check polygon
    osg::ref_ptr<osg::Geode> createCheck(const std::string &textureName);
    void createSharedLists(); ///< creates shared coordinate arrays

    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode;
    osg::ref_ptr<osg::Node> disabledNode; ///< disabled geometry

    osg::ref_ptr<osg::Switch> switchNode;

    coFlatButtonGeometry *button;

    OSGVruiTransformNode *myDCS;

    osg::ref_ptr<osg::Texture2D> defaulTexture;

private:
    //shared coord and color list
    static float A; ///< size parameters
    static float B; ///< size parameters
    static float D; ///< size parameters
    static osg::ref_ptr<osg::Vec3Array> coord1; ///< coordinates
    static osg::ref_ptr<osg::Vec3Array> coord2; ///< coordinates
    static osg::ref_ptr<osg::Vec3Array> normal; ///< normals
    ///< texture coordinates
    static osg::ref_ptr<osg::Vec2Array> texCoord;

    osg::ref_ptr<osg::Geode> geode1; ///< base geometry
    osg::ref_ptr<osg::Geode> geode2; ///< overlay geometry
};
}
#endif
