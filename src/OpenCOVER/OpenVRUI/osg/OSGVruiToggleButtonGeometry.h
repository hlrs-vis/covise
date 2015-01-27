/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_TOGGLE_BUTTON_GEOMETRY_H
#define OSG_VRUI_TOGGLE_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/StateSet>
#include <osg/Switch>
#include <osg/Texture2D>
#include <osg/Vec3>

#include <string>

namespace vrui
{

class coToggleButtonGeometry;
class OSGVruiTransformNode;

class OSGVRUIEXPORT OSGVruiToggleButtonGeometry : public vruiButtonProvider
{
public:
    OSGVruiToggleButtonGeometry(coToggleButtonGeometry *button);
    virtual ~OSGVruiToggleButtonGeometry();
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
    // create Texture, either normal or checked
    osg::ref_ptr<osg::Geode> createNode(const std::string &textureName, bool checkTexture);

    void createSharedLists();

    // kept for compatibility only! They re-call createTexture()
    osg::ref_ptr<osg::Geode> createBox(const std::string &textureName);
    osg::ref_ptr<osg::Geode> createCheck(const std::string &textureName);

    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode;
    osg::ref_ptr<osg::Node> disabledNode;

    osg::ref_ptr<osg::Switch> switchNode;

    coToggleButtonGeometry *button;

    OSGVruiTransformNode *myDCS;

private:
    //shared coord and color list
    static float A;
    static float B;
    static float D;
    static osg::ref_ptr<osg::Vec3Array> coord;
    static osg::ref_ptr<osg::Vec3Array> normal;
    static osg::ref_ptr<osg::Vec2Array> texCoord;

    osg::ref_ptr<osg::Texture2D> texture;
};
}
#endif
