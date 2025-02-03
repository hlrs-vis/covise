/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <vsg/state/Sampler.h>

#include <string>

namespace vrui
{

class coToggleButtonGeometry;
class VSGVruiTransformNode;

class VSGVRUIEXPORT VSGVruiToggleButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiToggleButtonGeometry(coToggleButtonGeometry *button);
    virtual ~VSGVruiToggleButtonGeometry();
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
    vsg::ref_ptr<vsg::Node> createNode(const std::string &textureName, bool checkTexture);

    void createSharedLists();

    // kept for compatibility only! They re-call createTexture()
    vsg::ref_ptr<vsg::Node> createBox(const std::string &textureName);
    vsg::ref_ptr<vsg::Node> createCheck(const std::string &textureName);

    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode;
    vsg::ref_ptr<vsg::Node> disabledNode;

    vsg::ref_ptr<vsg::Switch> switchNode;

    coToggleButtonGeometry *button;

    VSGVruiTransformNode *myDCS;

private:
    //shared coord and color list
    static float A;
    static float B;
    static float D;
    static vsg::ref_ptr<vsg::vec3Array> coord;
    static vsg::ref_ptr<vsg::vec3Array> normal;
    static vsg::ref_ptr<vsg::vec2Array> texCoord;

    vsg::ref_ptr<vsg::Sampler> texture;
};
}
