/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#pragma once
#include <OpenVRUI/sginterface/vruiButtonProvider.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>

#include <string>

namespace vrui
{

class coRectButtonGeometry;
class VSGVruiTransformNode;

/**
    this class implements a rect, textured button
*/
class VSGVRUIEXPORT VSGVruiRectButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiRectButtonGeometry(coRectButtonGeometry *button);
    virtual ~VSGVruiRectButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual float getWidth() const
    {
        return 2.0f * (A + B) + C;
    }
    virtual float getHeight() const
    {
        return 2.0f * (A + B) + D;
    }

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

protected:
    void createSharedLists(); ///< creates shared coordinate arrays
    ///< creates the base button polygon
    vsg::ref_ptr<vsg::Node> createGeode(const std::string &textureName, int style);

    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode;

    vsg::ref_ptr<vsg::Switch> switchNode;

    coRectButtonGeometry *button;

    VSGVruiTransformNode *myDCS;

private:
    static float A;
    static float B;
    float C;
    float D;


    vsg::ref_ptr<vsg::Node> geode1; ///< base geometry
    vsg::ref_ptr<vsg::Node> geode2; ///< overlay geometry
};
}

