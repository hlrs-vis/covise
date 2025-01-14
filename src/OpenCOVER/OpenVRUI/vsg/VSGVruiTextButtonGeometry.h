/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/coTextButtonGeometry.h>

#include <vsg/text/Text.h>

#include <string>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiTextButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiTextButtonGeometry(coTextButtonGeometry *geometry);
    virtual ~VSGVruiTextButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    virtual float getWidth() const;
    virtual float getHeight() const;

protected:
    std::string textString;
    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode;

    vsg::ref_ptr<vsg::Switch> switchNode;

    vruiTransformNode *myDCS;

    vsg::ref_ptr<vsg::Node> createNode(bool pressed, bool highlighted);
    vsg::vec4 color1, color2;
};
}
