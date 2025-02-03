/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_DEFAULT_BUTTON_GEOMETRY_H
#define OSG_VRUI_DEFAULT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/coDefaultButtonGeometry.h>

#include <vsg/text/Text.h>
#include <vsg/text/StandardLayout.h>

#include <string>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiDefaultButtonGeometry : public vruiButtonProvider
{
public:
    VSGVruiDefaultButtonGeometry(coDefaultButtonGeometry *geometry);
    virtual ~VSGVruiDefaultButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    virtual float getWidth() const;
    virtual float getHeight() const;

protected:
    vsg::ref_ptr<vsg::Node> normalNode; ///< normal geometry
    vsg::ref_ptr<vsg::Node> pressedNode; ///< pressed normal geometry
    vsg::ref_ptr<vsg::Node> highlightNode; ///< highlighted geometry
    vsg::ref_ptr<vsg::Node> pressedHighlightNode; ///< pressed highlighted geometry
    vsg::ref_ptr<vsg::Node> disabledNode; ///< disabled geometry

    vsg::ref_ptr<vsg::Switch> switchNode;

    vruiTransformNode *myDCS;

    vsg::ref_ptr<vsg::Node> createNode(bool pressed, bool highlighted, bool disabled = false);

    std::string textString;

    vsg::ref_ptr<vsg::Text> createText(const std::string &text,
                                           vsg::StandardLayout::Alignment align, float size);
};
}
#endif
