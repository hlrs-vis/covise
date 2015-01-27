/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_TEXT_BUTTON_GEOMETRY_H
#define OSG_VRUI_TEXT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/coTextButtonGeometry.h>

#include <osg/Switch>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Sequence>
#include <osg/Texture>
#include <osgText/Text>

#include <string>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiTextButtonGeometry : public vruiButtonProvider
{
public:
    OSGVruiTextButtonGeometry(coTextButtonGeometry *geometry);
    virtual ~OSGVruiTextButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    virtual float getWidth() const;
    virtual float getHeight() const;

protected:
    std::string textString;
    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode;

    osg::ref_ptr<osg::Switch> switchNode;

    vruiTransformNode *myDCS;

    osg::ref_ptr<osg::Node> createNode(bool pressed, bool highlighted);
    osg::Vec4 color1, color2;
};
}
#endif
