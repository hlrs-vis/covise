/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_DEFAULT_BUTTON_GEOMETRY_H
#define OSG_VRUI_DEFAULT_BUTTON_GEOMETRY_H

#include <OpenVRUI/sginterface/vruiButtonProvider.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenVRUI/coDefaultButtonGeometry.h>

#include <osg/Switch>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Sequence>
#include <osg/Texture>
#include <osgText/Text>

#include <string>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiDefaultButtonGeometry : public vruiButtonProvider
{
public:
    OSGVruiDefaultButtonGeometry(coDefaultButtonGeometry *geometry);
    virtual ~OSGVruiDefaultButtonGeometry();

    virtual void switchGeometry(coButtonGeometry::ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vruiTransformNode *getDCS();

    virtual float getWidth() const;
    virtual float getHeight() const;

protected:
    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode; ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> disabledNode; ///< disabled geometry

    osg::ref_ptr<osg::Switch> switchNode;

    vruiTransformNode *myDCS;

    osg::ref_ptr<osg::Node> createNode(bool pressed, bool highlighted, bool disabled = false);
    osg::ref_ptr<osg::StateSet> createGeoState(bool highlighted);

    std::string textString;

    osg::ref_ptr<osgText::Text> createText(const std::string &text,
                                           osgText::Text::AlignmentType align, float size);
};
}
#endif
