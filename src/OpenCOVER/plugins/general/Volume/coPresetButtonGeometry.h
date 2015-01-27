/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_PRESET_BUTTON_GEOMETRY_H
#define CO_PRESET_BUTTON_GEOMETRY_H

#include <OpenVRUI/coButtonGeometry.h>

#include <osg/Node>
#include <osg/StateSet>
#include <osg/Switch>

namespace vrui
{
class vruiTransformNode;
}

class coPresetButtonGeometry : public vrui::coButtonGeometry
{
public:
    coPresetButtonGeometry();
    virtual ~coPresetButtonGeometry();
    virtual float getWidth()
    {
        return 5;
    }
    virtual float getHeight()
    {
        return 5;
    }

    ///< Switch the shown geometry
    virtual void switchGeometry(ActiveGeometry active);

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual vrui::vruiTransformNode *getDCS();

protected:
    osg::ref_ptr<osg::Node> createNode(bool, bool);
    osg::ref_ptr<osg::StateSet> createGeoState(bool);

    osg::ref_ptr<osg::Node> normalNode; ///< normal geometry
    osg::ref_ptr<osg::Node> pressedNode; ///< pressed normal geometry
    osg::ref_ptr<osg::Node> highlightNode; ///< highlighted geometry
    ///< pressed highlighted geometry
    osg::ref_ptr<osg::Node> pressedHighlightNode;

    osg::ref_ptr<osg::Switch> switchNode;

    vrui::vruiTransformNode *myDCS;
};
#endif
