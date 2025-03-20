/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_COLORED_BACKGROUND_H
#define OSG_VRUI_COLORED_BACKGROUND_H

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/utils/GraphicsPipelineConfigurator.h>
#include <vsg/state/material.h>

namespace vrui
{

class coColoredBackground;

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class VSGVRUIEXPORT VSGVruiColoredBackground : public VSGVruiUIContainer
{
public:
    VSGVruiColoredBackground(coColoredBackground *background);
    virtual ~VSGVruiColoredBackground();

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

protected:
    void createSharedLists();

    coColoredBackground *background;

private:
    vsg::ref_ptr < vsg::GraphicsPipelineConfigurator> config;

    vsg::DataList vertexArrays;
    //shared coord and color list
    vsg::ref_ptr<vsg::vec3Array> coord; ///< Coordinates of background geometry
    vsg::ref_ptr<vsg::vec3Value> normal; ///< Normal of background geometry

    vsg::ref_ptr<vsg::StateGroup> stateGroup; ///< Normal geometry color

    vsg::ref_ptr<vsg::Node> node; ///< Geometry node

    vsg::ref_ptr<vsg::MatrixTransform> fancyDCS;
    vsg::ref_ptr <vsg::PhongMaterialValue> material;
};
}
#endif
