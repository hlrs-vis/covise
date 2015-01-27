/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_COLORED_BACKGROUND_H
#define OSG_VRUI_COLORED_BACKGROUND_H

#include <OpenVRUI/osg/OSGVruiUIContainer.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Array>

namespace vrui
{

class coColoredBackground;

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class OSGVRUIEXPORT OSGVruiColoredBackground : public OSGVruiUIContainer
{
public:
    OSGVruiColoredBackground(coColoredBackground *background);
    virtual ~OSGVruiColoredBackground();

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

protected:
    void createSharedLists();

    coColoredBackground *background;

private:
    //shared coord and color list
    osg::ref_ptr<osg::Vec3Array> coord; ///< Coordinates of background geometry
    static osg::ref_ptr<osg::Vec3Array> normal; ///< Normal of background geometry

    osg::ref_ptr<osg::StateSet> state; ///< Normal geometry color
    osg::ref_ptr<osg::StateSet> highlightState; ///< Highlighted geometry color
    osg::ref_ptr<osg::StateSet> disabledState; ///< Disabled geometry color

    osg::ref_ptr<osg::Geode> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::Geometry> geometry; ///< Geometry object

    osg::ref_ptr<osg::MatrixTransform> fancyDCS;
};
}
#endif
