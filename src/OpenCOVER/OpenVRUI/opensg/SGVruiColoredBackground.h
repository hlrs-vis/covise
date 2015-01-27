/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_COLORED_BACKGROUND_H
#define SG_VRUI_COLORED_BACKGROUND_H

#include <OpenVRUI/opensg/SGVruiUIContainer.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGGeometry.h>
#include <OpenSG/OSGVector.h>

class coColoredBackground;

/** This class provides background for GUI elements.
  The color of this background changes according to the elements state
  (normal/highlighted/disabled)
  A background should contain only one child, use another container to layout
  multiple chlidren inside the frame.
*/
class SGVRUIEXPORT SGVruiColoredBackground : public SGVruiUIContainer
{
public:
    SGVruiColoredBackground(coColoredBackground *background);
    virtual ~SGVruiColoredBackground();

    virtual void createGeometry();
    virtual void resizeGeometry();

    virtual void setEnabled(bool en);
    virtual void setHighlighted(bool hl);

protected:
    void createSharedLists();

    coColoredBackground *background;

private:
    //shared coord and color list
    osg::GeoPositions3fPtr coord; ///< Coordinates of background geometry
    static osg::GeoNormals3fPtr normal; ///< Normal of background geometry
    osg::GeoIndicesUI32Ptr indices; ///< Index

    osg::ChunkMaterialPtr material; ///< Normal geometry color
    osg::ChunkMaterialPtr highlightMaterial; ///< Highlighted geometry color
    osg::ChunkMaterialPtr disabledMaterial; ///< Disabled geometry color

    osg::NodePtr geometryNode; ///< Geometry node
    osg::GeometryPtr geometry; ///< Geometry object

    osg::NodePtr fancyDCS;
};
#endif
