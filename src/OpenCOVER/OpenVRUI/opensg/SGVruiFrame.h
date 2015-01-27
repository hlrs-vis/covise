/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef SG_VRUI_FRAME_H
#define SG_VRUI_FRAME_H

#include <util/coTypes.h>

#include <OpenVRUI/opensg/SGVruiUIContainer.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGGeometry.h>
#include <OpenSG/OSGVector.h>

#include <string>

#ifdef _WIN32
typedef unsigned short ushort;
#endif

class coFrame;

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple children inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/

class VRUIEXPORT SGVruiFrame : public SGVruiUIContainer
{

public:
    SGVruiFrame(coFrame *frame, const std::string &textureName = "UI/Frame");
    virtual ~SGVruiFrame();

    void createGeometry();

protected:
    virtual void resizeGeometry();
    void realign();

    coFrame *frame;

private:
    //shared coord and color list
    void createSharedLists();

    osg::GeoPositions3fPtr coord; ///< Coordinates of frame geometry

    static osg::GeoColors4fPtr color; ///< Color of frame polygon
    static osg::GeoNormals3fPtr normal; ///< Normal of frame polygon
    static osg::GeoTexCoords2fPtr texCoord; ///< texture coordinates
    static osg::GeoIndicesUI32Ptr indices; ///< Index
    ///< Texture material
    static osg::ChunkMaterialPtr textureMaterial;

    osg::NodePtr geometryNode; ///< Geometry node
    osg::GeometryPtr geometry;
};
#endif
