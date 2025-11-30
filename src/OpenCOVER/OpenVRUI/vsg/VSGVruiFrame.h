/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_FRAME_H
#define OSG_VRUI_FRAME_H

#include <util/coTypes.h>

#include <OpenVRUI/vsg/VSGVruiUIContainer.h>

#include <vsg/state/Sampler.h>
#include <vsg/utils/ShaderSet.h>
#include <vsg/utils/GraphicsPipelineConfigurator.h>

#include <string>

#ifdef _WIN32
typedef unsigned short ushort;
#endif

namespace vrui
{

class coFrame;

/** This class provides a flat textured frame arround objects.
  A frame should contain only one child, use another container to layout
  multiple children inside the frame.
  A frame can be configured to fit tight around its child or
  to maximize its size to always fit into its parent container
*/

class VSGVRUIEXPORT VSGVruiFrame : public VSGVruiUIContainer
{

public:
    VSGVruiFrame(coFrame *frame, const std::string &textureName = "UI/Frame");
    virtual ~VSGVruiFrame();

    void createGeometry();

protected:
    virtual void resizeGeometry();
    void realign();

    coFrame *frame;

private:
    //shared coord and color list
    void createSharedLists();

    //coordinates for the vertices of the frame geometry
    vsg::ref_ptr<vsg::vec3Array> coord; 

    //shared color, normal, geometry indices (of the geometry coords)
    //and texture coordinate vectors
    static vsg::ref_ptr <vsg::vec3Value> normal;
    static vsg::ref_ptr <vsg::vec4Value> color;
    static vsg::ref_ptr<vsg::uintArray> coordIndices;
    static vsg::ref_ptr<vsg::vec2Array> texCoords; 
    static vsg::ref_ptr<vsg::Data> textureData; 

    // test variable to test the order of the Frames created
    static std::int16_t frameCreatedCount;

    vsg::DataList vertexArrays;

    vsg::ref_ptr<vsg::Image> image;
    vsg::ref_ptr<vsg::Sampler> texture;

    vsg::ref_ptr<vsg::StateGroup> stateGroup; 
    vsg::ref_ptr<vsg::VertexIndexDraw> vertexIndexDraw;
    //vsg::ref_ptr<vsg::Geometry> geometry;

    vsg::ref_ptr<vsg::GraphicsPipelineConfigurator> gpConfigurator;
    //vsg::ref_ptr<vsg::DescriptorConfigurator> descConfigurator;
    //vsg::ref_ptr<vsg::ShaderSet> shaderSet;
    
    bool initiallyCompiled = false;
};
}
#endif
