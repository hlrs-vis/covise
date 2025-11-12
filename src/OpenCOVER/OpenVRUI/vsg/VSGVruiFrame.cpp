/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiFrame.h>

#include <OpenVRUI/coFrame.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/all.h>
#include <vsgXchange/all.h>

#include <string>
#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{
    //initialize shared variables for all VSGVruiFrame objects
    ref_ptr<vec4Array> VSGVruiFrame::colors = nullptr;
    ref_ptr<vec3Array> VSGVruiFrame::normals = nullptr;
    ref_ptr<vec2Array> VSGVruiFrame::texCoords = vec2Array::create(24);
    ref_ptr<uintArray> VSGVruiFrame::coordIndices = uintArray::create(72);
    ref_ptr<Data> VSGVruiFrame::textureData = nullptr;

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
VSGVruiFrame::VSGVruiFrame(coFrame *frame, const string &name)
    : VSGVruiUIContainer(frame)
{

    this->frame = frame;

    texture = 0;

    ref_ptr<MatrixTransform> transform = MatrixTransform::create();

    myDCS = new VSGVruiTransformNode(transform);

    transform->setValue("name", "VSGVruiFrame(" + name + ")");
    transform->setValue("coUIElement", frame); 

    createGeometry();

    transform->addChild(stateGroup);
    
}

/** Destructor
 */
VSGVruiFrame::~VSGVruiFrame()
{
}

/// recalculate and set new geometry coordinates
void VSGVruiFrame::resizeGeometry()
{
    // testing frame geometry resizing
    
    float fw = frame->getWidth();
    float fh = frame->getHeight();

    //VRUILOG("VSGVruiFrame::resizeGeometry info: resizing " << fw << "x" << fh)

    float bw = frame->getBorderWidth();
    float bh = frame->getBorderHeight();

    float iw = fw - 2 * bw;
    float ih = fh - 2 * bh;

    //recalculate and assign frame coords
    (*coord)[0].set(0, fh, 0);
    (*coord)[1].set(20, fh, 0);
    (*coord)[2].set(fw - 20, fh, 0);
    (*coord)[3].set(fw, fh, 0);
    (*coord)[4].set(fw, fh - 20, 0);
    (*coord)[5].set(fw, 20, 0);
    (*coord)[6].set(fw, 0, 0);
    (*coord)[7].set(fw - 20, 0, 0);
    (*coord)[8].set(20, 0, 0);
    (*coord)[9].set(0, 0, 0);
    (*coord)[10].set(0, 20, 0);
    (*coord)[11].set(0, fh - 20, 0);
    (*coord)[12].set(bw, ih + bh, 0);
    (*coord)[13].set(20, ih + bh, 0);
    (*coord)[14].set(fw - 20, ih + bh, 0);
    (*coord)[15].set(iw + bw, ih + bh, 0);
    (*coord)[16].set(iw + bw, fh - 20, 0);
    (*coord)[17].set(iw + bw, 20, 0);
    (*coord)[18].set(iw + bw, bh, 0);
    (*coord)[19].set(fw - 20, bh, 0);
    (*coord)[20].set(20, bh, 0);
    (*coord)[21].set(bw, bh, 0);
    (*coord)[22].set(bw, 20, 0);
    (*coord)[23].set(bw, fh - 20, 0);

    coord->dirty();
    
    // if compiled, buffers have been created
    if (!initiallyCompiled)
    {
        initiallyCompiled = vruiRendererInterface::the()->compileNode(myDCS);
    }
    else 
    {
        if ((*vertexIndexDraw).arrays[0]->buffer) 
        {
            vruiRendererInterface::the()->addToTransfer(vertexIndexDraw->arrays[0]);
        }
    }
}

/// allocate shared datastructures that can be used by all frames
void VSGVruiFrame::createSharedLists()
{
    if (!colors || !normals) 
    {
        // 12 quads * 2 tris * 3 indices
        coordIndices = uintArray::create({
            0,12,13 , 0,13,1 , 1,13,14 , 1,14,2 , 2,14,15 , 2,15,3,
            15,16,4 , 15,4,3 , 16,17,5 , 16,5,4 , 17,18,6 , 17,6,5,
            6,18,19 , 6,19,7 , 7,19,20 , 7,20,8 , 8,20,21 , 8,21,9,
            9,21,22 , 9,22,10 , 10,22,23 , 10,23,11 , 11,23,12 , 11,12,0
            });

        // color and normal
        normals = vec3Array::create(24, vec3{ 0.0f, 0.0f, 1.0f });
        colors = vec4Array::create(24, vec4{ 0.8f, 0.8f, 0.8f, 1.0f });

        // texture coordinates (might be flipped, need to check with another texture)
        texCoords->set(0, vec2{ (1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 32.0f });
        texCoords->set(1, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f });
        texCoords->set(2, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f });
        texCoords->set(3, vec2{ (1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 32.0f });
        texCoords->set(4, vec2{ (1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(5, vec2{ (1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(6, vec2{ (1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 0.0f });
        texCoords->set(7, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f });
        texCoords->set(8, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f });
        texCoords->set(9, vec2{ (1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 0.0f });
        texCoords->set(10, vec2{ (1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(11, vec2{ (1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(12, vec2{ (1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 24.0f });
        texCoords->set(13, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f });
        texCoords->set(14, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f });
        texCoords->set(15, vec2{ (1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 24.0f });
        texCoords->set(16, vec2{ (1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(17, vec2{ (1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(18, vec2{ (1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 8.0f });
        texCoords->set(19, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f });
        texCoords->set(20, vec2{ (1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f });
        texCoords->set(21, vec2{ (1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 8.0f });
        texCoords->set(22, vec2{ (1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f });
        texCoords->set(23, vec2{ (1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f });

    }
}

/// create the  geometry node
void VSGVruiFrame::createGeometry()
{
    // testing frame geometry here
    if (!stateGroup) 
    {
        // create shared lists (shared by all Frames)
        createSharedLists();

        // set vertices of geometry 
        coord = vec3Array::create({
        {0,60,0} , {20,60,0} , {40,60,0} ,  //outer frame
        {60,60,0} , {60,40,0} , {60,20,0} ,
        {60,0,0} , {40,0,0} , {20,0,0} ,
        {0,0,0} , {0,20,0} , {0,40,0} ,

        {5,55,0} , {20,55,0} , {40,55,0} , //inner frame
        {55,55,0} , {55,40,0} , {55,20,0} ,
        {55,5,0} , {40,5,0} , {20,5,0} ,
        {5,5,0} , {5,20,0} , {5,40,0}
        });
        
        // to test dataVariance = DYNAMIC_DATA 
        //coord->properties.dataVariance = DYNAMIC_DATA;

        // setup using GraphicsPipelineConfigurator, without Options
        ref_ptr<ShaderSet> shaderSet;
        shaderSet = createFlatShadedShaderSet();
        //shaderSet = createPhongShaderSet();

        // custom setting for PipelineStates, attachments[0] is the default
        auto colorBlendState = vsg::ColorBlendState::create();
        colorBlendState->attachments[0].blendEnable = VK_TRUE;
        colorBlendState->attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendState->attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendState->attachments[0].colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendState->attachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendState->attachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendState->attachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendState->attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendState->logicOpEnable = VK_FALSE;

        auto depthStencilState = vsg::DepthStencilState::create();
        depthStencilState->depthTestEnable = VK_TRUE;
        depthStencilState->depthWriteEnable = VK_FALSE;
        depthStencilState->depthCompareOp = VK_COMPARE_OP_GREATER;

        shaderSet->defaultGraphicsPipelineStates.push_back(colorBlendState);
        shaderSet->defaultGraphicsPipelineStates.push_back(depthStencilState);

        // load the texture
        VSGVruiTexture* oTex = dynamic_cast<VSGVruiTexture*>(vruiRendererInterface::the()->createTexture(frame->getTextureName()));
        image = Image::create(oTex->getTexture()->data);
        vruiRendererInterface::the()->deleteTexture(oTex);
        
        gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

        if (image->data)
        {
            gpConfigurator->assignTexture("diffuseMap", image->data);
        }

        gpConfigurator->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord);
        gpConfigurator->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, normals);
        gpConfigurator->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, texCoords);
        gpConfigurator->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, colors);

        gpConfigurator->init();

        stateGroup = StateGroup::create();
        gpConfigurator->copyTo(stateGroup);

        vertexIndexDraw = VertexIndexDraw::create();
        vertexIndexDraw->assignArrays(vertexArrays);
        vertexIndexDraw->assignIndices(coordIndices);
        vertexIndexDraw->indexCount = static_cast<uint32_t>(coordIndices->size());
        vertexIndexDraw->instanceCount = 1;

        stateGroup->addChild(vertexIndexDraw);

        resizeGeometry();
    }
 
}
}
