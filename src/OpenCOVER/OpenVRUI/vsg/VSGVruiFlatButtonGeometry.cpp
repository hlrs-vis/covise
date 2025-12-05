/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiFlatButtonGeometry.h>

#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>
#include <OpenVRUI/vsg/VSGVruiRendererInterface.h>

#include <OpenVRUI/coFlatButtonGeometry.h>

#include <vsg/all.h>
#include <vsgXchange/all.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{

#define STYLE_IN 1
#define STYLE_OUT 2

float VSGVruiFlatButtonGeometry::A = 30.0f;
float VSGVruiFlatButtonGeometry::B = 40.0f;
float VSGVruiFlatButtonGeometry::D = 5.0f;

ref_ptr<vec3Array> VSGVruiFlatButtonGeometry::coord1 = nullptr;
ref_ptr<vec3Array> VSGVruiFlatButtonGeometry::coord2 = nullptr;
ref_ptr<vec3Array> VSGVruiFlatButtonGeometry::normals = nullptr;
ref_ptr<vec2Array> VSGVruiFlatButtonGeometry::texCoord = nullptr;
ref_ptr<uintArray> VSGVruiFlatButtonGeometry::coordIndices = nullptr;
ref_ptr<vec4Array> VSGVruiFlatButtonGeometry::colors = nullptr;

/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
VSGVruiFlatButtonGeometry::VSGVruiFlatButtonGeometry(coFlatButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(nullptr)
{

    this->button = button;
}

/// Destructor.
VSGVruiFlatButtonGeometry::~VSGVruiFlatButtonGeometry()
{
    delete myDCS;
    myDCS = nullptr;
}

vsg::ref_ptr<vsg::Node> VSGVruiFlatButtonGeometry::createQuad(const vsg::vec3& origin, const vsg::vec3& horizontal, const vsg::vec3& vertical, ref_ptr<Data> image)
{

    auto builder = vsg::Builder::create();
    //builder->options = options;

    vsg::GeometryInfo geomInfo;
    geomInfo.position = origin;
    geomInfo.dx=horizontal;
    geomInfo.dy=vertical;
    geomInfo.dz.set(0.0f, 0.0f, 1.0f);

    vsg::StateInfo stateInfo;
    stateInfo.image = image; 

    return builder->createQuad(geomInfo, stateInfo);
}

void VSGVruiFlatButtonGeometry::createSharedLists()
{
    if (!coord1 || !coord2 || !normals || !coordIndices || !colors)
    {

        coord1 = new vec3Array(4);
        coord2 = new vec3Array(4);
        texCoord = new vec2Array(4);

        (*coord1)[3].set(0.0f, A, 0.0f);
        (*coord1)[2].set(A, A, 0.0f);
        (*coord1)[1].set(A, 0.0f, 0.0f);
        (*coord1)[0].set(0.0f, 0.0f, 0.0f);

        (*coord2)[3].set(0 - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D);
        (*coord2)[2].set(B - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D);
        (*coord2)[1].set(B - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D);
        (*coord2)[0].set(0 - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D);

        (*texCoord)[0].set(0.0f, 1.0f);
        (*texCoord)[1].set(1.0f, 1.0f);
        (*texCoord)[2].set(1.0f, 0.0f);
        (*texCoord)[3].set(0.0f, 0.0f);

        normals = vec3Array::create(4, vec3{ 0.0f, 0.0f, 1.0f });
        colors = vec4Array::create(4, vec4{ 0.9f, 0.9f, 0.9f, 0.7f });

        coordIndices = uintArray::create(
            {
                0,1,2 , 0,2,3
            }
        );
    }
}

ref_ptr<vsg::Node> VSGVruiFlatButtonGeometry::createBox(const string &textureName)
{

    createSharedLists();
    
    // setup using GraphicsPipelineConfigurator, without Options
    ref_ptr<ShaderSet> shaderSet;
    shaderSet = createFlatShadedShaderSet();

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
    depthStencilState->depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    shaderSet->defaultGraphicsPipelineStates.push_back(colorBlendState);
    shaderSet->defaultGraphicsPipelineStates.push_back(depthStencilState);
    
    DataList vertexArrays;


    ref_ptr<GraphicsPipelineConfigurator> gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

    auto image = VSGVruiRendererInterface::the()->createVsgTexture(textureName);
    if (image && image->data)
        gpConfigurator->assignTexture("diffuseMap", image->data);
    else
        cerr << "No texture could be loaded for flat button geometry!" << endl;

    gpConfigurator->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord1);
    gpConfigurator->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, normals);
    gpConfigurator->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, texCoord);
    gpConfigurator->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, colors);
     
    gpConfigurator->init();

    ref_ptr<StateGroup> stateGroup = StateGroup::create();
    gpConfigurator->copyTo(stateGroup);

    ref_ptr<VertexIndexDraw> vid = VertexIndexDraw::create();
    vid->assignArrays(vertexArrays);
    vid->assignIndices(coordIndices);
    vid->indexCount = static_cast<uint32_t>(coordIndices->size());
    vid->instanceCount = 1;

    stateGroup->addChild(vid);

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    node->addChild(stateGroup);
    return node;


}

ref_ptr<Node> VSGVruiFlatButtonGeometry::createCheck(const string &textureName)
{

    createSharedLists();

    // setup using GraphicsPipelineConfigurator, without Options
    ref_ptr<ShaderSet> shaderSet;
    shaderSet = createFlatShadedShaderSet();

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
    depthStencilState->depthWriteEnable = VK_TRUE;
    depthStencilState->depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    shaderSet->defaultGraphicsPipelineStates.push_back(colorBlendState);
    shaderSet->defaultGraphicsPipelineStates.push_back(depthStencilState);

    DataList vertexArrays;


    ref_ptr<GraphicsPipelineConfigurator> gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

    auto image = VSGVruiRendererInterface::the()->createVsgTexture(textureName);
    if (image && image->data)
        gpConfigurator->assignTexture("diffuseMap", image->data);
    else
        cerr << "No texture could be loaded for flat button geometry!" << endl;

    gpConfigurator->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord2);
    gpConfigurator->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, normals);
    gpConfigurator->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, texCoord);
    // gpConfigurator->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, colors);

    gpConfigurator->init();

    ref_ptr<StateGroup> stateGroup = StateGroup::create();
    gpConfigurator->copyTo(stateGroup);

    ref_ptr<VertexIndexDraw> vid = VertexIndexDraw::create();
    vid->assignArrays(vertexArrays);
    vid->assignIndices(coordIndices);
    vid->indexCount = static_cast<uint32_t>(coordIndices->size());
    vid->instanceCount = 1;

    stateGroup->addChild(vid);

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    node->addChild(stateGroup);
    return node;
}

void VSGVruiFlatButtonGeometry::createGeometry()
{

    if (myDCS)
        return;

    // name for highlighted geometry
    string highlightedName = string(button->getTextureName()) + "-highlighted";

    // name for checkmark geometry
    string checkName = string(button->getTextureName()) + "-check";

    // name for disabled geometry
    string disabledName = string(button->getTextureName()) + "-disabled";

    // Build checkMark and base/highlighted box geometries
    ref_ptr<Node> checkMark = createCheck(checkName);
    ref_ptr<Node> normalGeo = createBox(button->getTextureName());
    ref_ptr<Node> highlightGeo = createBox(highlightedName);
    ref_ptr<Node> disabledGeo = createBox(disabledName);


    // combine geometries pressed + normal
    ref_ptr<Group> pressedNormalGroup = Group::create();
    pressedNormalGroup->addChild(normalGeo);
    pressedNormalGroup->addChild(checkMark);

    // combine geometries pressed + highlighted
    ref_ptr<Group> pressedHighlightGroup = Group::create();
    pressedHighlightGroup->addChild(highlightGeo);
    pressedHighlightGroup->addChild(checkMark);

    // assign to 'base class' pointers
    normalNode = normalGeo;
    pressedNode = pressedNormalGroup;
    highlightNode = highlightGeo;
    pressedHighlightNode = pressedHighlightGroup;
    disabledNode = disabledGeo;

    ref_ptr<MatrixTransform> transformNode = MatrixTransform::create();
    switchNode = Switch::create();

    switchNode->addChild(true,normalNode);
    switchNode->addChild(false,pressedNode);
    switchNode->addChild(false, highlightNode);
    switchNode->addChild(false, pressedHighlightNode);
    switchNode->addChild(false, disabledNode);

    transformNode->addChild(switchNode);

    myDCS = new VSGVruiTransformNode(transformNode);
    transformNode->setValue("name", "VSGVruiFlatButtonGeometry(" + element->getTextureName() + ")");
    transformNode->setValue("coButtonGeometry", element);
}

void VSGVruiFlatButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiFlatButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void VSGVruiFlatButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
