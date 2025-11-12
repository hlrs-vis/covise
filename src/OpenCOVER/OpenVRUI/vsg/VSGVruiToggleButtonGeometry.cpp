/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiToggleButtonGeometry.h>

#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>
#include <vsg/nodes/StateGroup.h>
#include <vsg/nodes/VertexIndexDraw.h>

#define STYLE_IN 1
#define STYLE_OUT 2

using namespace std;
using namespace vsg;

namespace vrui
{

float VSGVruiToggleButtonGeometry::A = 30.0f;
float VSGVruiToggleButtonGeometry::B = 50.0f;
float VSGVruiToggleButtonGeometry::D = 5.0f;

ref_ptr<vec3Array> VSGVruiToggleButtonGeometry::coord = nullptr;
ref_ptr<vec3Array> VSGVruiToggleButtonGeometry::normals = nullptr;
ref_ptr<vec2Array> VSGVruiToggleButtonGeometry::texCoord = nullptr;
ref_ptr<uintArray> VSGVruiToggleButtonGeometry::coordIndices = nullptr;
ref_ptr<vec4Array> VSGVruiToggleButtonGeometry::colors = nullptr;

/// Toggle Button is supposed to be a Button with four
/// states (bitmap extensions also shown):
/// 1) off
/// 2) off & selected    '-selected'
/// 3) on                '-check'
/// 4) on & selected     '-check-selected'
VSGVruiToggleButtonGeometry::VSGVruiToggleButtonGeometry(coToggleButtonGeometry *button)
    : vruiButtonProvider(button)
{
    this->button = button;
}

VSGVruiToggleButtonGeometry::~VSGVruiToggleButtonGeometry()
{
    delete myDCS;
}

void VSGVruiToggleButtonGeometry::createSharedLists()
{
    // global, static parameters for all Objects!
    // Only set up once in a lifetime! Check existence over coord

    if (!coord || !normals || !texCoord || !coordIndices)
    {

        coord = new vec3Array(4);
        texCoord = new vec2Array(4);

        // 3D coordinates used for textures
        (*coord)[3].set(0.0f, A, 0.0f);
        (*coord)[2].set(A, A, 0.0f);
        (*coord)[1].set(A, 0.0f, 0.0f);
        (*coord)[0].set(0.0f, 0.0f, 0.0f);

        // 2D coordinates valid for all textures
        (*texCoord)[0].set(0.0f, 1.0f);
        (*texCoord)[1].set(1.0f, 1.0f);
        (*texCoord)[2].set(1.0f, 0.0f);
        (*texCoord)[3].set(0.0f, 0.0f);

        // valid for all textures
        normals = vec3Array::create(4, vec3{ 0.0f, 0.0f, 1.0f });
        colors = vec4Array::create(4, vec4{ 0.9f, 0.9f, 0.9f,0.6f });

        coordIndices = uintArray::create(
            {
                0,1,2 , 0,2,3
            }
        );
    }
}

void VSGVruiToggleButtonGeometry::createGeometry()
{

    if (normalNode.get() == nullptr)
    {

        string textureName = button->getTextureName();

        // set up names
        string selectedName = textureName + "-selected";
        string checkName = textureName + "-check";
        string checkSelectedName = textureName + "-check-selected";
        string disabledName = textureName + "-disabled";

        // create normal texture
        normalNode = createNode(textureName, false);

        // create highlighted (selected) texture
        highlightNode = createNode(selectedName, false);

        // create pressed (check), normal texture
        pressedNode = createNode(checkName, true);

        // create pressed (check), highlighted (selected) texture
        pressedHighlightNode = createNode(checkSelectedName, true);

        disabledNode = createNode(disabledName, false);

        ref_ptr<vsg::MatrixTransform> transformNode = MatrixTransform::create();
        switchNode = Switch::create();

        switchNode->addChild(true, normalNode);
        switchNode->addChild(false, pressedNode);
        switchNode->addChild(false, highlightNode);
        switchNode->addChild(false, pressedHighlightNode);
        switchNode->addChild(false, disabledNode);

        transformNode->addChild(switchNode);

        myDCS = new VSGVruiTransformNode(transformNode);
        transformNode->setValue("name", "VSGVruiToggleButtonGeometry(" + element->getTextureName() + ")");
        transformNode->setValue("coButtonGeometry", element);
    }
}

ref_ptr<Node> VSGVruiToggleButtonGeometry::createNode(const string &textureName, bool checkTexture)
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

    VSGVruiTexture* oTex = dynamic_cast<VSGVruiTexture*>(vruiRendererInterface::the()->createTexture(textureName));
    auto image = Image::create(oTex->getTexture()->data);
    vruiRendererInterface::the()->deleteTexture(oTex);

    ref_ptr<GraphicsPipelineConfigurator> gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

    if (image->data)
        gpConfigurator->assignTexture("diffuseMap", image->data);
    else
        cerr << "No texture could be loaded for toggle button geometry!" << endl;

    gpConfigurator->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord);
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

// Kept for compatibility only!
ref_ptr<Node> VSGVruiToggleButtonGeometry::createBox(const string &textureName)
{
    return createNode(textureName, false);
}

ref_ptr<Node> VSGVruiToggleButtonGeometry::createCheck(const string &textureName)
{
    return createNode(textureName, true);
}

void VSGVruiToggleButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiToggleButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void VSGVruiToggleButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
