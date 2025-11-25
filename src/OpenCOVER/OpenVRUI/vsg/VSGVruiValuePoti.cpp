/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiValuePoti.h>

#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/vsg/VSGVruiRendererInterface.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/util/vruiLog.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>
#include <vsg/text/GpuLayoutTechnique.h>
#include <vsg/nodes/VertexIndexDraw.h>
#include "mathUtils.h"

using namespace vsg;
using namespace std;

namespace vrui
{

VSGVruiValuePoti::VSGVruiValuePoti(coValuePoti *poti)
    : VSGVruiUIElement(poti)
    , poti(0)
{
    this->poti = poti;
}

VSGVruiValuePoti::~VSGVruiValuePoti()
{
}

void VSGVruiValuePoti::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("VSGVruiValuePoti::createGeometry info: making geometry")

    potiTransform = MatrixTransform::create();
    textTransform = MatrixTransform::create();

    potiText = Text::create();
    potiTextLayout = StandardLayout::create();
    potiTextString = stringValue::create();

    potiText->technique = GpuLayoutTechnique::create();

    potiText->font = VSGVruiPresets::instance()->font2;
    potiText->text = potiTextString;
    potiText->layout = potiTextLayout;

    ref_ptr<MatrixTransform> mainTransform = MatrixTransform::create();
    ref_ptr<MatrixTransform> panelTransform = MatrixTransform::create();

    ref_ptr<Group> icon = Group::create();
    ref_ptr<Group> iconDisabled = Group::create();

    VSGVruiNode *iconNode = dynamic_cast<VSGVruiNode *>(vruiRendererInterface::the()->getIcon("UI/poti2"));

    if (!iconNode || !iconNode->getNodePtr())
        ; //VRUILOG("VSGVruiValuePoti::createGeometry warn: cannot open icon node UI/poti2")
    else
        icon->addChild(iconNode->node);

    iconNode = dynamic_cast<VSGVruiNode *>(vruiRendererInterface::the()->getIcon("UI/poti2Disabled"));
    if (!iconNode || !iconNode->getNodePtr())
        ; //VRUILOG("VSGVruiValuePoti::createGeometry warn: cannot open icon node UI/poti2Disabled")
    else
        iconDisabled->addChild(iconNode->node);

    initText();

    mainTransform->addChild(panelTransform);
    mainTransform->addChild(potiTransform);

    mainTransform->addChild(textTransform);

    textTransform->addChild(potiText);

    panelTransform->addChild(createPanelNode(poti->getBackgroundTexture()));

    dmat4 panelMatrix;
    dmat4 s, hm, pm, rm;
    pm = rotate(0.0, 1.0, 0.0, 0.0);
    rm = rotate(0.0, 0.0, 1.0, 0.0);
    hm = rotate(vsg::radians(90.0), 0.0, 0.0, 1.0);
    s = scale(30.0, 30.0, 30.0);
    panelMatrix = s * hm * pm * rm;
    setTrans(panelMatrix,dvec3(0.0, 7.0, 2.0));
    panelTransform->matrix = (panelMatrix);

    stateSwitch = Switch::create();
    stateSwitch->addChild(true,icon);
    stateSwitch->addChild(false, iconDisabled);
    stateSwitch->setSingleChildOn(0);

    potiTransform->addChild(stateSwitch);

    myDCS = new VSGVruiTransformNode(mainTransform);

    oldValue = poti->getValue() - 1.0;
    oldButtonText = "";
    oldEnabled = true;
}

/// Initialize text parameters.
void VSGVruiValuePoti::initText()
{
    potiTextLayout->horizontalAlignment = StandardLayout::CENTER_ALIGNMENT;
    potiTextLayout->position = vec3(0.0, 1.0, 0.0);
    potiTextLayout->horizontal = vec3(1.0, 0.0, 0.0);
    potiTextLayout->vertical = vec3(0.0, 1.0, 0.0);
    potiTextLayout->color = vec4(0.85, 0.9, 0.8, 1.0);
    potiTextLayout->outlineWidth = 0.3f;
    potiTextLayout->billboard = false;

    potiTextString->value() = make_string(poti->getButtonText());
    potiText->setup(16, VSGVruiPresets::instance()->options);

    dmat4 s, textMatrix;
    s = scale(12.5, 12.5, 1.);
    textMatrix = s;

    setTrans(textMatrix, dvec3(-2.5, 14., 2.0));
    textTransform->matrix = textMatrix;
}

void VSGVruiValuePoti::resizeGeometry()
{
}

void VSGVruiValuePoti::update()
{

    if (poti->getValue() != oldValue)
    {

        //VRUILOG("VSGVruiValuePoti::update info: updating")

        oldValue = poti->getValue();

        dmat4 rot1, rot2, trans, result;

        float frac;

        setTrans(trans, dvec3(0.0, 0.0, 5.0));

        dmat4 s, hm, pm, rm;
        pm = rotate(vsg::radians(270.0), 1.0, 0.0, 0.0);
        rm = rotate(vsg::radians(0.0), 0.0, 1.0, 0.0);
        hm = rotate(vsg::radians(0.0), 0.0, 0.0, 1.0);
        rot1 = hm * pm * rm;

        coSlopePoti *sPoti = dynamic_cast<coSlopePoti *>(poti);
        if (sPoti)
        {
            pm = rotate(vsg::radians((1.0 - sPoti->convertSlopeToLinear(poti->getValue())) * 360), 0.0, 0.0, 1.0);
            //pm.makeRotate(sPoti->convertSlopeToLinear(-sPoti->getValue())*2*M_PI, 0.0, 0.0, 1.0);
        }
        else
        {

            const float RANGE = 300.0f; // value range in degrees

            if (poti->isInteger())
            {
                frac = (((int)(poti->getValue() + 0.5)) - poti->getMin()) / (poti->getMax() - poti->getMin());
            }
            else if (poti->isDiscrete())
            {
                frac = (poti->discreteValue(poti->getValue()) - poti->getMin()) / (poti->getMax() - poti->getMin());
            }
            else
            {
                frac = (poti->getValue() - poti->getMin()) / (poti->getMax() - poti->getMin());
            }

            pm = rotate(vsg::radians((0.5 - frac) * RANGE), 0.0, 0.0, 1.0);
        }

        s = scale(1.2, 1.2, 1.2);

        rot2 = s * hm * pm * rm;

        result = trans * rot2 * rot1;

        potiTransform->matrix = (result);
    }

    if (poti->getButtonText() != oldButtonText)
    {
        oldButtonText = poti->getButtonText();
        //VRUILOG("VSGVruiValuePoti::update info: setting text " << oldButtonText)
        //text->setText(oldButtonText, String::ENCODING_UTF8);
        potiTextString->value() = make_string(oldButtonText);
        potiText->setup(16, VSGVruiPresets::instance()->options);
    }

    if (poti->isEnabled() != oldEnabled)
    {
        if (poti->isEnabled())
        {
            stateSwitch->setSingleChildOn(0);
        }
        else
        {
            stateSwitch->setSingleChildOn(1);
        }
    }
}

/** Create a node for the poti background texture.
  @param textureName base name of texture file
  @return scene graph node
*/
ref_ptr<Node> VSGVruiValuePoti::createPanelNode(const string &textureName)
{
    ref_ptr<vec3Array> coord = vec3Array::create(4);
    ref_ptr<vec4Array> color = vec4Array::create(1);
    ref_ptr<vec3Array> normal = vec3Array::create(1);
    ref_ptr<vec2Array> texCoord = vec2Array::create(4);
    ref_ptr<uintArray> coordIndices = uintArray::create(6);

    (*coord)[0].set(-1.0f, 1.0f, 0.0f);
    (*coord)[1].set(-1.0f, -1.0f, 0.0f);
    (*coord)[2].set(1.0f, -1.0f, 0.0f);
    (*coord)[3].set(1.0f, 1.0f, 0.0f);

    (*color)[0].set(1.0f, 1.0f, 1.0f, 1.0f);

    (*normal)[0].set(0.0, 0.0, 1.0);

    (*texCoord)[0].set(0.0, 1.0);
    (*texCoord)[1].set(1.0, 1.0);
    (*texCoord)[2].set(1.0, 0.0);
    (*texCoord)[3].set(0.0, 0.0);

    coordIndices = uintArray::create(
        {
            0,1,2 , 0,2,3
        }
    );

    ref_ptr<ShaderSet> shaderSet;
    shaderSet = createFlatShadedShaderSet();

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

    shaderSet->defaultGraphicsPipelineStates.push_back(colorBlendState);
    shaderSet->defaultGraphicsPipelineStates.push_back(depthStencilState);

    DataList vertexArrays;

    ref_ptr<GraphicsPipelineConfigurator> gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

    auto image = VSGVruiRendererInterface::the()->createVsgTexture(textureName);
    if (image && image->data)
        gpConfigurator->assignTexture("diffuseMap", image->data);
    else
        cerr << "No texture could be loaded for poti panel!" << endl;

    gpConfigurator->assignArray(vertexArrays, "vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, coord);
    gpConfigurator->assignArray(vertexArrays, "vsg_Normal", VK_VERTEX_INPUT_RATE_INSTANCE, normal);
    gpConfigurator->assignArray(vertexArrays, "vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, texCoord);
    gpConfigurator->assignArray(vertexArrays, "vsg_Color", VK_VERTEX_INPUT_RATE_INSTANCE, color);

    gpConfigurator->init();

    ref_ptr<StateGroup> stateGroup = StateGroup::create();
    gpConfigurator->copyTo(stateGroup);

    ref_ptr<VertexIndexDraw> vid = VertexIndexDraw::create();
    vid->assignArrays(vertexArrays);
    vid->assignIndices(coordIndices);
    vid->indexCount = static_cast<uint32_t>(coordIndices->size());
    vid->instanceCount = 1;

    stateGroup->addChild(vid);

    return stateGroup;
}
}
