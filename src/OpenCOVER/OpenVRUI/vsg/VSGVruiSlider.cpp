/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiSlider.h>

#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/vsg/VSGVruiRendererInterface.h>

#include <vsg/all.h>
#include <vsgXchange/all.h>

#define ZOFFSET 5

using namespace std;
using namespace vsg;

namespace vrui
{

ref_ptr<uintArray> VSGVruiSlider::coordIndices = nullptr;
ref_ptr<vec3Array> VSGVruiSlider::normals = nullptr;
ref_ptr<vec4Array> VSGVruiSlider::colors = nullptr; 

VSGVruiSlider::VSGVruiSlider(coSlider *slider)
    : VSGVruiUIElement(slider)
    , sliderDialSize(0.0f)
    , initiallyCompiled(0)
{
    this->slider = slider;
}

/** Destructor.
  The slider is removed from all parents to which it is attached.
*/
VSGVruiSlider::~VSGVruiSlider()
{
}

void VSGVruiSlider::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("VSGVruiSlider::createGeometry info: creating geometry")

    ref_ptr<MatrixTransform> transform = MatrixTransform::create();
    myDCS = new VSGVruiTransformNode(transform);

    myDCS->setName("coSlider");

    sliderTransform = new MatrixTransform();
    transform->addChild(sliderTransform);

    positionNode = createSlider("UI/slider");
    positionNodeDisabled = createSlider("UI/slider-disabled");
    dialNode = createDial("UI/scale");
    dialNodeDisabled = createDial("UI/scale-disabled");

    switchPosition = Switch::create();
    switchPosition->addChild(true,positionNode);
    switchPosition->addChild(false,positionNodeDisabled);
    switchPosition->setSingleChildOn(0);

    switchDial = Switch::create();
    switchDial->addChild(true,dialNode);
    switchDial->addChild(false,dialNodeDisabled);
    switchDial->setSingleChildOn(0);

    sliderTransform->addChild(switchPosition);

    sliderText = Text::create();
    sliderTextLayout = StandardLayout::create();
    sliderTextString = stringValue::create();

    sliderText->technique = GpuLayoutTechnique::create();
    
    sliderText->font = VSGVruiPresets::instance()->font2;
    sliderText->text = sliderTextString;
    sliderText->layout = sliderTextLayout;

    if (slider->getShowValue())
    {
        sliderTransform->addChild(createText());
    }

    transform->addChild(switchDial);
    transform->setValue("name", "VSGVruiSlider");
    transform->setValue("coUIElement", slider);

    resizeGeometry();
}

void VSGVruiSlider::createSharedLists()
{
    if (!normals)
    {
        normals = vec3Array::create(4, vec3{ 0.0f, 0.0f, 1.0f });
    }

    if (!coordIndices)
    {
        coordIndices = uintArray::create(
            {
                0,1,2 , 0,2,3
            }
        );
    }

    if (!colors) 
    {
        colors = vec4Array::create(4, vec4{ 0.9f, 0.9f, 0.9f, 0.9f });
    }
}

/** This method is called whenever the GUI element containing the slider changes its size.
  The method resizes dial and position indicator and recomputes the respective Geodes.
*/
void VSGVruiSlider::resizeGeometry()
{

    float dialSize = slider->getDialSize();
    float myWidth = slider->getWidth();

    (*coord1)[0].set(-dialSize, -dialSize, 0.0f);
    (*coord1)[1].set(dialSize, -dialSize, 0.0f);
    (*coord1)[2].set(dialSize, dialSize, 0.0f);
    (*coord1)[3].set(-dialSize, dialSize, 0.0f);

    (*coord2)[0].set(dialSize, dialSize, 0.0f);
    (*coord2)[1].set(myWidth - dialSize, dialSize, 0.0f);
    (*coord2)[2].set(myWidth - dialSize, dialSize * 2.0f, 0.0f);
    (*coord2)[3].set(dialSize, dialSize * 2.0f, 0.0f);

    coord1->dirty();
    coord2->dirty();

    if (!initiallyCompiled)
    {
        initiallyCompiled = vruiRendererInterface::the()->compileNode(myDCS);
    }
    else
    {
        if ((*positionNodeVid).arrays[0]->buffer)
        {
            VSGVruiRendererInterface::the()->addToTransfer(positionNodeVid->arrays[0]);
        }
        if ((*dialNodeVid).arrays[0]->buffer)
        {
            VSGVruiRendererInterface::the()->addToTransfer(dialNodeVid->arrays[0]);
        }
        
    }

    updateSlider();
    updateDial();
}

void VSGVruiSlider::update()
{
    //VRUILOG("VSGVruiSlider::update info: called")
    updateSlider();
    updateDial();
}

/// This routine regenerates the texture mapping for the dial.
void VSGVruiSlider::updateDial()
{
    if (slider->getMax() >= slider->getMin())
    {
        float numTicks = slider->getNumTicks();
        (*texCoord2)[0].set(1.0f / 64.0f, 1.0f);
        (*texCoord2)[1].set(1.0f / 64.0f + (numTicks / 5.0f), 1.0f);
        (*texCoord2)[2].set(1.0f / 64.0f + (numTicks / 5.0f), 0.0f);
        (*texCoord2)[3].set(1.0f / 64.0f, 0.0f);

        texCoord2->dirty();

        if (!initiallyCompiled)
        {
            initiallyCompiled = vruiRendererInterface::the()->compileNode(myDCS);
        }
        else
        {
            if ((*dialNodeVid).arrays[2]->buffer)
            {
                VSGVruiRendererInterface::the()->addToTransfer(dialNodeVid->arrays[2]);
            }
        }
    }
}

/// This routine resets the location of the position indicator according to the current slider value.
void VSGVruiSlider::updateSlider()
{
    float r, step;

    float value = slider->getLinearValue();
    float minVal = slider->getLinearMin();
    float maxVal = slider->getLinearMax();
    int precision = slider->getPrecision();

    slider->adjustSlider(minVal, maxVal, value, step, precision);

    if (precision < 0 || precision > 15)
        precision = 0; //// @@@ change to exponential format if too high
    if (precision > 30)
    {
        VRUILOG("VSGVruiSlider::updateSlider info: precision = " << precision
                                                                 << ", numerical problems in slider, please check plugin!")
        precision = 1;
    }

    if ((maxVal - minVal) > 0.0f)
    {
        r = (value - minVal) / (maxVal - minVal);
    }
    else
    {
        r = 0.5f;
    }

    float myWidth = slider->getWidth();
    float dialSize = slider->getDialSize();

    sliderTransform->matrix = translate((double)((myWidth - 2.0f * dialSize) * r) + dialSize, (double)dialSize, (double)ZOFFSET);

    if (slider->getShowValue())
    {
        createText(((myWidth - 2.0f * dialSize) * r) + dialSize);
    }
}

/** This routine creates the text for the slider value and sets the correct position
  relative to the position indicator.
  @param xPos position of slider value string
*/
ref_ptr<Node> VSGVruiSlider::createText(float xPos)
{

    char number[200];
    float value = slider->getValue();
    int precision = slider->getPrecision();

    if (slider->isInteger())
    {
        sprintf(number, "%d", (int)value);
    }
    else
    {
        sprintf(number, "%.*f", precision, value);
    }

    vec3 position;

    float myWidth = slider->getWidth();
    float dialSize = slider->getDialSize();

    float xSize=50;// = stringBoundingBox.xMax() - stringBoundingBox.xMin();

    if (xPos - xSize + dialSize < 0.0f)
    {
        position = vec3(xSize - xPos - dialSize, 2.0f * dialSize, 0.0f);
    }
    else if (xPos + xSize - dialSize > myWidth)
    {
        position = vec3(myWidth - xPos - xSize + dialSize, 2.0f * dialSize, 0.0f);
    }
    else
    {
        position = vec3(0.0f, 2.0f * dialSize, 0.0f);
    }

    sliderTextLayout->horizontalAlignment = StandardLayout::CENTER_ALIGNMENT;
    sliderTextLayout->position = vec3(0.0, 1.0, 0.0);
    sliderTextLayout->horizontal = vec3(1.0, 0.0, 0.0);
    sliderTextLayout->vertical = vec3(0.0, 1.0, 0.0);
    sliderTextLayout->color = vec4(1.0, 1.0, 1.0, 1.0);
    sliderTextLayout->outlineWidth = 0.1f;
    sliderTextLayout->billboard = false;

    sliderTextString->value() = make_string(number);
    sliderText->setup(32, VSGVruiPresets::instance()->options);

    ref_ptr<MatrixTransform> textNode = MatrixTransform::create();
    textNode->addChild(sliderText); 
    textNode->matrix = scale((double)dialSize * 2, (double)dialSize * 2, (double)1.0f);
    
    return textNode;
}

/** This method creates the visual representation of the slider position
  indicator. Several attributes need to be set accordingly,
  a texture map is used for the slider button image.
  @param textureName defines the texture image to use for the slider
*/
ref_ptr<Node> VSGVruiSlider::createSlider(const string &textureName)
{
    createSharedLists();

    float dialSize = slider->getDialSize();

    if (!coord1)
    {
        coord1 = new vec3Array(4);
        (*coord1)[0].set(-dialSize, -dialSize, 0.0f);
        (*coord1)[1].set(dialSize, -dialSize, 0.0f);
        (*coord1)[2].set(dialSize, dialSize, 0.0f);
        (*coord1)[3].set(-dialSize, dialSize, 0.0f);
    }

    if (!texCoord1)
    {
        texCoord1 = new vec2Array(4);
        (*texCoord1)[0].set(0.0f, 1.0f);
        (*texCoord1)[1].set(1.0f, 1.0f);
        (*texCoord1)[2].set(1.0f, 0.0f);
        (*texCoord1)[3].set(0.0f, 0.0f);
    }

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    node->addChild(createNode(textureName, coord1, texCoord1, true));

    return node;
}

/** Creates the visual representation of the slider dial, which is
  a texture map, composited of textures of tick marks and horizontal lines.
  @param textureName defines the texture image to use for the composition
*/
ref_ptr<Node> VSGVruiSlider::createDial(const string &textureName)
{
    float myWidth = slider->getWidth();
    float dialSize = slider->getDialSize();

    if (!coord2)
    {
        coord2 = new vec3Array(4);
        (*coord2)[0].set(0.0f, dialSize, 0.0f);
        (*coord2)[1].set(myWidth, dialSize, 0.0f);
        (*coord2)[2].set(myWidth, dialSize * 2.0f, 0.0f);
        (*coord2)[3].set(0.0f, dialSize * 2.0f, 0.0f);
    }

    if (!texCoord2)
    {
        texCoord2 = new vec2Array(4);
        (*texCoord2)[0].set(0.0f, 1.0f);
        (*texCoord2)[1].set(1.0f, 1.0f);
        (*texCoord2)[2].set(1.0f, 0.0f);
        (*texCoord2)[3].set(0.0f, 0.0f);
    }

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    node->addChild(createNode(textureName, coord2, texCoord2, false));

    return node;    
}

ref_ptr<Node> VSGVruiSlider::createNode(const string& textureName, ref_ptr<vec3Array> coord, ref_ptr<vec2Array> texCoord, bool isSlider)
{
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
    
    if (isSlider)
    {
        depthStencilState->depthWriteEnable = VK_FALSE;
    }
    
    depthStencilState->depthCompareOp = VK_COMPARE_OP_GREATER;

    shaderSet->defaultGraphicsPipelineStates.push_back(colorBlendState);
    shaderSet->defaultGraphicsPipelineStates.push_back(depthStencilState);

    DataList vertexArrays;


    ref_ptr<GraphicsPipelineConfigurator> gpConfigurator = GraphicsPipelineConfigurator::create(shaderSet);

    auto image = VSGVruiRendererInterface::the()->createVsgTexture(textureName);
    if (image && image->data)
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

    if (isSlider)
    {
        positionNodeVid = vid;
        stateGroup->addChild(positionNodeVid);
    }
    else
    {
        dialNodeVid = vid;
        stateGroup->addChild(dialNodeVid);
    }

    return stateGroup;
}
}
