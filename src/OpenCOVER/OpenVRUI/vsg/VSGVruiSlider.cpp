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

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <vsg/all.h>
#include <vsgXchange/all.h>

#define ZOFFSET 5

using namespace std;
using namespace vsg;

namespace vrui
{

VSGVruiSlider::VSGVruiSlider(coSlider *slider)
    : VSGVruiUIElement(slider)
    , sliderDialSize(0.0f)
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

    normal = 0;

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
    if (slider->getShowValue())
    {
        sliderTransform->addChild(createText());
    }
    transform->addChild(switchDial);
    resizeGeometry();
}

/** This method is called whenever the GUI element containing the slider changes its size.
  The method resizes dial and position indicator and recomputes the respective Geodes.
*/
void VSGVruiSlider::resizeGeometry()
{

    float dialSize = slider->getDialSize();
    float myWidth = slider->getWidth();

  /*  (*coord1)[0].set(-dialSize, -dialSize, 0.0f);
    (*coord1)[1].set(dialSize, -dialSize, 0.0f);
    (*coord1)[2].set(dialSize, dialSize, 0.0f);
    (*coord1)[3].set(-dialSize, dialSize, 0.0f);

    (*coord2)[0].set(dialSize, dialSize, 0.0f);
    (*coord2)[1].set(myWidth - dialSize, dialSize, 0.0f);
    (*coord2)[2].set(myWidth - dialSize, dialSize * 2.0f, 0.0f);
    (*coord2)[3].set(dialSize, dialSize * 2.0f, 0.0f);

    positionNode->dirtyBound();
    dialNode->dirtyBound();*/

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
      /*  (*texCoord2)[0].set(1.0f / 64.0f, 0.0f);
        (*texCoord2)[1].set(1.0f / 64.0f + (numTicks / 5.0f), 0.0f);
        (*texCoord2)[2].set(1.0f / 64.0f + (numTicks / 5.0f), 1.0f);
        (*texCoord2)[3].set(1.0f / 64.0f, 1.0f);*/
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

    /*if (!numberText)
    {
        textNode = new Geode();

        numberText = new Text();
        numberText->setFont(VSGVruiPresets::getFontFile());
        numberText->setDrawMode(Text::TEXT);
        numberText->setColor(vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));

        numberText->setAlignment(Text::CENTER_BASE_LINE);
        numberText->setLayout(Text::LEFT_TO_RIGHT);
        numberText->setAxisAlignment(Text::XY_PLANE);

        sliderDialSize = slider->getDialSize();
        numberText->setCharacterSize(sliderDialSize * 2.0f);

        textNode->setStateSet(VSGVruiPresets::getStateSetCulled(coUIElement::YELLOW));
        textNode->addDrawable(numberText.get());
    }

    if (sliderDialSize != slider->getDialSize())
    {
        sliderDialSize = slider->getDialSize();
        numberText->setCharacterSize(sliderDialSize * 2.0f);
    }*/

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

   // numberText->setText(number);

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

    //numberText->setPosition(position);
    auto layout = vsg::StandardLayout::create();
    layout->horizontalAlignment = vsg::StandardLayout::CENTER_ALIGNMENT;
    layout->position = vsg::vec3(6.0, 0.0, 0.0);
    layout->horizontal = vsg::vec3(1.0, 0.0, 0.0);
    layout->vertical = vsg::vec3(0.0, 1.0, 0.0);
    layout->color = vsg::vec4(1.0, 1.0, 1.0, 1.0);
    layout->outlineWidth = 0.1f;
    layout->billboard = true;

    textNode = vsg::Text::create();
    textNode->text = vsg::stringValue::create(number);
    textNode->font = VSGVruiPresets::instance()->font;
    textNode->layout = layout;
    textNode->setup(0, VSGVruiPresets::instance()->options);

    return textNode;
}

/** This method creates the visual representation of the slider position
  indicator. Several attributes need to be set accordingly,
  a texture map is used for the slider button image.
  @param textureName defines the texture image to use for the slider
*/
ref_ptr<Node> VSGVruiSlider::createSlider(const string &textureName)
{
    float dialSize = slider->getDialSize();

   /* if (coord1 == 0)
    {
        coord1 = new vec3Array(4);
        (*coord1)[0].set(-dialSize, -dialSize, 0.0f);
        (*coord1)[1].set(dialSize, -dialSize, 0.0f);
        (*coord1)[2].set(dialSize, dialSize, 0.0f);
        (*coord1)[3].set(-dialSize, dialSize, 0.0f);
    }

    if (normal == 0)
    {
        normal = new vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }

    if (texCoord1 == 0)
    {
        texCoord1 = new vec2Array(4);
        (*texCoord1)[0].set(0.0f, 0.0f);
        (*texCoord1)[1].set(1.0f, 0.0f);
        (*texCoord1)[2].set(1.0f, 1.0f);
        (*texCoord1)[3].set(0.0f, 1.0f);
    }

    vsg::ref_ptr<vsg::Geometry> node = new Geode();
    vsg::ref_ptr<vsg::Geometry> geometry = new Geometry();

    geometry->setVertexArray(coord1.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord1.get());

    ref_ptr<StateSet> stateSet = new StateSet();

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    ref_ptr<Texture2D> texture = oTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(oTex);

    if (texture.valid())
    {
        texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texture->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP);
    }
    else
    {
        VRUILOG("VSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = VSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = VSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = VSGVruiPresets::getPolyModeFill();

    VSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    node->setStateSet(stateSet.get());
    node->addDrawable(geometry.get());*/

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
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
/*
    if (coord2 == 0)
    {
        coord2 = new vec3Array(4);
        (*coord2)[0].set(0.0f, dialSize, 0.0f);
        (*coord2)[1].set(myWidth, dialSize, 0.0f);
        (*coord2)[2].set(myWidth, dialSize * 2.0f, 0.0f);
        (*coord2)[3].set(0.0f, dialSize * 2.0f, 0.0f);
    }

    if (normal == 0)
    {
        normal = new vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }

    if (texCoord2 == 0)
    {
        texCoord2 = new vec2Array(4);
        (*texCoord2)[0].set(0.0f, 0.0f);
        (*texCoord2)[1].set(1.0f, 0.0f);
        (*texCoord2)[2].set(1.0f, 1.0f);
        (*texCoord2)[3].set(0.0f, 1.0f);
    }

    vsg::ref_ptr<vsg::Geometry> node = new Geode();
    vsg::ref_ptr<vsg::Geometry> geometry = new Geometry();

    geometry->setVertexArray(coord2.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord2.get());

    ref_ptr<StateSet> stateSet = new StateSet();

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    ref_ptr<Texture2D> texture = oTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(oTex);

    if (texture.valid())
    {
        texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP);
    }
    else
    {
        VRUILOG("VSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = VSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = VSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = VSGVruiPresets::getPolyModeFill();

    VSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    node->setStateSet(stateSet.get());
    node->addDrawable(geometry.get());*/

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();

    return node;
}
}
