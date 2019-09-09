/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiSlider.h>

#include <OpenVRUI/coSlider.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <osg/Version>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

#define ZOFFSET 5

using namespace std;
using namespace osg;
using namespace osgText;

namespace vrui
{

OSGVruiSlider::OSGVruiSlider(coSlider *slider)
    : OSGVruiUIElement(slider)
    , coord1(0)
    , coord2(0)
    , normal(0)
    , texCoord1(0)
    , texCoord2(0)
    , positionNode(0)
    , dialNode(0)
    , positionNodeDisabled(0)
    , dialNodeDisabled(0)
    , numberText(0)
    , sliderDialSize(0.0f)
{
    this->slider = slider;
}

/** Destructor.
  The slider is removed from all parents to which it is attached.
*/
OSGVruiSlider::~OSGVruiSlider()
{
}

void OSGVruiSlider::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("OSGVruiSlider::createGeometry info: creating geometry")

    ref_ptr<MatrixTransform> transform = new MatrixTransform();
    myDCS = new OSGVruiTransformNode(transform.get());

    myDCS->setName("coSlider");

    sliderTransform = new MatrixTransform();
    transform->addChild(sliderTransform.get());

    normal = 0;

    positionNode = createSlider("UI/slider");
    positionNodeDisabled = createSlider("UI/slider-disabled");
    dialNode = createDial("UI/scale");
    dialNodeDisabled = createDial("UI/scale-disabled");

    switchPosition = new Switch();
    switchPosition->addChild(positionNode.get());
    switchPosition->addChild(positionNodeDisabled.get());
    switchPosition->setSingleChildOn(0);

    switchDial = new Switch();
    switchDial->addChild(dialNode.get());
    switchDial->addChild(dialNodeDisabled.get());
    switchDial->setSingleChildOn(0);

    sliderTransform->addChild(switchPosition.get());
    if (slider->getShowValue())
    {
        sliderTransform->addChild(createText().get());
    }
    transform->addChild(switchDial.get());
    resizeGeometry();
}

/** This method is called whenever the GUI element containing the slider changes its size.
  The method resizes dial and position indicator and recomputes the respective Geodes.
*/
void OSGVruiSlider::resizeGeometry()
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

    positionNode->dirtyBound();
    dialNode->dirtyBound();

    updateSlider();
    updateDial();
}

void OSGVruiSlider::update()
{
    //VRUILOG("OSGVruiSlider::update info: called")
    updateSlider();
    updateDial();
}

/// This routine regenerates the texture mapping for the dial.
void OSGVruiSlider::updateDial()
{
    if (slider->getMax() >= slider->getMin())
    {
        float numTicks = slider->getNumTicks();
        (*texCoord2)[0].set(1.0f / 64.0f, 0.0f);
        (*texCoord2)[1].set(1.0f / 64.0f + (numTicks / 5.0f), 0.0f);
        (*texCoord2)[2].set(1.0f / 64.0f + (numTicks / 5.0f), 1.0f);
        (*texCoord2)[3].set(1.0f / 64.0f, 1.0f);
    }
}

/// This routine resets the location of the position indicator according to the current slider value.
void OSGVruiSlider::updateSlider()
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
        VRUILOG("OSGVruiSlider::updateSlider info: precision = " << precision
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

    Matrix mat = sliderTransform->getMatrix();
    mat.makeTranslate(((myWidth - 2.0f * dialSize) * r) + dialSize, dialSize, ZOFFSET);
    sliderTransform->setMatrix(mat);

    if (slider->getShowValue())
    {
        createText(((myWidth - 2.0f * dialSize) * r) + dialSize);
    }
}

/** This routine creates the text for the slider value and sets the correct position
  relative to the position indicator.
  @param xPos position of slider value string
*/
ref_ptr<Geode> OSGVruiSlider::createText(float xPos)
{

    if (!numberText)
    {
        textNode = new Geode();

        numberText = new Text();
        numberText->setFont(OSGVruiPresets::getFontFile());
        numberText->setDrawMode(Text::TEXT);
        numberText->setColor(Vec4(1.0f, 1.0f, 1.0f, 1.0f));

        numberText->setAlignment(Text::CENTER_BASE_LINE);
        numberText->setLayout(Text::LEFT_TO_RIGHT);
        numberText->setAxisAlignment(Text::XY_PLANE);

        sliderDialSize = slider->getDialSize();
        numberText->setCharacterSize(sliderDialSize * 2.0f);

        textNode->setStateSet(OSGVruiPresets::getStateSetCulled(coUIElement::YELLOW));
        textNode->addDrawable(numberText.get());
    }

    if (sliderDialSize != slider->getDialSize())
    {
        sliderDialSize = slider->getDialSize();
        numberText->setCharacterSize(sliderDialSize * 2.0f);
    }

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

    numberText->setText(number, String::ENCODING_UTF8);
    numberText->dirtyBound();

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    BoundingBox stringBoundingBox = numberText->getBoundingBox();
#else
    BoundingBox stringBoundingBox = numberText->getBound();
#endif
    Vec3 position;

    float myWidth = slider->getWidth();
    float dialSize = slider->getDialSize();

    float xSize = stringBoundingBox.xMax() - stringBoundingBox.xMin();

    if (xPos - xSize + dialSize < 0.0f)
    {
        position = Vec3(xSize - xPos - dialSize, 2.0f * dialSize, 0.0f);
    }
    else if (xPos + xSize - dialSize > myWidth)
    {
        position = Vec3(myWidth - xPos - xSize + dialSize, 2.0f * dialSize, 0.0f);
    }
    else
    {
        position = Vec3(0.0f, 2.0f * dialSize, 0.0f);
    }

    numberText->setPosition(position);

    return textNode;
}

/** This method creates the visual representation of the slider position
  indicator. Several attributes need to be set accordingly,
  a texture map is used for the slider button image.
  @param textureName defines the texture image to use for the slider
*/
ref_ptr<Geode> OSGVruiSlider::createSlider(const string &textureName)
{
    float dialSize = slider->getDialSize();

    if (coord1 == 0)
    {
        coord1 = new Vec3Array(4);
        (*coord1)[0].set(-dialSize, -dialSize, 0.0f);
        (*coord1)[1].set(dialSize, -dialSize, 0.0f);
        (*coord1)[2].set(dialSize, dialSize, 0.0f);
        (*coord1)[3].set(-dialSize, dialSize, 0.0f);
    }

    if (normal == 0)
    {
        normal = new Vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }

    if (texCoord1 == 0)
    {
        texCoord1 = new Vec2Array(4);
        (*texCoord1)[0].set(0.0f, 0.0f);
        (*texCoord1)[1].set(1.0f, 0.0f);
        (*texCoord1)[2].set(1.0f, 1.0f);
        (*texCoord1)[3].set(0.0f, 1.0f);
    }

    osg::ref_ptr<osg::Geode> node = new Geode();
    osg::ref_ptr<osg::Geometry> geometry = new Geometry();

    geometry->setVertexArray(coord1.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord1.get());

    ref_ptr<StateSet> stateSet = new StateSet();

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
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
        VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = OSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = OSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = OSGVruiPresets::getPolyModeFill();

    OSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    node->setStateSet(stateSet.get());
    node->addDrawable(geometry.get());

    return node;
}

/** Creates the visual representation of the slider dial, which is
  a texture map, composited of textures of tick marks and horizontal lines.
  @param textureName defines the texture image to use for the composition
*/
ref_ptr<Geode> OSGVruiSlider::createDial(const string &textureName)
{
    float myWidth = slider->getWidth();
    float dialSize = slider->getDialSize();

    if (coord2 == 0)
    {
        coord2 = new Vec3Array(4);
        (*coord2)[0].set(0.0f, dialSize, 0.0f);
        (*coord2)[1].set(myWidth, dialSize, 0.0f);
        (*coord2)[2].set(myWidth, dialSize * 2.0f, 0.0f);
        (*coord2)[3].set(0.0f, dialSize * 2.0f, 0.0f);
    }

    if (normal == 0)
    {
        normal = new Vec3Array(1);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }

    if (texCoord2 == 0)
    {
        texCoord2 = new Vec2Array(4);
        (*texCoord2)[0].set(0.0f, 0.0f);
        (*texCoord2)[1].set(1.0f, 0.0f);
        (*texCoord2)[2].set(1.0f, 1.0f);
        (*texCoord2)[3].set(0.0f, 1.0f);
    }

    osg::ref_ptr<osg::Geode> node = new Geode();
    osg::ref_ptr<osg::Geometry> geometry = new Geometry();

    geometry->setVertexArray(coord2.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord2.get());

    ref_ptr<StateSet> stateSet = new StateSet();

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
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
        VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = OSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = OSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = OSGVruiPresets::getPolyModeFill();

    OSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    node->setStateSet(stateSet.get());
    node->addDrawable(geometry.get());

    return node;
}
}
