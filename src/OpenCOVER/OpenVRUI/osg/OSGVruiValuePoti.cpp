/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiValuePoti.h>

#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/util/vruiLog.h>
#include <OpenVRUI/osg/NodeDumpVisitor.h>

#include <osg/Array>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>

using namespace osg;
using namespace osgText;
using namespace std;

namespace vrui
{

OSGVruiValuePoti::OSGVruiValuePoti(coValuePoti *poti)
    : OSGVruiUIElement(poti)
    , poti(0)
    , potiTransform(0)
    , textTransform(0)
    , text(0)
    , textNode(0)
    , stateSwitch(0)
{
    this->poti = poti;
}

OSGVruiValuePoti::~OSGVruiValuePoti()
{
}

void OSGVruiValuePoti::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("OSGVruiValuePoti::createGeometry info: making geometry")

    potiTransform = new MatrixTransform();
    textTransform = new MatrixTransform();

    //potiTransform->setName("PotiTransform");
    //textTransform->setName("TextTransform");

    textNode = new Geode();
    //textNode->setName("TextNode");

    ref_ptr<MatrixTransform> mainTransform = new MatrixTransform();
    ref_ptr<MatrixTransform> panelTransform = new MatrixTransform();
    //mainTransform->setName("MainTransform");
    //panelTransform->setName("PanelTransform");

    ref_ptr<Group> icon = new Group();
    ref_ptr<Group> iconDisabled = new Group();
    //icon->setName("Icon");
    //iconDisabled->setName("IconDisabled");

    OSGVruiNode *iconNode = dynamic_cast<OSGVruiNode *>(vruiRendererInterface::the()->getIcon("UI/poti2"));

    if (!iconNode || !iconNode->getNodePtr())
        ; //VRUILOG("OSGVruiValuePoti::createGeometry warn: cannot open icon node UI/poti2")
    else
        icon->addChild(iconNode->getNodePtr());

    iconNode = dynamic_cast<OSGVruiNode *>(vruiRendererInterface::the()->getIcon("UI/poti2Disabled"));
    if (!iconNode || !iconNode->getNodePtr())
        ; //VRUILOG("OSGVruiValuePoti::createGeometry warn: cannot open icon node UI/poti2Disabled")
    else
        iconDisabled->addChild(iconNode->getNodePtr());

    initText();
    text->setText(poti->getButtonText(), String::ENCODING_UTF8);

    mainTransform->addChild(panelTransform.get());
    mainTransform->addChild(potiTransform.get());

    mainTransform->addChild(textTransform.get());

    textTransform->addChild(textNode.get());

    panelTransform->addChild(createPanelNode(poti->getBackgroundTexture()).get());

    Matrix panelMatrix;
    Matrix s, hm, pm, rm;
    pm.makeRotate(osg::inDegrees(0.0), 1.0, 0.0, 0.0);
    rm.makeRotate(osg::inDegrees(0.0), 0.0, 1.0, 0.0);
    hm.makeRotate(osg::inDegrees(90.0), 0.0, 0.0, 1.0);
    s.makeScale(30.0, 30.0, 30.0);
    panelMatrix = rm * pm * hm * s;
    panelMatrix.setTrans(0.0, 7.0, 1.0);
    panelTransform->setMatrix(panelMatrix);

    stateSwitch = new Switch();
    stateSwitch->addChild(icon.get());
    stateSwitch->addChild(iconDisabled.get());
    stateSwitch->setSingleChildOn(0);
    //stateSwitch->setName("StateSwitch");

    potiTransform->addChild(stateSwitch.get());

    myDCS = new OSGVruiTransformNode(mainTransform.get());

    oldValue = poti->getValue() - 1.0;
    oldButtonText = "";
    oldEnabled = true;
}

/// Initialize text parameters.
void OSGVruiValuePoti::initText()
{

    ref_ptr<Material> textMaterial = new Material();
    textMaterial->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    textMaterial->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2, 0.2, 0.2, 1.0));
    textMaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9, 0.9, 0.9, 1.0));
    textMaterial->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9, 0.9, 0.9, 1.0));
    textMaterial->setEmission(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    textMaterial->setShininess(Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<StateSet> textStateSet = textNode->getOrCreateStateSet();

    OSGVruiPresets::makeTransparent(textStateSet);
    textStateSet->setAttributeAndModes(textMaterial.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    textStateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);

    text = new Text();
    text->setFont(OSGVruiPresets::getFontFile());
    text->setDrawMode(Text::TEXT);
    text->setAlignment(Text::LEFT_BASE_LINE);
    text->setColor(Vec4(1.0, 1.0, 1.0, 1.0));
    text->setCharacterSize(10.0f);
    text->setText(poti->getButtonText(), String::ENCODING_UTF8);
    text->setLayout(Text::LEFT_TO_RIGHT);
    text->setAxisAlignment(Text::YZ_PLANE);

    Matrix textMatrix;

    Matrix s, hm, pm, rm;
    pm.makeRotate(osg::inDegrees(0.0), 1.0, 0.0, 0.0);
    rm.makeRotate(osg::inDegrees(-90.0), 0.0, 1.0, 0.0);
    hm.makeRotate(osg::inDegrees(270.0), 0.0, 0.0, 1.0);
    s.makeScale(1.4, 1.4, 1.4);
    textMatrix = rm * pm * hm * s;

    textMatrix.setTrans(-50.0 * 0.4 - 7.0, 50.0 * 0.4 + 7.0, 2.0);

    textTransform->setMatrix(textMatrix);

    textNode->addDrawable(text.get());
}

void OSGVruiValuePoti::resizeGeometry()
{
}

void OSGVruiValuePoti::update()
{

    if (poti->getValue() != oldValue)
    {

        //VRUILOG("OSGVruiValuePoti::update info: updating")

        oldValue = poti->getValue();

        Matrix rot1, rot2, trans, result;

        float frac;

        trans.makeTranslate(0.0, 0.0, 5.0);

        Matrix s, hm, pm, rm;
        pm.makeRotate(osg::inDegrees(270.0), 1.0, 0.0, 0.0);
        rm.makeRotate(osg::inDegrees(0.0), 0.0, 1.0, 0.0);
        hm.makeRotate(osg::inDegrees(0.0), 0.0, 0.0, 1.0);
        rot1 = rm * pm * hm;

        coSlopePoti *sPoti = dynamic_cast<coSlopePoti *>(poti);
        if (sPoti)
        {
            pm.makeRotate(osg::inDegrees((1.0 - sPoti->convertSlopeToLinear(poti->getValue())) * 360), 0.0, 0.0, 1.0);
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

            pm.makeRotate(osg::inDegrees((0.5 - frac) * RANGE), 0.0, 0.0, 1.0);
        }

        s.makeScale(1.2, 1.2, 1.2);
        //rm.makeRotate(osg::inDegrees(0.0), 0.0, 1.0, 0.0);

        rot2 = rm * pm * hm * s;

        result = rot1 * rot2 * trans;

        potiTransform->setMatrix(result);
    }

    if (poti->getButtonText() != oldButtonText)
    {
        oldButtonText = poti->getButtonText();
        //VRUILOG("OSGVruiValuePoti::update info: setting text " << oldButtonText)
        text->setText(oldButtonText, String::ENCODING_UTF8);
    }

    textNode->setNodeMask((poti->isLabelVisible()) ? (~1) : 0);

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
ref_ptr<Geode> OSGVruiValuePoti::createPanelNode(const string &textureName)
{

    ref_ptr<Vec3Array> coord = new Vec3Array(4);
    ref_ptr<Vec4Array> color = new Vec4Array(1);
    ref_ptr<Vec3Array> normal = new Vec3Array(1);
    ref_ptr<Vec2Array> texCoord = new Vec2Array(4);

    (*coord)[0].set(-1.0, 1.0, 0.0f);
    (*coord)[1].set(-1.0, -1.0, 0.0f);
    (*coord)[2].set(1.0, -1.0, 0.0f);
    (*coord)[3].set(1.0, 1.0, 0.0f);

    (*color)[0].set(1.0, 1.0, 1.0, 1.0);

    (*normal)[0].set(0.0, 0.0, 1.0);

    (*texCoord)[0].set(0.0, 0.0);
    (*texCoord)[1].set(1.0, 0.0);
    (*texCoord)[2].set(1.0, 1.0);
    (*texCoord)[3].set(0.0, 1.0);

    // Define a material:
    ref_ptr<Material> material = new Material();
    material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2, 0.2, 0.2, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    material->setShininess(Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<StateSet> stateSet = new StateSet();

    stateSet->setAttributeAndModes(material.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    //VRUILOG("OSGVruiValuePoti::createPanelNode info: loading texture image " << textureName)

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
        VRUILOG("OSGVruiValuePoti::createPanelNode err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = OSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = OSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = OSGVruiPresets::getPolyModeFill();

    ref_ptr<Geode> geometryNode = new Geode();
    //geometryNode->setName("GeometryNode");
    ref_ptr<Geometry> geometry = new Geometry();

    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setColorArray(color.get());
    geometry->setColorBinding(Geometry::BIND_OVERALL);
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    OSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    geometryNode->setName(textureName);

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;
}
}
