/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiValuePoti.h>

#include <OpenVRUI/coValuePoti.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/util/vruiLog.h>

#include <vsg/nodes/MatrixTransform.h>
#include <vsg/nodes/Switch.h>
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

    //potiTransform->setName("PotiTransform");
    //textTransform->setName("TextTransform");

    text = Text::create();
    //textNode->setName("TextNode");

    ref_ptr<MatrixTransform> mainTransform = MatrixTransform::create();
    ref_ptr<MatrixTransform> panelTransform = MatrixTransform::create();
    //mainTransform->setName("MainTransform");
    //panelTransform->setName("PanelTransform");

    ref_ptr<Group> icon = Group::create();
    ref_ptr<Group> iconDisabled = Group::create();
    //icon->setName("Icon");
    //iconDisabled->setName("IconDisabled");

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
    text->text = vsg::stringValue::create(poti->getButtonText());

    mainTransform->addChild(panelTransform);
    mainTransform->addChild(potiTransform);

    mainTransform->addChild(textTransform);

    textTransform->addChild(text);

    panelTransform->addChild(createPanelNode(poti->getBackgroundTexture()));

    dmat4 panelMatrix;
    dmat4 s, hm, pm, rm;
    pm = rotate(0.0, 1.0, 0.0, 0.0);
    rm = rotate(0.0, 0.0, 1.0, 0.0);
    hm = rotate(0.0, 0.0, 0.0, 1.0);
    s = scale(30.0, 30.0, 30.0);
    panelMatrix = s*hm*pm*rm;
    setTrans(panelMatrix,dvec3(0.0, 7.0, 1.0));
    panelTransform->matrix = (panelMatrix);

    stateSwitch = Switch::create();
    stateSwitch->addChild(true,icon);
    stateSwitch->addChild(false, iconDisabled);
    stateSwitch->setSingleChildOn(0);
    //stateSwitch->setName("StateSwitch");

    potiTransform->addChild(stateSwitch);

    myDCS = new VSGVruiTransformNode(mainTransform);

    oldValue = poti->getValue() - 1.0;
    oldButtonText = "";
    oldEnabled = true;
}

/// Initialize text parameters.
void VSGVruiValuePoti::initText()
{

    /*ref_ptr<Material> textMaterial = new Material();
    textMaterial->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    textMaterial->setAmbient(Material::FRONT_AND_BACK, vsg::vec4(0.2, 0.2, 0.2, 1.0));
    textMaterial->setDiffuse(Material::FRONT_AND_BACK, vsg::vec4(0.9, 0.9, 0.9, 1.0));
    textMaterial->setSpecular(Material::FRONT_AND_BACK, vsg::vec4(0.9, 0.9, 0.9, 1.0));
    textMaterial->setEmission(Material::FRONT_AND_BACK, vsg::vec4(0.0, 0.0, 0.0, 1.0));
    textMaterial->setShininess(Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<StateSet> textStateSet = textNode->getOrCreateStateSet();

    VSGVruiPresets::makeTransparent(textStateSet);
    textStateSet->setAttributeAndModes(textMaterial.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    textStateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);

    text = new Text();
    text->setFont(VSGVruiPresets::getFontFile());
    text->setDrawMode(Text::TEXT);
    text->setAlignment(Text::LEFT_BASE_LINE);
    text->setColor(vsg::vec4(1.0, 1.0, 1.0, 1.0));
    text->setCharacterSize(10.0f);
    text->setText(poti->getButtonText(), String::ENCODING_UTF8);
    text->setLayout(Text::LEFT_TO_RIGHT);
    text->setAxisAlignment(Text::YZ_PLANE);

    Matrix textMatrix;

    Matrix s, hm, pm, rm;
    pm.makeRotate(vsg::inDegrees(0.0), 1.0, 0.0, 0.0);
    rm.makeRotate(vsg::inDegrees(-90.0), 0.0, 1.0, 0.0);
    hm.makeRotate(vsg::inDegrees(270.0), 0.0, 0.0, 1.0);
    s.makeScale(1.4, 1.4, 1.4);
    textMatrix = rm * pm * hm * s;

    textMatrix.setTrans(-50.0 * 0.4 - 7.0, 50.0 * 0.4 + 7.0, 2.0);

    textTransform->setMatrix(textMatrix);

    textNode->addDrawable(text.get());*/
}

void VSGVruiValuePoti::resizeGeometry()
{
}

void VSGVruiValuePoti::update()
{

   /* if (poti->getValue() != oldValue)
    {

        //VRUILOG("VSGVruiValuePoti::update info: updating")

        oldValue = poti->getValue();

        Matrix rot1, rot2, trans, result;

        float frac;

        trans.makeTranslate(0.0, 0.0, 5.0);

        Matrix s, hm, pm, rm;
        pm.makeRotate(vsg::inDegrees(270.0), 1.0, 0.0, 0.0);
        rm.makeRotate(vsg::inDegrees(0.0), 0.0, 1.0, 0.0);
        hm.makeRotate(vsg::inDegrees(0.0), 0.0, 0.0, 1.0);
        rot1 = rm * pm * hm;

        coSlopePoti *sPoti = dynamic_cast<coSlopePoti *>(poti);
        if (sPoti)
        {
            pm.makeRotate(vsg::inDegrees((1.0 - sPoti->convertSlopeToLinear(poti->getValue())) * 360), 0.0, 0.0, 1.0);
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

            pm.makeRotate(vsg::inDegrees((0.5 - frac) * RANGE), 0.0, 0.0, 1.0);
        }

        s.makeScale(1.2, 1.2, 1.2);
        //rm.makeRotate(vsg::inDegrees(0.0), 0.0, 1.0, 0.0);

        rot2 = rm * pm * hm * s;

        result = rot1 * rot2 * trans;

        potiTransform->setMatrix(result);
    }

    if (poti->getButtonText() != oldButtonText)
    {
        oldButtonText = poti->getButtonText();
        //VRUILOG("VSGVruiValuePoti::update info: setting text " << oldButtonText)
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
    }*/
}

/** Create a node for the poti background texture.
  @param textureName base name of texture file
  @return scene graph node
*/
ref_ptr<Node> VSGVruiValuePoti::createPanelNode(const string &textureName)
{

   /* ref_ptr<vec3Array> coord = new vec3Array(4);
    ref_ptr<vsg::vec4Array> color = new vsg::vec4Array(1);
    ref_ptr<vec3Array> normal = new vec3Array(1);
    ref_ptr<vec2Array> texCoord = new vec2Array(4);

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
    material->setAmbient(Material::FRONT_AND_BACK, vsg::vec4(0.2, 0.2, 0.2, 1.0));
    material->setDiffuse(Material::FRONT_AND_BACK, vsg::vec4(1.0, 1.0, 1.0, 1.0));
    material->setSpecular(Material::FRONT_AND_BACK, vsg::vec4(1.0, 1.0, 1.0, 1.0));
    material->setEmission(Material::FRONT_AND_BACK, vsg::vec4(0.0, 0.0, 0.0, 1.0));
    material->setShininess(Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<StateSet> stateSet = new StateSet();

    stateSet->setAttributeAndModes(material.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    //VRUILOG("VSGVruiValuePoti::createPanelNode info: loading texture image " << textureName)

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
        VRUILOG("VSGVruiValuePoti::createPanelNode err: texture image " << textureName << " not found")
    }

    ref_ptr<TexEnv> texEnv = VSGVruiPresets::getTexEnvModulate();
    ref_ptr<CullFace> cullFace = VSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = VSGVruiPresets::getPolyModeFill();

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

    VSGVruiPresets::makeTransparent(stateSet);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    stateSet->setTextureAttribute(0, texEnv.get());
    stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    geometryNode->setName(textureName);

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;*/

    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    return node;
}
}
