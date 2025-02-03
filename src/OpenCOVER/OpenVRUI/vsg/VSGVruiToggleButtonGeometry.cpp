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

#define STYLE_IN 1
#define STYLE_OUT 2

using namespace std;
using namespace vsg;

namespace vrui
{

float VSGVruiToggleButtonGeometry::A = 30.0f;
float VSGVruiToggleButtonGeometry::B = 50.0f;
float VSGVruiToggleButtonGeometry::D = 5.0f;


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

  /*  if (coord == 0)
    {

        coord = new vec3Array(4);
        normal = new vec3Array(1);
        texCoord = new vec2Array(4);

        // 3D coordinates used for textures
        (*coord)[3].set(0.0f, A, 0.0f);
        (*coord)[2].set(A, A, 0.0f);
        (*coord)[1].set(A, 0.0f, 0.0f);
        (*coord)[0].set(0.0f, 0.0f, 0.0f);

        // 2D coordinates valid for all textures
        (*texCoord)[0].set(0.0f, 0.0f);
        (*texCoord)[1].set(1.0f, 0.0f);
        (*texCoord)[2].set(1.0f, 1.0f);
        (*texCoord)[3].set(0.0f, 1.0f);

        // valid for all textures
        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }*/
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
    }
}

ref_ptr<Node> VSGVruiToggleButtonGeometry::createNode(const string &textureName, bool checkTexture)
{

    createSharedLists();

    /*ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Geode> geometryNode = new Geode();
    ref_ptr<StateSet> stateSet = new StateSet();

    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    if (checkTexture)
    {
        stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE_NL), StateAttribute::ON | StateAttribute::PROTECTED);
    }
    else
    {
        stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);
    }

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    texture = oTex->getTexture();
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

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;*/
    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
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
