/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiToggleButtonGeometry.h>

#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <osg/Geometry>
#include <osg/Material>
#include <osgDB/ReadFile>

#define STYLE_IN 1
#define STYLE_OUT 2

using namespace std;
using namespace osg;

namespace vrui
{

float OSGVruiToggleButtonGeometry::A = 30.0f;
float OSGVruiToggleButtonGeometry::B = 50.0f;
float OSGVruiToggleButtonGeometry::D = 5.0f;

ref_ptr<Vec3Array> OSGVruiToggleButtonGeometry::coord = 0;
ref_ptr<Vec3Array> OSGVruiToggleButtonGeometry::normal = 0;
ref_ptr<Vec2Array> OSGVruiToggleButtonGeometry::texCoord = 0;

/// Toggle Button is supposed to be a Button with four
/// states (bitmap extensions also shown):
/// 1) off
/// 2) off & selected    '-selected'
/// 3) on                '-check'
/// 4) on & selected     '-check-selected'
OSGVruiToggleButtonGeometry::OSGVruiToggleButtonGeometry(coToggleButtonGeometry *button)
    : vruiButtonProvider(button)
    , normalNode(0)
    , pressedNode(0)
    , highlightNode(0)
    , pressedHighlightNode(0)
    , disabledNode(0)
    , myDCS(0)
    , texture(0)
{
    this->button = button;
}

OSGVruiToggleButtonGeometry::~OSGVruiToggleButtonGeometry()
{
}

void OSGVruiToggleButtonGeometry::createSharedLists()
{
    // global, static parameters for all Objects!
    // Only set up once in a lifetime! Check existence over coord

    if (coord == 0)
    {

        coord = new Vec3Array(4);
        normal = new Vec3Array(1);
        texCoord = new Vec2Array(4);

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
    }
}

void OSGVruiToggleButtonGeometry::createGeometry()
{

    if (normalNode == 0)
    {

        string textureName = button->getTextureName();

        // set up names
        string selectedName = textureName + "-selected";
        string checkName = textureName + "-check";
        string checkSelectedName = textureName + "-check-selected";
        string disabledName = textureName + "-disabled";

        // create normal texture
        normalNode = createNode(textureName, false).get();

        // create highlighted (selected) texture
        highlightNode = createNode(selectedName, false).get();

        // create pressed (check), normal texture
        pressedNode = createNode(checkName, true).get();

        // create pressed (check), highlighted (selected) texture
        pressedHighlightNode = createNode(checkSelectedName, true).get();

        disabledNode = createNode(disabledName, false).get();

        ref_ptr<MatrixTransform> transformNode = new MatrixTransform();
        switchNode = new Switch();

        switchNode->addChild(normalNode.get());
        switchNode->addChild(pressedNode.get());
        switchNode->addChild(highlightNode.get());
        switchNode->addChild(pressedHighlightNode.get());
        switchNode->addChild(disabledNode.get());

        transformNode->addChild(switchNode.get());

        myDCS = new OSGVruiTransformNode(transformNode.get());
    }
}

ref_ptr<Geode> OSGVruiToggleButtonGeometry::createNode(const string &textureName, bool checkTexture)
{

    createSharedLists();

    ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Geode> geometryNode = new Geode();
    ref_ptr<StateSet> stateSet = new StateSet();

    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    if (checkTexture)
    {
        stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE_NL), StateAttribute::ON | StateAttribute::PROTECTED);
    }
    else
    {
        stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);
    }

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
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

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;
}

// Kept for compatibility only!
ref_ptr<Geode> OSGVruiToggleButtonGeometry::createBox(const string &textureName)
{
    return createNode(textureName, false);
}

ref_ptr<Geode> OSGVruiToggleButtonGeometry::createCheck(const string &textureName)
{
    return createNode(textureName, true);
}

void OSGVruiToggleButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *OSGVruiToggleButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void OSGVruiToggleButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
