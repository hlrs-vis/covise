/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiFlatButtonGeometry.h>

#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/coFlatButtonGeometry.h>

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

#include <OpenVRUI/util/vruiLog.h>

using namespace osg;
using namespace std;

namespace vrui
{

#define STYLE_IN 1
#define STYLE_OUT 2

float OSGVruiFlatButtonGeometry::A = 30.0f;
float OSGVruiFlatButtonGeometry::B = 50.0f;
float OSGVruiFlatButtonGeometry::D = 5.0f;

ref_ptr<Vec3Array> OSGVruiFlatButtonGeometry::coord1 = 0;
ref_ptr<Vec3Array> OSGVruiFlatButtonGeometry::coord2 = 0;
ref_ptr<Vec3Array> OSGVruiFlatButtonGeometry::normal = 0;
ref_ptr<Vec2Array> OSGVruiFlatButtonGeometry::texCoord = 0;

/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
OSGVruiFlatButtonGeometry::OSGVruiFlatButtonGeometry(coFlatButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(0)
    , defaulTexture(0)
{

    this->button = button;
}

/// Destructor.
OSGVruiFlatButtonGeometry::~OSGVruiFlatButtonGeometry()
{
    delete myDCS;
    myDCS = 0;
    if (defaulTexture != 0)
        defaulTexture->unref();
}

void OSGVruiFlatButtonGeometry::createSharedLists()
{
    if (coord1 == 0)
    {

        coord1 = new Vec3Array(4);
        coord2 = new Vec3Array(4);
        normal = new Vec3Array(1);
        texCoord = new Vec2Array(4);

        (*coord1)[3].set(0.0f, A, 0.0f);
        (*coord1)[2].set(A, A, 0.0f);
        (*coord1)[1].set(A, 0.0f, 0.0f);
        (*coord1)[0].set(0.0f, 0.0f, 0.0f);

        (*coord2)[3].set(0 - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D);
        (*coord2)[2].set(B - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D);
        (*coord2)[1].set(B - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D);
        (*coord2)[0].set(0 - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D);

        (*texCoord)[0].set(0.0f, 0.0f);
        (*texCoord)[1].set(1.0f, 0.0f);
        (*texCoord)[2].set(1.0f, 1.0f);
        (*texCoord)[3].set(0.0f, 1.0f);

        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }
}

ref_ptr<Geode> OSGVruiFlatButtonGeometry::createBox(const string &textureName)
{

    createSharedLists();

    ref_ptr<StateSet> stateSet = new StateSet();
    ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Texture2D> texture = 0;

    geometry->setVertexArray(coord1.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    texture = oTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(oTex);

    if (texture.valid())
    {
        texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texture->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP);
        if (defaulTexture == 0)
        {
            defaulTexture = texture;
            defaulTexture->ref();
        }
    }
    else
    {
        //VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
        if (defaulTexture != 0)
            texture = defaulTexture;
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

    ref_ptr<Geode> geometryNode = new Geode();

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;
}

ref_ptr<Geode> OSGVruiFlatButtonGeometry::createCheck(const string &textureName)
{

    createSharedLists();

    ref_ptr<StateSet> stateSet = new StateSet();
    ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Texture2D> texture = 0;

    geometry->setVertexArray(coord2.get()); // use coord2 if you want the checked symbol to be larger
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    stateSet->setAttributeAndModes(OSGVruiPresets::getMaterial(coUIElement::WHITE_NL), StateAttribute::ON | StateAttribute::PROTECTED);

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    texture = oTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(oTex);

    if (texture.valid())
    {
        texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texture->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP);
        if (defaulTexture == 0)
        {
            defaulTexture = texture;
            defaulTexture->ref();
        }
    }
    else
    {
        //VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
        if (defaulTexture != 0)
            texture = defaulTexture;
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

    ref_ptr<Geode> geometryNode = new Geode();

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;
}

void OSGVruiFlatButtonGeometry::createGeometry()
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
    ref_ptr<Geode> checkMark = createCheck(checkName);
    ref_ptr<Geode> normalGeo = createBox(button->getTextureName());
    ref_ptr<Geode> highlightGeo = createBox(highlightedName);
    ref_ptr<Geode> disabledGeo = createBox(disabledName);

    // make output readable
    checkMark->setName("CheckMark");
    normalGeo->setName("normalGeo");
    highlightGeo->setName("highlightGeo");
    disabledGeo->setName("disabledGeo");

    // combine geometries pressed + normal
    ref_ptr<Group> pressedNormalGroup = new Group();
    pressedNormalGroup->addChild(normalGeo.get());
    pressedNormalGroup->addChild(checkMark.get());

    // combine geometries pressed + highlighted
    ref_ptr<Group> pressedHighlightGroup = new Group();
    pressedHighlightGroup->addChild(highlightGeo.get());
    pressedHighlightGroup->addChild(checkMark.get());

    // assign to 'base class' pointers
    normalNode = normalGeo.get();
    pressedNode = pressedNormalGroup.get();
    highlightNode = highlightGeo.get();
    pressedHighlightNode = pressedHighlightGroup.get();
    disabledNode = disabledGeo.get();

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

void OSGVruiFlatButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *OSGVruiFlatButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void OSGVruiFlatButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
