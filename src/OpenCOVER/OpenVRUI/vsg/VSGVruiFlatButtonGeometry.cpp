/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiFlatButtonGeometry.h>

#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/coFlatButtonGeometry.h>

#include <vsg/all.h>
#include <vsgXchange/all.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{

#define STYLE_IN 1
#define STYLE_OUT 2

float VSGVruiFlatButtonGeometry::A = 30.0f;
float VSGVruiFlatButtonGeometry::B = 50.0f;
float VSGVruiFlatButtonGeometry::D = 5.0f;


/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
VSGVruiFlatButtonGeometry::VSGVruiFlatButtonGeometry(coFlatButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(nullptr)
{

    this->button = button;
}

/// Destructor.
VSGVruiFlatButtonGeometry::~VSGVruiFlatButtonGeometry()
{
    delete myDCS;
    myDCS = nullptr;
}

vsg::ref_ptr<vsg::Node> VSGVruiFlatButtonGeometry::createQuad(const vsg::vec3& origin, const vsg::vec3& horizontal, const vsg::vec3& vertical)
{

    auto builder = vsg::Builder::create();
    //builder->options = options;

    vsg::GeometryInfo geomInfo;
    geomInfo.position = origin;
    geomInfo.dx=horizontal;
    geomInfo.dy=vertical;
    geomInfo.dz.set(0.0f, 0.0f, 1.0f);

    vsg::StateInfo stateInfo;

    return builder->createQuad(geomInfo, stateInfo);
}

void VSGVruiFlatButtonGeometry::createSharedLists()
{
  /*  if (coord1 == 0)
    {

        coord1 = new vec3Array(4);
        coord2 = new vec3Array(4);
        normal = new vec3Array(1);
        texCoord = new vec2Array(4);

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
    }*/
}

ref_ptr<vsg::Node> VSGVruiFlatButtonGeometry::createBox(const string &textureName)
{

    createSharedLists();

   /* ref_ptr<StateSet> stateSet = new StateSet();
    ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Texture2D> texture = 0;

    geometry->setVertexArray(coord1.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE), StateAttribute::ON | StateAttribute::PROTECTED);

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
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
        //VRUILOG("VSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
        if (defaulTexture != 0)
            texture = defaulTexture;
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

    ref_ptr<Geode> geometryNode = new Geode();

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;*/
    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    return node;

}

ref_ptr<Node> VSGVruiFlatButtonGeometry::createCheck(const string &textureName)
{

    createSharedLists();

    /*ref_ptr<StateSet> stateSet = new StateSet();
    ref_ptr<Geometry> geometry = new Geometry();
    ref_ptr<Texture2D> texture = 0;

    geometry->setVertexArray(coord2.get()); // use coord2 if you want the checked symbol to be larger
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);
    geometry->setTexCoordArray(0, texCoord.get());

    stateSet->setAttributeAndModes(VSGVruiPresets::getMaterial(coUIElement::WHITE_NL), StateAttribute::ON | StateAttribute::PROTECTED);

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
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
        //VRUILOG("VSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
        if (defaulTexture != 0)
            texture = defaulTexture;
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

    ref_ptr<Geode> geometryNode = new Geode();

    geometryNode->setStateSet(stateSet.get());
    geometryNode->addDrawable(geometry.get());

    return geometryNode;*/
    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    return node;
}

void VSGVruiFlatButtonGeometry::createGeometry()
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
    ref_ptr<Node> checkMark = createCheck(checkName);
    ref_ptr<Node> normalGeo = createBox(button->getTextureName());
    ref_ptr<Node> highlightGeo = createBox(highlightedName);
    ref_ptr<Node> disabledGeo = createBox(disabledName);


    // combine geometries pressed + normal
    ref_ptr<Group> pressedNormalGroup = Group::create();
    pressedNormalGroup->addChild(normalGeo);
    pressedNormalGroup->addChild(checkMark);

    // combine geometries pressed + highlighted
    ref_ptr<Group> pressedHighlightGroup = Group::create();
    pressedHighlightGroup->addChild(highlightGeo);
    pressedHighlightGroup->addChild(checkMark);

    // assign to 'base class' pointers
    normalNode = normalGeo;
    pressedNode = pressedNormalGroup;
    highlightNode = highlightGeo;
    pressedHighlightNode = pressedHighlightGroup;
    disabledNode = disabledGeo;

    ref_ptr<MatrixTransform> transformNode = MatrixTransform::create();
    switchNode = Switch::create();

    switchNode->addChild(true,normalNode);
    switchNode->addChild(false,pressedNode);
    switchNode->addChild(false, highlightNode);
    switchNode->addChild(false, pressedHighlightNode);
    switchNode->addChild(false, disabledNode);

    transformNode->addChild(switchNode);

    myDCS = new VSGVruiTransformNode(transformNode);
}

void VSGVruiFlatButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiFlatButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void VSGVruiFlatButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
