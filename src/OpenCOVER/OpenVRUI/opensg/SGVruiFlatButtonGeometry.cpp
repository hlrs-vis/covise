/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiFlatButtonGeometry.h>

#include <OpenVRUI/opensg/SGVruiPresets.h>
#include <OpenVRUI/opensg/SGVruiTransformNode.h>
#include <OpenVRUI/opensg/SGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/coFlatButtonGeometry.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGMaterialChunk.h>
#include <OpenSG/OSGTextureChunk.h>

#include <OpenVRUI/util/vruiLog.h>

#include <OpenSG/OSGSimpleAttachments.h>

OSG_USING_NAMESPACE
using namespace std;

#define STYLE_IN 1
#define STYLE_OUT 2

float SGVruiFlatButtonGeometry::A = 30.0f;
float SGVruiFlatButtonGeometry::B = 50.0f;
float SGVruiFlatButtonGeometry::D = 5.0f;

GeoPositions3fPtr SGVruiFlatButtonGeometry::coord1 = NullFC;
GeoPositions3fPtr SGVruiFlatButtonGeometry::coord2 = NullFC;
GeoNormals3fPtr SGVruiFlatButtonGeometry::normal = NullFC;
GeoTexCoords2fPtr SGVruiFlatButtonGeometry::texCoord = NullFC;

/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
SGVruiFlatButtonGeometry::SGVruiFlatButtonGeometry(coFlatButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(0)
{

    this->button = button;
}

/// Destructor.
SGVruiFlatButtonGeometry::~SGVruiFlatButtonGeometry()
{
    delete myDCS;
    myDCS = 0;
}

void SGVruiFlatButtonGeometry::createSharedLists()
{

    if (coord1 == NullFC)
    {

        coord1 = GeoPositions3f::create();
        coord2 = GeoPositions3f::create();
        normal = GeoNormals3f::create();
        texCoord = GeoTexCoords2f::create();

        beginEditCP(coord1);
        coord1->addValue(Pnt3f(0.0f, 0.0f, 0.0f));
        coord1->addValue(Pnt3f(A, 0.0f, 0.0f));
        coord1->addValue(Pnt3f(A, A, 0.0f));
        coord1->addValue(Pnt3f(0.0f, A, 0.0f));
        endEditCP(coord1);

        beginEditCP(coord2);
        coord2->addValue(Pnt3f(0 - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D));
        coord2->addValue(Pnt3f(B - ((B - A) / 2.0f), 0 - ((B - A) / 2.0f), D));
        coord2->addValue(Pnt3f(B - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D));
        coord2->addValue(Pnt3f(0 - ((B - A) / 2.0f), B - ((B - A) / 2.0f), D));
        endEditCP(coord2);

        beginEditCP(texCoord);
        texCoord->addValue(Vec2f(0.0f, 0.0f));
        texCoord->addValue(Vec2f(1.0f, 0.0f));
        texCoord->addValue(Vec2f(1.0f, 1.0f));
        texCoord->addValue(Vec2f(0.0f, 1.0f));
        endEditCP(texCoord);

        beginEditCP(normal);
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        endEditCP(normal);
    }
}

NodePtr SGVruiFlatButtonGeometry::createBox(const string &textureName)
{

    createSharedLists();

    ChunkMaterialPtr stateSet = ChunkMaterial::create();
    TextureChunkPtr textureChunk = TextureChunk::create();

    NodePtr geometryNode = makeCoredNode<Geometry>();
    GeometryPtr geometry = GeometryPtr::dcast(geometryNode->getCore());

    GeoPLengthsPtr length = GeoPLengthsUI32::create();
    beginEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);
    length->addValue(4);
    endEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);

    GeoPTypesPtr types = GeoPTypesUI8::create();
    beginEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);
    types->addValue(GL_QUADS);
    endEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);

    beginEditCP(geometry);
    geometry->setPositions(coord1);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setNormals(normal);
    geometry->setTexCoords(texCoord);
    geometry->setMaterial(stateSet);
    endEditCP(geometry);

    SGVruiTexture *sgTex = dynamic_cast<SGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    ImagePtr textureImage = sgTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(sgTex);

    beginEditCP(stateSet, ChunkMaterial::ChunksFieldMask);

    if (textureImage != NullFC)
    {
        beginEditCP(textureChunk);
        textureChunk->setImage(textureImage);
        textureChunk->setMinFilter(GL_LINEAR);
        textureChunk->setWrapS(GL_CLAMP);
        textureChunk->setWrapT(GL_CLAMP);
        textureChunk->setWrapR(GL_CLAMP);
        textureChunk->setEnvMode(GL_MODULATE);
        endEditCP(textureChunk);

        stateSet->addChunk(textureChunk);
    }
    else
    {
        //VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    stateSet->addChunk(SGVruiPresets::getMaterial(coUIElement::WHITE));
    stateSet->addChunk(SGVruiPresets::getPolyChunkFillCulled());
    endEditCP(stateSet, ChunkMaterial::ChunksFieldMask);

    return geometryNode;
}

NodePtr SGVruiFlatButtonGeometry::createCheck(const string &textureName)
{

    createSharedLists();

    ChunkMaterialPtr stateSet = ChunkMaterial::create();
    TextureChunkPtr textureChunk = TextureChunk::create();

    NodePtr geometryNode = makeCoredNode<Geometry>();
    GeometryPtr geometry = GeometryPtr::dcast(geometryNode->getCore());

    GeoPLengthsPtr length = GeoPLengthsUI32::create();
    beginEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);
    length->addValue(4);
    endEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);

    GeoPTypesPtr types = GeoPTypesUI8::create();
    beginEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);
    types->addValue(GL_QUADS);
    endEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);

    beginEditCP(geometry);
    geometry->setPositions(coord2);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setNormals(normal);
    geometry->setTexCoords(texCoord);
    geometry->setMaterial(stateSet);
    endEditCP(geometry);

    SGVruiTexture *sgTex = dynamic_cast<SGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    ImagePtr textureImage = sgTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(sgTex);

    if (textureImage != NullFC)
    {
        beginEditCP(textureChunk);
        textureChunk->setImage(textureImage);
        textureChunk->setMinFilter(GL_LINEAR);
        textureChunk->setWrapS(GL_CLAMP);
        textureChunk->setWrapT(GL_CLAMP);
        textureChunk->setWrapR(GL_CLAMP);
        textureChunk->setEnvMode(GL_MODULATE);
        endEditCP(textureChunk);
    }
    else
    {
        //VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    beginEditCP(stateSet, ChunkMaterial::ChunksFieldMask);
    stateSet->addChunk(SGVruiPresets::getMaterial(coUIElement::WHITE));
    stateSet->addChunk(textureChunk);
    stateSet->addChunk(SGVruiPresets::getPolyChunkFillCulled());
    endEditCP(stateSet, ChunkMaterial::ChunksFieldMask);

    return geometryNode;
}

void SGVruiFlatButtonGeometry::createGeometry()
{

    if (myDCS)
        return;

    //VRUILOG("SGVruiFlatButtonGeometry::createGeometry info: creating geometry")

    // name for highlighted geometry
    string highlightedName = string(button->getTextureName()) + "-highlighted";

    // name for checkmark geometry
    string checkName = string(button->getTextureName()) + "-check";

    // Build checkMark and base/highlighted box geometries
    NodePtr checkMark = createCheck(checkName);
    NodePtr normalGeo = createBox(button->getTextureName());
    NodePtr highlightGeo = createBox(highlightedName);

    // make output readable
    setName(checkMark, "CheckMark");
    setName(normalGeo, "normalGeo");
    setName(highlightGeo, "highlightGeo");

    // combine geometries pressed + normal
    GroupPtr pressedNormalGroupCore = Group::create();
    NodePtr pressedNormalGroup = makeNodeFor(pressedNormalGroupCore);

    beginEditCP(pressedNormalGroup, Node::ChildrenFieldMask);
    pressedNormalGroup->addChild(normalGeo);
    pressedNormalGroup->addChild(checkMark);
    endEditCP(pressedNormalGroup, Node::ChildrenFieldMask);

    // combine geometries pressed + highlighted
    GroupPtr pressedHighlightGroupCore = Group::create();
    NodePtr pressedHighlightGroup = makeNodeFor(pressedHighlightGroupCore);

    beginEditCP(pressedHighlightGroup, Node::ChildrenFieldMask);
    pressedHighlightGroup->addChild(highlightGeo);
    pressedHighlightGroup->addChild(checkMark);
    endEditCP(pressedHighlightGroup, Node::ChildrenFieldMask);

    // assign to 'base class' pointers
    normalNode = normalGeo;
    pressedNode = pressedNormalGroup;
    highlightNode = highlightGeo;
    pressedHighlightNode = pressedHighlightGroup;

    ComponentTransformPtr transform = ComponentTransform::create();
    NodePtr transformNode = makeNodeFor(transform);

    switchCore = Switch::create();
    NodePtr switchNode = makeNodeFor(switchCore);

    beginEditCP(switchNode, Node::ChildrenFieldMask);
    switchNode->addChild(normalNode);
    switchNode->addChild(pressedNode);
    switchNode->addChild(highlightNode);
    switchNode->addChild(pressedHighlightNode);
    endEditCP(switchNode, Node::ChildrenFieldMask);

    beginEditCP(transformNode, Node::ChildrenFieldMask);
    transformNode->addChild(switchNode);
    endEditCP(transformNode, Node::ChildrenFieldMask);

    myDCS = new SGVruiTransformNode(transformNode);
}

void SGVruiFlatButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *SGVruiFlatButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void SGVruiFlatButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    beginEditCP(switchCore, Switch::ChoiceFieldMask);
    switchCore->setChoice(active);
    endEditCP(switchCore, Switch::ChoiceFieldMask);
}
