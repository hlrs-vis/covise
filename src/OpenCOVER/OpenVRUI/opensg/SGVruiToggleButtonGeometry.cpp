/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiToggleButtonGeometry.h>

#include <OpenVRUI/coToggleButtonGeometry.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenVRUI/opensg/SGVruiTransformNode.h>
#include <OpenVRUI/opensg/SGVruiPresets.h>
#include <OpenVRUI/opensg/SGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGMaterialChunk.h>

#define STYLE_IN 1
#define STYLE_OUT 2

using namespace std;

OSG_USING_NAMESPACE

float SGVruiToggleButtonGeometry::A = 30.0f;
float SGVruiToggleButtonGeometry::B = 50.0f;
float SGVruiToggleButtonGeometry::D = 5.0f;

GeoPositions3fPtr SGVruiToggleButtonGeometry::coord = NullFC;
GeoNormals3fPtr SGVruiToggleButtonGeometry::normal = NullFC;
GeoTexCoords2fPtr SGVruiToggleButtonGeometry::texCoord = NullFC;

/// Toggle Button is supposed to be a Button with four
/// states (bitmap extensions also shown):
/// 1) off
/// 2) off & selected    '-selected'
/// 3) on                '-check'
/// 4) on & selected     '-check-selected'
SGVruiToggleButtonGeometry::SGVruiToggleButtonGeometry(coToggleButtonGeometry *button)
    : vruiButtonProvider(button)
    , normalNode(NullFC)
    , pressedNode(NullFC)
    , highlightNode(NullFC)
    , pressedHighlightNode(NullFC)
    , myDCS(0)
    , textureChunk(NullFC)
{

    this->button = button;
}

SGVruiToggleButtonGeometry::~SGVruiToggleButtonGeometry()
{
}

void SGVruiToggleButtonGeometry::createSharedLists()
{
    // global, static parameters for all Objects!
    // Only set up once in a lifetime! Check existence over coord

    if (coord == NullFC)
    {

        coord = GeoPositions3f::create();
        normal = GeoNormals3f::create();
        texCoord = GeoTexCoords2f::create();

        beginEditCP(coord);
        coord->addValue(Pnt3f(0.0f, A, 0.0f));
        coord->addValue(Pnt3f(A, A, 0.0f));
        coord->addValue(Pnt3f(A, 0.0f, 0.0f));
        coord->addValue(Pnt3f(0.0f, 0.0f, 0.0f));
        endEditCP(coord);

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

void SGVruiToggleButtonGeometry::createGeometry()
{

    if (normalNode == NullFC)
    {

        string textureName = button->getTextureName();

        // set up names
        string selectedName = textureName + "-selected";
        string checkName = textureName + "-check";
        string checkSelectedName = textureName + "-check-selected";

        // create normal texture
        normalNode = createNode(textureName, false);

        // create highlighted (selected) texture
        highlightNode = createNode(selectedName, false);

        // create pressed (check), normal texture
        pressedNode = createNode(checkName, true);

        // create pressed (check), highlighted (selected) texture
        pressedHighlightNode = createNode(checkSelectedName, true);

        NodePtr transformNode = makeCoredNode<ComponentTransform>();

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
}

NodePtr SGVruiToggleButtonGeometry::createNode(const string &textureName, bool checkTexture)
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
    geometry->setPositions(coord);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setNormals(normal);
    geometry->setTexCoords(texCoord);
    geometry->setMaterial(stateSet);
    endEditCP(geometry);

    beginEditCP(stateSet, ChunkMaterial::ChunksFieldMask);

    if (checkTexture)
    {
        stateSet->addChunk(SGVruiPresets::getMaterial(coUIElement::WHITE_NL));
    }
    else
    {
        stateSet->addChunk(SGVruiPresets::getMaterial(coUIElement::WHITE));
    }

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

        stateSet->addChunk(textureChunk);
    }
    else
    {
        //VRUILOG("OSGVruiFlatButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    stateSet->addChunk(SGVruiPresets::getPolyChunkFillCulled());

    endEditCP(stateSet, ChunkMaterial::ChunksFieldMask);

    return geometryNode;
}

// Kept for compatibility only!
NodePtr SGVruiToggleButtonGeometry::createBox(const string &textureName)
{
    return createNode(textureName, false);
}

NodePtr SGVruiToggleButtonGeometry::createCheck(const string &textureName)
{
    return createNode(textureName, true);
}

void SGVruiToggleButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *SGVruiToggleButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void SGVruiToggleButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();
    beginEditCP(switchCore, Switch::ChoiceFieldMask);
    switchCore->setChoice(active);
    endEditCP(switchCore, Switch::ChoiceFieldMask);
}
