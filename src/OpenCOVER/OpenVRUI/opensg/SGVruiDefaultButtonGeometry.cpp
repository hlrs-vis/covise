/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiDefaultButtonGeometry.h>

#include <OpenVRUI/opensg/SGVruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <OpenSG/OSGBlendChunk.h>
#include <OpenSG/OSGChunkMaterial.h>
#include <OpenSG/OSGComponentTransform.h>
#include <OpenSG/OSGFontStyle.h>
#include <OpenSG/OSGFontStyleFactory.h>
#include <OpenSG/OSGMaterialChunk.h>
#include <OpenSG/OSGSimpleAttachments.h>
#include <OpenSG/OSGSimpleGeometry.h>
#include <OpenSG/OSGTextureChunk.h>

#include <vector>
#include <string>

using namespace std;

#define STYLE_IN 1
#define STYLE_OUT 2
#define DETAIL_LEVEL 40 // number of triangles

OSG_USING_NAMESPACE

SGVruiDefaultButtonGeometry::SGVruiDefaultButtonGeometry(coButtonGeometry *geometry)
    : vruiButtonProvider(geometry)
    , normalNode(NullFC)
    , pressedNode(NullFC)
    , highlightNode(NullFC)
    , pressedHighlightNode(NullFC)
    , myDCS(0)
{

    textString = geometry->getTextureName().c_str();
}

SGVruiDefaultButtonGeometry::~SGVruiDefaultButtonGeometry()
{
    delete myDCS;
}

void SGVruiDefaultButtonGeometry::createGeometry()
{

    if (normalNode == NullFC)
    {

        VRUILOG("SGVruiDefaultButtonGeometry::initText fixme: hard-coded font");

        PathHandler paths;
        paths.push_backPath("/mnt/pro/cod/extern_libs/src/OpenSG-cvs/OpenSG/Source/Experimental/Text/");
        paths.push_backPath("/usr/X11R6/lib/X11/fonts/TTF/");

        FontStyle *fontStyle = FontStyleFactory::the().create(paths, "test.txf", 8.0f);
        //FontStyle * fontStyle = FontStyleFactory::the().create(paths, "Vera.ttf", 8.0f);

        if (!fontStyle)
        {
            VRUILOG("SGVruiDefaultButtonGeometry::createGeometry err: cannot create font");
            return;
        }

        text.setFontStyle(fontStyle);
        text.setJustifyMajor(MIDDLE_JT);
        text.setSize(8.0f);

        normalNode = createNode(false, false);
        pressedNode = createNode(true, false);
        highlightNode = createNode(false, true);
        pressedHighlightNode = createNode(true, true);

        NodePtr transformNode = makeCoredNode<ComponentTransform>();
        switchNode = makeCoredNode<Switch>();

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

void SGVruiDefaultButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{

    createGeometry();

    SwitchPtr switchCore = SwitchPtr::dcast(switchNode->getCore());

    beginEditCP(switchCore, Switch::ChoiceFieldMask);
    switchCore->setChoice(active);
    endEditCP(switchCore, Switch::ChoiceFieldMask);
}

NodePtr SGVruiDefaultButtonGeometry::createNode(bool pressed, bool highlighted)
{

    vector<string> line;

    line.push_back(textString);

    // Create button geometry:
    GeometryPtr geometry = makeCylinderGeo(1, 1, DETAIL_LEVEL, true, true, true);
    NodePtr geometryNode = Node::create();

    beginEditCP(geometryNode, Node::CoreFieldMask);
    geometryNode->setCore(geometry);
    endEditCP(geometryNode, Node::CoreFieldMask);

    ChunkMaterialPtr material = ChunkMaterial::create();
    MaterialChunkPtr materialChunk = MaterialChunk::create();

    beginEditCP(materialChunk);
    //material->setSide(PFMTL_BOTH);
    materialChunk->setColorMaterial(GL_AMBIENT_AND_DIFFUSE);
    if (highlighted)
    {
        materialChunk->setAmbient(Color4f(0.2f, 0.2f, 0.0f, 1.0f));
        materialChunk->setDiffuse(Color4f(0.9f, 0.9f, 0.0f, 1.0f));
        materialChunk->setSpecular(Color4f(0.9f, 0.9f, 0.0f, 1.0f));
    }
    else
    {
        materialChunk->setAmbient(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
        materialChunk->setDiffuse(Color4f(0.9f, 0.9f, 0.9f, 1.0f));
        materialChunk->setSpecular(Color4f(0.9f, 0.9f, 0.9f, 1.0f));
    }
    materialChunk->setEmission(Color4f(0.0f, 0.0f, 0.0f, 1.0f));
    materialChunk->setShininess(16.0f);
    materialChunk->setLit(true);
    endEditCP(materialChunk);

    beginEditCP(material, ChunkMaterial::ChunksFieldMask);
    material->addChunk(materialChunk);
    endEditCP(material, ChunkMaterial::ChunksFieldMask);

    beginEditCP(geometry, Geometry::MaterialFieldMask);
    geometry->setMaterial(material);
    endEditCP(geometry, Geometry::MaterialFieldMask);

    osg::setName(geometryNode, textString.latin1());

    ComponentTransformPtr transform = ComponentTransform::create();
    NodePtr transformNode = Node::create();

    beginEditCP(transformNode, Node::CoreFieldMask);
    transformNode->setCore(transform);
    endEditCP(transformNode, Node::CoreFieldMask);

    QString name = QString("SGVruiDefaultButtonGeometry%1%2(%3)")
                       .arg(pressed ? "-pressed" : "")
                       .arg(highlighted ? "-highlighted" : "")
                       .arg(textString);

    osg::setName(transformNode, name.latin1());

    ComponentTransformPtr buttonTransform = ComponentTransform::create();
    NodePtr buttonTransformNode = Node::create();

    beginEditCP(buttonTransform, ComponentTransform::ScaleFieldMask | ComponentTransform::TranslationFieldMask);
    if (pressed)
    {
        buttonTransform->setTranslation(Vec3f(0.0f, 0.0f, 2.0f));
        buttonTransform->setScale(Vec3f(10.0f, 5.0f, 2.0f));
    }
    else
    {
        buttonTransform->setTranslation(Vec3f(0.0f, 0.0f, 5.0f));
        buttonTransform->setScale(Vec3f(10.0f, 5.0f, 5.0f));
    }
    endEditCP(buttonTransform, ComponentTransform::ScaleFieldMask | ComponentTransform::TranslationFieldMask);

    beginEditCP(buttonTransformNode, Node::ChildrenFieldMask | Node::CoreFieldMask);
    buttonTransformNode->setCore(buttonTransform);
    buttonTransformNode->addChild(geometryNode);
    endEditCP(buttonTransformNode, Node::ChildrenFieldMask | Node::CoreFieldMask);

    ComponentTransformPtr textTransform = ComponentTransform::create();
    NodePtr textTransformNode = Node::create();

    beginEditCP(textTransform,
                ComponentTransform::RotationFieldMask | ComponentTransform::TranslationFieldMask | ComponentTransform::ScaleFieldMask);

    Quaternion q;
    q.setValueAsAxisDeg(Vec3f(1, 0, 0), 270);
    textTransform->setTranslation(Vec3f(0.0f, 3.0f, 7.5f));
    textTransform->setRotation(q);
    textTransform->setScale(Vec3f(8.0f, 8.0f, 1.0f));

    endEditCP(textTransform,
              ComponentTransform::RotationFieldMask | ComponentTransform::TranslationFieldMask | ComponentTransform::ScaleFieldMask);

    //GeometryPtr textGeometry = osg::makePlaneGeo(1.0f, 1.0f, 1, 1);
    GeometryPtr textGeometry = Geometry::create();
    ImagePtr textTexture = Image::create();
    NodePtr textNode = Node::create();

    if (text.fillTXFGeo(*textGeometry, true, line))
    {
        text.fillTXFImage(textTexture);
    }
    else
    {
        VRUILOG("SGVruiDefaultButtonGeometry::createNode err: cannot create text node");
    }

    beginEditCP(textNode, Node::CoreFieldMask);
    textNode->setCore(textGeometry);
    endEditCP(textNode, Node::CoreFieldMask);

    beginEditCP(textTransformNode, Node::ChildrenFieldMask | Node::CoreFieldMask);
    textTransformNode->setCore(textTransform);
    textTransformNode->addChild(textNode);
    endEditCP(textTransformNode, Node::ChildrenFieldMask | Node::CoreFieldMask);

    ChunkMaterialPtr textMaterial = ChunkMaterial::create();
    MaterialChunkPtr textMaterialChunk = MaterialChunk::create();
    TextureChunkPtr textTextureChunk = TextureChunk::create();
    BlendChunkPtr textBlendChunk = BlendChunk::create();

    //textMat->setSide(PFMTL_BOTH);
    beginEditCP(textMaterialChunk);
    textMaterialChunk->setColorMaterial(GL_AMBIENT_AND_DIFFUSE);
    textMaterialChunk->setAmbient(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
    textMaterialChunk->setDiffuse(Color4f(0.9f, 0.9f, 0.9f, 1.0f));
    textMaterialChunk->setSpecular(Color4f(0.9f, 0.9f, 0.9f, 1.0f));
    textMaterialChunk->setEmission(Color4f(0.0f, 0.0f, 0.0f, 1.0f));
    textMaterialChunk->setShininess(16.0f);
    textMaterialChunk->setLit(true);
    endEditCP(textMaterialChunk);

    beginEditCP(textTextureChunk, TextureChunk::ImageFieldMask);
    textTextureChunk->setImage(textTexture);
    endEditCP(textTextureChunk, TextureChunk::ImageFieldMask);

    beginEditCP(textBlendChunk, BlendChunk::AlphaFuncFieldMask | BlendChunk::AlphaValueFieldMask);
    textBlendChunk->setAlphaFunc(GL_NOTEQUAL);
    textBlendChunk->setAlphaValue(0);
    endEditCP(textBlendChunk, BlendChunk::AlphaFuncFieldMask | BlendChunk::AlphaValueFieldMask);

    beginEditCP(textMaterial, ChunkMaterial::ChunksFieldMask);
    textMaterial->addChunk(textMaterialChunk);
    textMaterial->addChunk(textTextureChunk);
    textMaterial->addChunk(textBlendChunk);
    endEditCP(textMaterial, ChunkMaterial::ChunksFieldMask);

    beginEditCP(textGeometry, Geometry::MaterialFieldMask);
    textGeometry->setMaterial(textMaterial);
    endEditCP(textGeometry, Geometry::MaterialFieldMask);

    beginEditCP(transformNode, Node::ChildrenFieldMask);
    transformNode->addChild(buttonTransformNode);
    transformNode->addChild(textTransformNode);
    endEditCP(transformNode, Node::ChildrenFieldMask);

    return transformNode;
}

void SGVruiDefaultButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *SGVruiDefaultButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

float SGVruiDefaultButtonGeometry::getWidth() const
{
    return 10.0f;
}

float SGVruiDefaultButtonGeometry::getHeight() const
{
    return 5.0f;
}
