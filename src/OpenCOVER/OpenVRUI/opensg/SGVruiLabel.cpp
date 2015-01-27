/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiLabel.h>

#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coUIContainer.h>

#include <OpenVRUI/opensg/SGVruiTransformNode.h>
#include <OpenVRUI/opensg/SGVruiPresets.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenSG/OSGFontStyleFactory.h>
#include <OpenSG/OSGQuaternion.h>
#include <OpenSG/OSGTXFFont.h>
#include <OpenSG/OSGTXFFontStyle.h>
#include <OpenSG/OSGTTFontStyle.h>
#include <OpenSG/OSGSimpleGeometry.h>

#include <OpenSG/OSGPathHandler.h>

#include <OpenVRUI/util/vruiLog.h>

#include <string>

using namespace std;

OSG_USING_NAMESPACE

#define BORDERWIDTH 5.0

/// Constructor.
SGVruiLabel::SGVruiLabel(coLabel *label)
    : SGVruiUIElement(label)
    , textColor(Color4ub(255, 255, 255, 255))
    , textColorHL(Color4ub(0, 255, 255, 255))
    , textForeground(Color4ub(0, 255, 255, 255))
    , textBackground(Color4ub(0, 0, 0, 255))
{

    this->label = label;
}

/// Destructor.
SGVruiLabel::~SGVruiLabel()
{
}

float SGVruiLabel::getWidth() const
{

    float rv = 0.0f;

    if (image != NullFC)
        rv = image->getWidth();

    //VRUILOG("OSGVruiLabel::getWidth info: width is " << rv)
    return rv;
}

float SGVruiLabel::getHeight() const
{

    float rv = 0.0f;

    if (image != NullFC)
        rv = image->getHeight();

    //VRUILOG("OSGVruiLabel::getHeight info: height is " << rv)
    return rv;
}

float SGVruiLabel::getDepth() const
{

    float rv = 0.0f;
    return rv;
}

void SGVruiLabel::createGeometry()
{

    if (myDCS)
        return;

    NodePtr transformNode = makeCoredNode<ComponentTransform>();

    myDCS = new SGVruiTransformNode(transformNode);

    PathHandler path;
    path.push_backPath("/");
    FontStyle *fontStyle = FontStyleFactory::the().create(path, vruiRendererInterface::the()->getName("share/covise/fonts/test.txf").c_str(), 1);
    assert(fontStyle);
    labelText.setFontStyle(fontStyle);

    material = ChunkMaterial::create();
    textureChunk = TextureChunk::create();
    image = Image::create();

    beginEditCP(textureChunk);
    textureChunk->setImage(image);
    textureChunk->setMinFilter(GL_LINEAR);
    textureChunk->setWrapS(GL_CLAMP);
    textureChunk->setWrapT(GL_CLAMP);
    textureChunk->setWrapR(GL_CLAMP);
    textureChunk->setEnvMode(GL_MODULATE);
    endEditCP(textureChunk);

    GeometryPtr geometry = Geometry::create();
    NodePtr geoNode = makeNodeFor(geometry);

    coord = GeoPositions3f::create();
    GeoNormals3fPtr normal = GeoNormals3f::create();
    GeoTexCoords2fPtr texCoord = GeoTexCoords2f::create();

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->addValue(Pnt3f(0.0f, 0.0f, 0.0f));
    coord->addValue(Pnt3f(1.0f, 0.0f, 0.0f));
    coord->addValue(Pnt3f(1.0f, 1.0f, 0.0f));
    coord->addValue(Pnt3f(0.0f, 1.0f, 0.0f));
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);

    beginEditCP(texCoord, GeoTexCoords2f::GeoPropDataFieldMask);
    texCoord->addValue(Vec2f(0.0f, 0.0f));
    texCoord->addValue(Vec2f(1.0f, 0.0f));
    texCoord->addValue(Vec2f(1.0f, 1.0f));
    texCoord->addValue(Vec2f(0.0f, 1.0f));
    endEditCP(texCoord, GeoTexCoords2f::GeoPropDataFieldMask);

    beginEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);
    normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
    normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
    normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
    normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
    endEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);

    GeoPLengthsPtr length = GeoPLengthsUI32::create();
    beginEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);
    length->addValue(4);
    endEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);

    GeoPTypesPtr types = GeoPTypesUI8::create();
    beginEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);
    types->addValue(GL_QUADS);
    endEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);

    beginEditCP(material, ChunkMaterial::ChunksFieldMask);
    material->addChunk(textureChunk);
    material->addChunk(SGVruiPresets::getMaterial(coUIElement::WHITE));
    material->addChunk(SGVruiPresets::getPolyChunkFillCulled());
    endEditCP(material, ChunkMaterial::ChunksFieldMask);

    beginEditCP(geometry);
    geometry->setPositions(coord);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setNormals(normal);
    geometry->setTexCoords(texCoord);
    geometry->setMaterial(material);
    endEditCP(geometry);

    //   Quaternion rot;
    //   rot.makeRotate(-90.0f, 1.0f, 0.0f, 0.0f);
    //   labelText->setRotation(rot);

    makeText();

    beginEditCP(transformNode, Node::ChildrenFieldMask);
    transformNode->addChild(geoNode);
    endEditCP(transformNode, Node::ChildrenFieldMask);
}

/// Private method to generate OpenSG text string and attach it to a node.
void SGVruiLabel::makeText()
{

    if (label->getString() == 0 || strcmp(label->getString(), "") == 0)
    {
        VRUILOG("SGVruiLabel::makeText warn: skipped text creation")
        return;
    }

    VRUILOG("SGVruiLabel::makeText info: Making text '" << (label->getString() ? label->getString() : "*NULL*") << "'")

    switch (label->getJustify())
    {
    case coLabel::LEFT:
        labelText.setJustifyMajor(BEGIN_JT);
        break;
    case coLabel::CENTER:
        labelText.setJustifyMajor(MIDDLE_JT);
        break;
    case coLabel::RIGHT:
        labelText.setJustifyMajor(END_JT);
        break;
    }

    std::vector<std::string> line;
    line.push_back(label->getString());

    if (labelText.fillImage(image, line, &textForeground, &textBackground,
                            false, 0, 0,
                            FILL_TCM, CLEAR_ADD_MM,
                            24, false))
    {

        image->reformat(Image::OSG_RGBA_PF);
    }

    VRUILOG("SGVruiLabel::makeText info: image width = " << image->getWidth() << ", height = " << image->getHeight());

    textureChunk->imageContentChanged();

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->addValue(Pnt3f(0.0f, 0.0f, 0.0f));
    coord->addValue(Pnt3f(image->getWidth(), 0.0f, 0.0f));
    coord->addValue(Pnt3f(image->getWidth(), image->getHeight(), 0.0f));
    coord->addValue(Pnt3f(0.0f, image->getHeight(), 0.0f));
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);

    //    Text::Layout direction;
    //    switch (label->getDirection())
    //    {
    //       case coLabel::HORIZONTAL: direction = Text::LEFT_TO_RIGHT; break;
    //       case coLabel::VERTICAL:   direction = Text::VERTICAL; break;
    //    }
}

void SGVruiLabel::setHighlighted(bool hl)
{
    if (hl)
    {
        textForeground = textColorHL;
    }
    else
    {
        textBackground = textColor;
    }
    makeText();
}

void SGVruiLabel::resizeGeometry()
{
    createGeometry();
    makeText();
}

void SGVruiLabel::update()
{
    createGeometry();
    makeText();
}
