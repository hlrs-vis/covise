/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiFrame.h>

#include <OpenVRUI/coFrame.h>

#include <OpenVRUI/opensg/SGVruiPresets.h>
#include <OpenVRUI/opensg/SGVruiTransformNode.h>
#include <OpenVRUI/opensg/SGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>

#include <OpenSG/OSGImage.h>
#include <OpenSG/OSGMaterialChunk.h>
#include <OpenSG/OSGTextureChunk.h>

#include <qstring.h>

OSG_USING_NAMESPACE

GeoColors4fPtr SGVruiFrame::color = NullFC;
GeoNormals3fPtr SGVruiFrame::normal = NullFC;
GeoIndicesUI32Ptr SGVruiFrame::indices = NullFC;
GeoTexCoords2fPtr SGVruiFrame::texCoord = NullFC;

ChunkMaterialPtr SGVruiFrame::textureMaterial = ChunkMaterial::create();

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
SGVruiFrame::SGVruiFrame(coFrame *frame, const std::string &name)
    : SGVruiUIContainer(frame)
{

    this->frame = frame;

    NodePtr transformNode = makeCoredNode<ComponentTransform>();
    myDCS = new SGVruiTransformNode(transformNode);

    QString nodeName = QString("SGVruiFrame(%1)").arg(name);

    myDCS->setName(nodeName.latin1());

    createGeometry();

    beginEditCP(transformNode, Node::ChildrenFieldMask);
    transformNode->addChild(geometryNode);
    endEditCP(transformNode, Node::ChildrenFieldMask);
}

/** Destructor
 */
SGVruiFrame::~SGVruiFrame()
{
}

/// recalculate and set new geometry coordinates
void SGVruiFrame::resizeGeometry()
{

    float fw = frame->getWidth();
    float fh = frame->getHeight();

    float bw = frame->getBorderWidth();
    float bh = frame->getBorderHeight();

    if (fw < bw)
    {
        //VRUILOG("SGVruiFrame::resizeGeometry err: illegal frame width: " << fw);
        fw = bw * 3.0f;
    }

    if (fh < bh)
    {
        //VRUILOG("SGVruiFrame::resizeGeometry err: illegal frame height: " << fh);
        fh = bh * 3.0f;
    }

    float iw = fw - 2 * bw;
    float ih = fh - 2 * bh;

    VRUILOG("SGVruiFrame::resizeGeometry info: inner frame width : " << iw << ", height: " << ih);

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->setValue(Pnt3f(0, fh, 0), 0);
    coord->setValue(Pnt3f(20, fh, 0), 1);
    coord->setValue(Pnt3f(fw - 20, fh, 0), 2);
    coord->setValue(Pnt3f(fw, fh, 0), 3);
    coord->setValue(Pnt3f(fw, fh - 20, 0), 4);
    coord->setValue(Pnt3f(fw, 20, 0), 5);
    coord->setValue(Pnt3f(fw, 0, 0), 6);
    coord->setValue(Pnt3f(fw - 20, 0, 0), 7);
    coord->setValue(Pnt3f(20, 0, 0), 8);
    coord->setValue(Pnt3f(0, 0, 0), 9);
    coord->setValue(Pnt3f(0, 20, 0), 10);
    coord->setValue(Pnt3f(0, fh - 20, 0), 11);
    coord->setValue(Pnt3f(bw, ih + bh, 0), 12);
    coord->setValue(Pnt3f(20, ih + bh, 0), 13);
    coord->setValue(Pnt3f(fw - 20, ih + bh, 0), 14);
    coord->setValue(Pnt3f(iw + bw, ih + bh, 0), 15);
    coord->setValue(Pnt3f(iw + bw, fh - 20, 0), 16);
    coord->setValue(Pnt3f(iw + bw, 20, 0), 17);
    coord->setValue(Pnt3f(iw + bw, bh, 0), 18);
    coord->setValue(Pnt3f(fw - 20, bh, 0), 19);
    coord->setValue(Pnt3f(20, bh, 0), 20);
    coord->setValue(Pnt3f(bw, bh, 0), 21);
    coord->setValue(Pnt3f(bw, 20, 0), 22);
    coord->setValue(Pnt3f(bw, fh - 20, 0), 23);
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);

    beginEditCP(geometry, Geometry::DlistCacheFieldId);
    geometry->invalidateDlistCache();
    endEditCP(geometry, Geometry::DlistCacheFieldId);
}

/// allocate shared datastructures that can be used by all frames
void SGVruiFrame::createSharedLists()
{

    if (color == NullFC)
    {

        VRUILOG("SGVruiFrame::createSharedLists info: creating shared data");

        color = GeoColors4f::create();
        normal = GeoNormals3f::create();
        indices = GeoIndicesUI32::create();
        texCoord = GeoTexCoords2f::create();

        ushort vertices[12 * 4];

        beginEditCP(texCoord, GeoTexCoords2f::GeoPropDataFieldMask);
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 32.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 32.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 0.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 0.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 24.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 24.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 8.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 8.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f));
        texCoord->addValue(Vec2f((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f));
        endEditCP(texCoord, GeoTexCoords2f::GeoPropDataFieldMask);

        beginEditCP(color, GeoColors3f::GeoPropDataFieldMask);
        color->getFieldPtr()->push_back(Color4f(0.8f, 0.8f, 0.8f, 1.0f));
        endEditCP(color, GeoColors3f::GeoPropDataFieldMask);

        beginEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);
        normal->addValue(Vec3f(0.0f, 0.0f, 1.0f));
        endEditCP(normal, GeoNormals3f::GeoPropDataFieldMask);

        vertices[0] = 12;
        vertices[1] = 13;
        vertices[2] = 1;
        vertices[3] = 0;
        vertices[4] = 13;
        vertices[5] = 14;
        vertices[6] = 2;
        vertices[7] = 1;
        vertices[8] = 14;
        vertices[9] = 15;
        vertices[10] = 3;
        vertices[11] = 2;
        vertices[12] = 15;
        vertices[13] = 16;
        vertices[14] = 4;
        vertices[15] = 3;
        vertices[16] = 16;
        vertices[17] = 17;
        vertices[18] = 5;
        vertices[19] = 4;
        vertices[20] = 17;
        vertices[21] = 18;
        vertices[22] = 6;
        vertices[23] = 5;
        vertices[24] = 18;
        vertices[25] = 19;
        vertices[26] = 7;
        vertices[27] = 6;
        vertices[28] = 19;
        vertices[29] = 20;
        vertices[30] = 8;
        vertices[31] = 7;
        vertices[32] = 20;
        vertices[33] = 21;
        vertices[34] = 9;
        vertices[35] = 8;
        vertices[36] = 21;
        vertices[37] = 22;
        vertices[38] = 10;
        vertices[39] = 9;
        vertices[40] = 22;
        vertices[41] = 23;
        vertices[42] = 11;
        vertices[43] = 10;
        vertices[44] = 23;
        vertices[45] = 12;
        vertices[46] = 0;
        vertices[47] = 11;

        beginEditCP(indices, GeoIndicesUI32::GeoPropDataFieldMask);
        for (int ctr = 0; ctr < 48; ++ctr)
        {
            indices->push_back(vertices[ctr]); // Vertex / TextureCoords
            indices->push_back(0); // Color / Normal
        }
        endEditCP(indices, GeoIndicesUI32::GeoPropDataFieldMask);

        MaterialChunkPtr materialChunk = MaterialChunk::create();

        beginEditCP(materialChunk);
        materialChunk->setColorMaterial(GL_AMBIENT_AND_DIFFUSE);
        materialChunk->setAmbient(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
        materialChunk->setDiffuse(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
        materialChunk->setSpecular(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
        materialChunk->setEmission(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
        materialChunk->setShininess(16.0f);
        materialChunk->setLit(true);
        endEditCP(materialChunk);

        TextureChunkPtr textureChunk = TextureChunk::create();

        SGVruiTexture *sgTex = dynamic_cast<SGVruiTexture *>(vruiRendererInterface::the()->createTexture(frame->getTextureName()));
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
            //VRUILOG(QString("SGVruiFrame::createGeometry warn: texture image %1 not found").arg(textureFileName).latin1());
        }

        beginEditCP(textureMaterial, ChunkMaterial::ChunksFieldMask);
        textureMaterial->addChunk(materialChunk);
        //textureMaterial->addChunk(textureChunk);
        textureMaterial->addChunk(SGVruiPresets::getPolyChunkFillCulled());
        endEditCP(textureMaterial, ChunkMaterial::ChunksFieldMask);
    }
}

/// create the  geometry node
void SGVruiFrame::createGeometry()
{

    createSharedLists();

    coord = GeoPositions3f::create();

    beginEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);
    coord->addValue(Pnt3f(0, 60, 0));
    coord->addValue(Pnt3f(20, 60, 0));
    coord->addValue(Pnt3f(40, 60, 0));
    coord->addValue(Pnt3f(60, 60, 0));
    coord->addValue(Pnt3f(60, 40, 0));
    coord->addValue(Pnt3f(60, 20, 0));
    coord->addValue(Pnt3f(60, 0, 0));
    coord->addValue(Pnt3f(40, 0, 0));
    coord->addValue(Pnt3f(20, 0, 0));
    coord->addValue(Pnt3f(0, 0, 0));
    coord->addValue(Pnt3f(0, 20, 0));
    coord->addValue(Pnt3f(0, 40, 0));
    coord->addValue(Pnt3f(5, 55, 0));
    coord->addValue(Pnt3f(20, 55, 0));
    coord->addValue(Pnt3f(40, 55, 0));
    coord->addValue(Pnt3f(55, 55, 0));
    coord->addValue(Pnt3f(55, 40, 0));
    coord->addValue(Pnt3f(55, 20, 0));
    coord->addValue(Pnt3f(55, 5, 0));
    coord->addValue(Pnt3f(40, 5, 0));
    coord->addValue(Pnt3f(20, 5, 0));
    coord->addValue(Pnt3f(5, 5, 0));
    coord->addValue(Pnt3f(5, 20, 0));
    coord->addValue(Pnt3f(5, 40, 0));
    endEditCP(coord, GeoPositions3f::GeoPropDataFieldMask);

    geometryNode = makeCoredNode<Geometry>();
    geometry = GeometryPtr::dcast(geometryNode->getCore());

    GeoPLengthsPtr length = GeoPLengthsUI32::create();
    beginEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);
    length->addValue(48);
    endEditCP(length, GeoPLengthsUI32::GeoPropDataFieldMask);

    GeoPTypesPtr types = GeoPTypesUI8::create();
    beginEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);
    types->addValue(GL_QUADS);
    endEditCP(types, GeoPTypesUI8::GeoPropDataFieldMask);

    beginEditCP(geometry);
    geometry->setTypes(types);
    geometry->setLengths(length);
    geometry->setIndices(indices);
    geometry->getIndexMapping().push_back(Geometry::MapPosition | Geometry::MapTexCoords);
    geometry->getIndexMapping().push_back(Geometry::MapColor | Geometry::MapNormal);
    geometry->setPositions(coord);
    geometry->setColors(color);
    geometry->setNormals(normal);
    geometry->setTexCoords(texCoord);
    geometry->setMaterial(textureMaterial);
    endEditCP(geometry);

    resizeGeometry();
}
