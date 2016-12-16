/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiFrame.h>

#include <OpenVRUI/coFrame.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <osg/Material>
#include <osgDB/ReadFile>

#include <string>
#include <OpenVRUI/util/vruiLog.h>

using namespace osg;
using namespace std;

namespace vrui
{

ref_ptr<Vec4Array> OSGVruiFrame::color = 0;
ref_ptr<Vec3Array> OSGVruiFrame::normal = new Vec3Array(1);
ref_ptr<Vec2Array> OSGVruiFrame::texCoord = new Vec2Array(24);
ref_ptr<DrawElementsUShort> OSGVruiFrame::coordIndex = 0;

/** Constructor
 @param name Texture name, default is "UI/Frame",
 a white frame with round edges
*/
OSGVruiFrame::OSGVruiFrame(coFrame *frame, const string &name)
    : OSGVruiUIContainer(frame)
{

    this->frame = frame;

    stateSet = 0;
    geometry = 0;
    geometryNode = 0;
    texture = 0;

    ref_ptr<MatrixTransform> transform = new MatrixTransform();
    myDCS = new OSGVruiTransformNode(transform.get());

    string nodeName = "OSGVruiFrame(" + name + ")";

    myDCS->setName(nodeName);

    createGeometry();

    transform->addChild(geometryNode.get());
}

/** Destructor
 */
OSGVruiFrame::~OSGVruiFrame()
{
}

/// recalculate and set new geometry coordinates
void OSGVruiFrame::resizeGeometry()
{

    float fw = frame->getWidth();
    float fh = frame->getHeight();

    //VRUILOG("OSGVruiFrame::resizeGeometry info: resizing " << fw << "x" << fh)

    float bw = frame->getBorderWidth();
    float bh = frame->getBorderHeight();

    float iw = fw - 2 * bw;
    float ih = fh - 2 * bh;

    (*coord)[0].set(0, fh, 0);
    (*coord)[1].set(20, fh, 0);
    (*coord)[2].set(fw - 20, fh, 0);
    (*coord)[3].set(fw, fh, 0);
    (*coord)[4].set(fw, fh - 20, 0);
    (*coord)[5].set(fw, 20, 0);
    (*coord)[6].set(fw, 0, 0);
    (*coord)[7].set(fw - 20, 0, 0);
    (*coord)[8].set(20, 0, 0);
    (*coord)[9].set(0, 0, 0);
    (*coord)[10].set(0, 20, 0);
    (*coord)[11].set(0, fh - 20, 0);
    (*coord)[12].set(bw, ih + bh, 0);
    (*coord)[13].set(20, ih + bh, 0);
    (*coord)[14].set(fw - 20, ih + bh, 0);
    (*coord)[15].set(iw + bw, ih + bh, 0);
    (*coord)[16].set(iw + bw, fh - 20, 0);
    (*coord)[17].set(iw + bw, 20, 0);
    (*coord)[18].set(iw + bw, bh, 0);
    (*coord)[19].set(fw - 20, bh, 0);
    (*coord)[20].set(20, bh, 0);
    (*coord)[21].set(bw, bh, 0);
    (*coord)[22].set(bw, 20, 0);
    (*coord)[23].set(bw, fh - 20, 0);

	coord->dirty();
	geometry->setVertexArray(coord.get());
    geometryNode->dirtyBound();
	//geometry->dirtyBound(); this is done by setVertexArray
	//geometry->dirtyDisplayList();
}

/// allocate shared datastructures that can be used by all frames
void OSGVruiFrame::createSharedLists()
{

    if (color == 0)
    {

        //VRUILOG("OSGVruiFrame::createSharedLists info: creating shared data")

        color = new Vec4Array(1);

        (*texCoord)[0].set((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 32.0f);
        (*texCoord)[1].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f);
        (*texCoord)[2].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 32.0f);
        (*texCoord)[3].set((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 32.0f);
        (*texCoord)[4].set((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[5].set((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[6].set((1.0f / 32.0f) * 32.0f, (1.0f / 32.0f) * 0.0f);
        (*texCoord)[7].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f);
        (*texCoord)[8].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 0.0f);
        (*texCoord)[9].set((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 0.0f);
        (*texCoord)[10].set((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[11].set((1.0f / 32.0f) * 0.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[12].set((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 24.0f);
        (*texCoord)[13].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f);
        (*texCoord)[14].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 24.0f);
        (*texCoord)[15].set((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 24.0f);
        (*texCoord)[16].set((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[17].set((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[18].set((1.0f / 32.0f) * 24.0f, (1.0f / 32.0f) * 8.0f);
        (*texCoord)[19].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f);
        (*texCoord)[20].set((1.0f / 32.0f) * 16.0f, (1.0f / 32.0f) * 8.0f);
        (*texCoord)[21].set((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 8.0f);
        (*texCoord)[22].set((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f);
        (*texCoord)[23].set((1.0f / 32.0f) * 8.0f, (1.0f / 32.0f) * 16.0f);

        (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);
        (*normal)[0].set(0.0f, 0.0f, 1.0f);

        ushort *vertices = new ushort[12 * 4];

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

        coordIndex = new DrawElementsUShort(PrimitiveSet::QUADS, 48, vertices);

        delete[] vertices;
    }
}

/// greate the  geometry node
void OSGVruiFrame::createGeometry()
{

    if (geometry == 0)
    {
        createSharedLists();

        //VRUILOG("OSGVruiFrame::createGeometry info: creating geometry")

        coord = new Vec3Array(24);

        (*coord)[0] = Vec3(0, 60, 0);
        (*coord)[1] = Vec3(20, 60, 0);
        (*coord)[2] = Vec3(40, 60, 0);
        (*coord)[3] = Vec3(60, 60, 0);
        (*coord)[4] = Vec3(60, 40, 0);
        (*coord)[5] = Vec3(60, 20, 0);
        (*coord)[6] = Vec3(60, 0, 0);
        (*coord)[7] = Vec3(40, 0, 0);
        (*coord)[8] = Vec3(20, 0, 0);
        (*coord)[9] = Vec3(0, 0, 0);
        (*coord)[10] = Vec3(0, 20, 0);
        (*coord)[11] = Vec3(0, 40, 0);
        (*coord)[12] = Vec3(5, 55, 0);
        (*coord)[13] = Vec3(20, 55, 0);
        (*coord)[14] = Vec3(40, 55, 0);
        (*coord)[15] = Vec3(55, 55, 0);
        (*coord)[16] = Vec3(55, 40, 0);
        (*coord)[17] = Vec3(55, 20, 0);
        (*coord)[18] = Vec3(55, 5, 0);
        (*coord)[19] = Vec3(40, 5, 0);
        (*coord)[20] = Vec3(20, 5, 0);
        (*coord)[21] = Vec3(5, 5, 0);
        (*coord)[22] = Vec3(5, 20, 0);
        (*coord)[23] = Vec3(5, 40, 0);

        OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(frame->getTextureName()));
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
            VRUILOG("OSGVruiFrame::createGeometry err: texture image " << frame->getTextureName() << " not found");
        }

        ref_ptr<Material> material = new Material();

        material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
        material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        material->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        material->setShininess(Material::FRONT_AND_BACK, 80.0f);
        material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

        ref_ptr<TexEnv> texEnv = OSGVruiPresets::getTexEnvModulate();
        ref_ptr<CullFace> cullFace = OSGVruiPresets::getCullFaceBack();
        ref_ptr<PolygonMode> polyMode = OSGVruiPresets::getPolyModeFill();

        geometryNode = new Geode();
        geometry = new Geometry();

        geometry->setVertexArray(coord.get());
        geometry->addPrimitiveSet(coordIndex.get());
        geometry->setColorArray(color.get());
        geometry->setColorBinding(Geometry::BIND_OVERALL);
        geometry->setNormalArray(normal.get());
        geometry->setNormalBinding(Geometry::BIND_OVERALL);
        geometry->setTexCoordArray(0, texCoord.get());

        stateSet = geometryNode->getOrCreateStateSet();

        OSGVruiPresets::makeTransparent(stateSet);
        stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setAttributeAndModes(material.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);

        stateSet->setTextureAttributeAndModes(0, texEnv.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

        geometryNode->setStateSet(stateSet.get());

        geometryNode->addDrawable(geometry.get());

        resizeGeometry();
    }
}
}
