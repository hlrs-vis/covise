/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiPanelGeometry.h>

#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <osgDB/ReadFile>

#include <OpenVRUI/util/vruiLog.h>

#define STYLE_IN 1
#define STYLE_OUT 2

using namespace osg;
using std::string;

namespace vrui
{

float OSGVruiPanelGeometry::A = 0.2f;
float OSGVruiPanelGeometry::B = 0.4f;
float OSGVruiPanelGeometry::C = 12.0f;

ref_ptr<Vec3Array> OSGVruiPanelGeometry::coord = 0;
ref_ptr<Vec4Array> OSGVruiPanelGeometry::color = 0;
ref_ptr<Vec3Array> OSGVruiPanelGeometry::normal = 0;
ref_ptr<Vec2Array> OSGVruiPanelGeometry::texcoord = 0;

ref_ptr<DrawElementsUShort> OSGVruiPanelGeometry::coordIndex = 0;

ref_ptr<Material> OSGVruiPanelGeometry::textureMaterial = 0;

OSGVruiPanelGeometry::OSGVruiPanelGeometry(coPanelGeometry *geometry)
    : vruiPanelGeometryProvider(geometry)
{
    texture = 0;
}

OSGVruiPanelGeometry::~OSGVruiPanelGeometry()
{
}

void OSGVruiPanelGeometry::createSharedLists()
{
    if (color == 0)
    {

        color = new Vec4Array(1);
        coord = new Vec3Array(12);
        normal = new Vec3Array(36);
        texcoord = new Vec2Array(36);

        (*coord)[0].set(0.0f, 2.0f * (B) + C, 0.0f);
        (*coord)[1].set(2.0f * (B) + C, 2.0f * (B) + C, 0.0f);
        (*coord)[2].set(2.0f * (B) + C, 0.0f, 0.0f);
        (*coord)[3].set(0.0f, 0.0f, 0.0f);
        (*coord)[4].set(0.0f, 2.0f * (B) + C, A);
        (*coord)[5].set(B + C + B, 2.0f * (B) + C, A);
        (*coord)[6].set(B + C + B, 0.0f, A);
        (*coord)[7].set(0.0f, 0.0f, A);
        (*coord)[8].set(B, B + C, A + B);
        (*coord)[9].set(B + C, B + C, A + B);
        (*coord)[10].set(B + C, B, A + B);
        (*coord)[11].set(B, B, A + B);

        float CT, BT;
        CT = 4.0f;
        BT = (B / C) * CT;

        std::vector<osg::Vec2> tc;
        tc.reserve(12);
        tc.push_back(osg::Vec2(0.0f, 0.0f));
        tc.push_back(osg::Vec2(2.0f * (BT) + CT, 0.0f));
        tc.push_back(osg::Vec2(2.0f * (BT) + CT, (-(2.0f * (BT) + CT)) / 2.0f));
        tc.push_back(osg::Vec2(0.0f, (-(2.0f * (BT) + CT)) / 2.0f));
        tc.push_back(osg::Vec2(0.0f, 0.0f));
        tc.push_back(osg::Vec2(BT + CT + BT, 0.0f));
        tc.push_back(osg::Vec2(BT + CT + BT, (-(BT + CT + BT)) / 2.0f));
        tc.push_back(osg::Vec2(0.0f, (-(BT + CT + BT)) / 2.0f));
        tc.push_back(osg::Vec2(BT, (-(BT)) / 2.0f));
        tc.push_back(osg::Vec2(BT + CT, (-(BT)) / 2.0f));
        tc.push_back(osg::Vec2(BT + CT, (-(BT + CT)) / 2.0f));
        tc.push_back(osg::Vec2(BT, (-(BT + CT)) / 2.0f));

        (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);

        float isqrtwo = 1.0 / sqrt(2.0);

        for (int i = 0; i < 4; ++i)
        {
            (*normal)[0 * 4 + i].set(0.0, 1.0, 0.0);
            (*normal)[1 * 4 + i].set(1.0, 0.0, 0.0);
            (*normal)[2 * 4 + i].set(0.0, -1.0, 0.0);
            (*normal)[3 * 4 + i].set(-1.0, 0.0, 0.0);
            (*normal)[4 * 4 + i].set(0.0, isqrtwo, isqrtwo);
            (*normal)[5 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
            (*normal)[6 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
            (*normal)[7 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
            (*normal)[8 * 4 + i].set(0.0, 0.0, 1.0);
        }

        ushort *vertices = new ushort[9 * 4];

        vertices[0] = 0;
        vertices[1] = 4;
        vertices[2] = 5;
        vertices[3] = 1;
        vertices[4] = 1;
        vertices[5] = 5;
        vertices[6] = 6;
        vertices[7] = 2;
        vertices[8] = 2;
        vertices[9] = 6;
        vertices[10] = 7;
        vertices[11] = 3;
        vertices[12] = 3;
        vertices[13] = 7;
        vertices[14] = 4;
        vertices[15] = 0;
        vertices[16] = 4;
        vertices[17] = 8;
        vertices[18] = 9;
        vertices[19] = 5;
        vertices[20] = 5;
        vertices[21] = 9;
        vertices[22] = 10;
        vertices[23] = 6;
        vertices[24] = 6;
        vertices[25] = 10;
        vertices[26] = 11;
        vertices[27] = 7;
        vertices[28] = 7;
        vertices[29] = 11;
        vertices[30] = 8;
        vertices[31] = 4;
        vertices[32] = 11;
        vertices[33] = 10;
        vertices[34] = 9;
        vertices[35] = 8;

        for (int i = 0; i < 36; ++i)
        {
            (*texcoord)[i] = tc[vertices[i]];
        }

        coordIndex = new DrawElementsUShort(PrimitiveSet::QUADS, 36, vertices);

        delete[] vertices;

        textureMaterial = new Material();
        textureMaterial->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        textureMaterial->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
        textureMaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        textureMaterial->setSpecular(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
        textureMaterial->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
        textureMaterial->setShininess(Material::FRONT_AND_BACK, 80.0f);
    }
}

void OSGVruiPanelGeometry::attachGeode(vruiTransformNode *node)
{

    createSharedLists();

    OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(geometry->getTextureName()));
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
        VRUILOG("OSGVruiPanelGeometry::createSharedLists err: texture image "
                << geometry->getTextureName() << " not found")
    }

    ref_ptr<Geode> geode = new Geode();

    ref_ptr<Geometry> geometry = new Geometry();
    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(coordIndex.get());
    geometry->setColorArray(color.get());
    geometry->setColorBinding(Geometry::BIND_OVERALL);
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geometry->setTexCoordArray(0, texcoord.get());

    ref_ptr<StateSet> stateSet = geode->getOrCreateStateSet();
    //geostate->setAttr(PFSTATE_BACKMTL, textureMat);

    ref_ptr<CullFace> cullFace = OSGVruiPresets::getCullFaceBack();
    ref_ptr<PolygonMode> polyMode = OSGVruiPresets::getPolyModeFill();

    stateSet->setAttributeAndModes(textureMaterial.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(polyMode.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(cullFace.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);

    geode->setStateSet(stateSet.get());

    geode->setName(this->geometry->getTextureName());
    geode->addDrawable(geometry.get());

    OSGVruiTransformNode *transform = dynamic_cast<OSGVruiTransformNode *>(node);
    if (transform)
    {
        transform->getNodePtr()->asGroup()->addChild(geode.get());
    }
    else
    {
        VRUILOG("OSGVruiPanelGeometry::attachGeode: err: node to attach to is no transform node")
    }
}

float OSGVruiPanelGeometry::getWidth() const
{
    return 2 * B + C;
}

float OSGVruiPanelGeometry::getHeight() const
{
    return 2 * B + C;
}

float OSGVruiPanelGeometry::getDepth() const
{
    return A + B;
}
}
