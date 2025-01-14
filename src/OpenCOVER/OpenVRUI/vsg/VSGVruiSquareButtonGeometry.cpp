/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiSquareButtonGeometry.h>

#include <OpenVRUI/vsg/VSGVruiPresets.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiTexture.h>

#include <OpenVRUI/sginterface/vruiRendererInterface.h>

#include <OpenVRUI/coSquareButtonGeometry.h>

#include <vsg/all.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{

#define STYLE_IN 1
#define STYLE_OUT 2

float VSGVruiSquareButtonGeometry::A = 0.5f;
float VSGVruiSquareButtonGeometry::B = 0.4f;
float VSGVruiSquareButtonGeometry::C = 9.1f;



/**
    creates the button.
    @param name texture files to load
    it is looking for textures "name".rgb, "name"-selected.rgb and"name"-check.rgb.
*/
VSGVruiSquareButtonGeometry::VSGVruiSquareButtonGeometry(coSquareButtonGeometry *button)
    : vruiButtonProvider(button)
    , myDCS(0)
{

    this->button = button;
}

/// Destructor.
VSGVruiSquareButtonGeometry::~VSGVruiSquareButtonGeometry()
{
    delete myDCS;
    myDCS = 0;
}

void VSGVruiSquareButtonGeometry::createSharedLists()
{
    /* if (coord1 == 0)
    {

       coord1 = new vec3Array(12);
        coord2 = new vec3Array(12);
        coordt1 = new vec3Array(4);
        coordt2 = new vec3Array(4);
        normal1 = new vec3Array(32);
        normal2 = new vec3Array(32);
        normalt = new vec3Array(1);
        texCoord = new vec2Array(4);
        color = new vsg::vec4Array(1);

        ushort *vertices = new ushort[8 * 4];

        (*coord1)[0].set(0.0, (2.0 * (A + B) + C), 0.0);
        (*coord1)[1].set(2.0 * (A + B) + C, (2.0 * (A + B) + C), 0.0);
        (*coord1)[2].set(2.0 * (A + B) + C, 0.0, 0.0);
        (*coord1)[3].set(0.0, 0.0, 0.0);
        (*coord1)[4].set(A, A + B + C + B, A);
        (*coord1)[5].set(A + B + C + B, A + B + C + B, A);
        (*coord1)[6].set(A + B + C + B, A, A);
        (*coord1)[7].set(A, A, A);
        (*coord1)[8].set(A + B, A + B + C, A + B);
        (*coord1)[9].set(A + B + C, A + B + C, A + B);
        (*coord1)[10].set(A + B + C, A + B, A + B);
        (*coord1)[11].set(A + B, A + B, A + B);

        (*coord2)[0].set(0.0, (2.0 * (A + B) + C), 0.0);
        (*coord2)[1].set(2.0 * (A + B) + C, (2.0 * (A + B) + C), 0.0);
        (*coord2)[2].set(2.0 * (A + B) + C, 0.0, 0.0);
        (*coord2)[3].set(0.0, 0.0, 0.0);
        (*coord2)[4].set(A, A + B + C + B, A);
        (*coord2)[5].set(A + B + C + B, A + B + C + B, A);
        (*coord2)[6].set(A + B + C + B, A, A);
        (*coord2)[7].set(A, A, A);
        (*coord2)[8].set(A + B, A + B + C, A - B);
        (*coord2)[9].set(A + B + C, A + B + C, A - B);
        (*coord2)[10].set(A + B + C, A + B, A - B);
        (*coord2)[11].set(A + B, A + B, A - B);

        (*texCoord)[0].set(0.0, 0.0);
        (*texCoord)[1].set(1.0, 0.0);
        (*texCoord)[2].set(1.0, 1.0);
        (*texCoord)[3].set(0.0, 1.0);

        (*coordt1)[3].set(A + B, A + B + C, A + B);
        (*coordt1)[2].set(A + B + C, A + B + C, A + B);
        (*coordt1)[1].set(A + B + C, A + B, A + B);
        (*coordt1)[0].set(A + B, A + B, A + B);

        (*coordt2)[3].set(A + B, A + B + C, A - B);
        (*coordt2)[2].set(A + B + C, A + B + C, A - B);
        (*coordt2)[1].set(A + B + C, A + B, A - B);
        (*coordt2)[0].set(A + B, A + B, A - B);

        (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);

        float isqrtwo = (float)1.0 / sqrt(2.0);

        for (int i = 0; i < 4; ++i)
        {
            (*normal1)[0 * 4 + i].set(0.0, isqrtwo, isqrtwo);
            (*normal1)[1 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
            (*normal1)[2 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
            (*normal1)[3 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
            (*normal1)[4 * 4 + i].set(0.0, isqrtwo, isqrtwo);
            (*normal1)[5 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
            (*normal1)[6 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
            (*normal1)[7 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
        }

        for (int i = 0; i < 4; ++i)
        {
            (*normal2)[0 * 4 + i].set(0.0, isqrtwo, isqrtwo);
            (*normal2)[1 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
            (*normal2)[2 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
            (*normal2)[3 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
            (*normal2)[4 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
            (*normal2)[5 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
            (*normal2)[6 * 4 + i].set(0.0, isqrtwo, isqrtwo);
            (*normal2)[7 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
        }

        (*normalt)[0].set(0.0, 0.0, 1.0);

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

        coordIndex = new DrawElementsUShort(PrimitiveSet::QUADS, 32, vertices);

        delete[] vertices;

        textureMat = new Material();
        textureMat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        textureMat->setAmbient(Material::FRONT_AND_BACK, vsg::vec4(0.2f, 0.2f, 0.2f, 1.0f));
        textureMat->setDiffuse(Material::FRONT_AND_BACK, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        textureMat->setSpecular(Material::FRONT_AND_BACK, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        textureMat->setEmission(Material::FRONT_AND_BACK, vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        textureMat->setShininess(Material::FRONT_AND_BACK, 80.0f);
        ref_ptr<Material> mtl;

        mtl = new Material();
        mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
        mtl->setAmbient(Material::FRONT_AND_BACK, vsg::vec4(0.1f, 0.1f, 0.1f, 1.0f));
        mtl->setDiffuse(Material::FRONT_AND_BACK, vsg::vec4(0.6f, 0.6f, 0.6f, 1.0f));
        mtl->setSpecular(Material::FRONT_AND_BACK, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        mtl->setEmission(Material::FRONT_AND_BACK, vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        mtl->setShininess(Material::FRONT_AND_BACK, 80.0f);

        normalStateSet = new StateSet();
        normalStateSet->setGlobalDefaults();
        normalStateSet->setAttributeAndModes(VSGVruiPresets::getCullFaceBack(), StateAttribute::ON | StateAttribute::PROTECTED);
        normalStateSet->setAttributeAndModes(mtl.get(), StateAttribute::ON | StateAttribute::PROTECTED);
        normalStateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
        normalStateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    }*/
}

ref_ptr<Node> VSGVruiSquareButtonGeometry::createGeode(const string &textureName, int style)
{

    createSharedLists();

  /*  ref_ptr<Geometry> geometry1 = new Geometry();
    ref_ptr<Geometry> geometry2 = new Geometry();

    geometry1->setColorArray(color.get());
    geometry1->setColorBinding(Geometry::BIND_OVERALL);
    if (style == STYLE_OUT)
    {
        geometry1->setVertexArray(coord1.get());
        geometry1->setNormalArray(normal1.get());
    }
    else
    {
        geometry1->setVertexArray(coord2.get());
        geometry1->setNormalArray(normal2.get());
    }
    geometry1->setNormalBinding(Geometry::BIND_PER_VERTEX);
    geometry1->addPrimitiveSet(coordIndex.get());

    geometry2->setColorArray(color.get());
    geometry2->setColorBinding(Geometry::BIND_OVERALL);
    geometry2->setTexCoordArray(0, texCoord.get());
    if (style == STYLE_OUT)
    {
        geometry2->setVertexArray(coordt1.get());
    }
    else
    {
        geometry2->setVertexArray(coordt2.get());
    }
    geometry2->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry2->setNormalArray(normalt.get());
    geometry2->setNormalBinding(Geometry::BIND_OVERALL);

    ref_ptr<StateSet> stateSet = new StateSet();
    stateSet->setGlobalDefaults();
    stateSet->setAttributeAndModes(textureMat.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    VSGVruiTexture *oTex = dynamic_cast<VSGVruiTexture *>(vruiRendererInterface::the()->createTexture(textureName));
    ref_ptr<Texture2D> texture = oTex->getTexture();
    vruiRendererInterface::the()->deleteTexture(oTex);

    if (texture.valid())
    {
        texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        texture->setWrap(Texture::WRAP_S, Texture::CLAMP);
        texture->setWrap(Texture::WRAP_T, Texture::CLAMP);
        stateSet->setTextureAttributeAndModes(0, texture.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    }
    else
    {
        VRUILOG("VSGVruiSquareButtonGeometry::createBox err: texture image " << textureName << " not found")
    }

    stateSet->setAttributeAndModes(VSGVruiPresets::getCullFaceBack(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setAttributeAndModes(VSGVruiPresets::getPolyModeFill(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setRenderingHint(StateSet::OPAQUE_BIN);

    ref_ptr<Geode> geode1 = new Geode();
    ref_ptr<Geode> geode2 = new Geode();

    geode1->setStateSet(normalStateSet.get());
    geode2->setStateSet(stateSet.get());
    geode1->addDrawable(geometry1.get());
    geode2->addDrawable(geometry2.get());

    ref_ptr<Group> group = new Group();
    group->setName(textureName);

    group->addChild(geode1.get());
    group->addChild(geode2.get());

    return group.get();*/
    vsg::ref_ptr<vsg::MatrixTransform> node = MatrixTransform::create();
    return node;
}

void VSGVruiSquareButtonGeometry::createGeometry()
{

    if (myDCS)
        return;

   /* string name = button->getTextureName();
    normalNode = createGeode(name, STYLE_OUT);
    pressedNode = createGeode(name, STYLE_IN);
    highlightNode = createGeode(name + "-selected", STYLE_OUT);
    pressedHighlightNode = createGeode(name + "-selected", STYLE_IN);
    disabledNode = createGeode(name + "-disabled", STYLE_OUT);*/

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

void VSGVruiSquareButtonGeometry::resizeGeometry()
{
}

vruiTransformNode *VSGVruiSquareButtonGeometry::getDCS()
{
    createGeometry();
    return myDCS;
}

void VSGVruiSquareButtonGeometry::switchGeometry(coButtonGeometry::ActiveGeometry active)
{
    createGeometry();
    switchNode->setSingleChildOn(active);
}
}
