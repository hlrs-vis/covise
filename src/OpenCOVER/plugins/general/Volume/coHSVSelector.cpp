/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRCollaboration.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <virvo/vvtransfunc.h>
#include <virvo/vvtoolshed.h>
#include "coPinEditor.h"
#include "coFunctionEditor.h"
#include "coHSVSelector.h"
#include "coPreviewCube.h"
#include "VolumePlugin.h"

#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/osg/OSGVruiTexture.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>
#include <OpenVRUI/util/vruiLog.h>

#include <osg/CullFace>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osgUtil/IntersectVisitor>

#include <config/CoviseConfig.h>

using namespace vrui;
using namespace opencover;

#define TEXTURE_RES 64

using namespace osg;

coHSVSelector::coHSVSelector(coPreviewCube *prevCube, coFunctionEditor *functionEditor)
    : vruiCollabInterface(VolumeCoim.get(), "HSV", vruiCollabInterface::HSVWHEEL)
{
    A = 0.5;
    B = 0.4;
    C = 29.1;
    D = 5.0;
    OFFSET = 1.0;
    brightness = 1.0;
    unregister = false;
    myFunctionEditor = functionEditor;
    myCube = prevCube;
    createLists();
    createCross();
    myDCS = new OSGVruiTransformNode(new MatrixTransform());
    crossDCS = new MatrixTransform();
    myDCS->getNodePtr()->asGroup()->addChild(createGeodes().get());
    myDCS->getNodePtr()->asGroup()->addChild(crossDCS.get());
    vruiIntersection::getIntersectorForAction("coAction")->add(myDCS, this);
    setPos(100, -20);
    crossDCS->addChild(crosshair.get());
    setCross(0.5, 0.5);
    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "HSVEditor", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "HSVEditor", coInteraction::Menu);
}

coHSVSelector::coHSVSelector(coFunctionEditor *functionEditor)
    : vruiCollabInterface(VolumeCoim.get(), "HSV", vruiCollabInterface::HSVWHEEL)
{
    A = 0.5;
    B = 0.4;
    C = 29.1;
    D = 5.0;
    OFFSET = 1.0;
    brightness = 1.0;
    unregister = false;
    myFunctionEditor = functionEditor;
    myCube = NULL;
    createLists();
    createCross();
    myDCS = new OSGVruiTransformNode(new MatrixTransform());
    crossDCS = new MatrixTransform();
    myDCS->getNodePtr()->asGroup()->addChild(createGeodes().get());
    myDCS->getNodePtr()->asGroup()->addChild(crossDCS.get());
    vruiIntersection::getIntersectorForAction("coAction")->add(myDCS, this);
    setPos(100, -20);
    crossDCS->addChild(crosshair.get());
    setCross(0.5, 0.5);
    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "HSVEditor", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "HSVEditor", coInteraction::Menu);
}

coHSVSelector::~coHSVSelector()
{
    delete interactionA;
    delete interactionB;

    vruiIntersection::getIntersectorForAction("coAction")->remove(this);
    myDCS->removeAllChildren();
    myDCS->removeAllParents();
    delete myDCS;
}

void coHSVSelector::createGeometry()
{
}

void coHSVSelector::resizeGeometry()
{
}

void coHSVSelector::remoteLock(const char *message)
{
    vruiCollabInterface::remoteLock(message);
}

void coHSVSelector::remoteOngoing(const char *message)
{
    float h, s, v, x, y;
    sscanf(message, "%f %f %f", &h, &s, &v);
    if (myCube)
        myCube->setHS(h, s);
    vvToolshed::convertHS2XY(h, s, &x, &y);
    setCross(x, y);
    myFunctionEditor->setColor(h, s, v, remoteContext);
}

void coHSVSelector::releaseRemoteLock(const char *message)
{
    vruiCollabInterface::releaseRemoteLock(message);
}

void coHSVSelector::setPos(float x, float y, float)
{
    myX = x;
    myY = y;
    myDCS->setTranslation(x, y + getHeight(), 0.0);
}

void coHSVSelector::setColorRGB(float r, float g, float b)
{
    float h, s, v;
    vvToolshed::RGBtoHSB(r, g, b, &h, &s, &v);
    setColorHSB(h, s, v);
}

void coHSVSelector::setColorHSB(float h, float s, float v)
{
    float x, y;
    vvToolshed::convertHS2XY(h, s, &x, &y);
    setCross(x, y);
    if (v != brightness)
    {
        brightness = v;
        if (myCube)
            myCube->setBrightness(brightness);
        vvToolshed::makeColorBoardTexture(TEXTURE_RES, TEXTURE_RES, brightness, textureData);
        Image *image = tex->getImage();
        image->setImage(TEXTURE_RES, TEXTURE_RES, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                        textureData, Image::NO_DELETE, 4);
        image->dirty();
    }
}

void coHSVSelector::setBrightness(float v)
{
    if (v != brightness)
    {
        brightness = v;
        if (myCube)
            myCube->setBrightness(brightness);
        vvToolshed::makeColorBoardTexture(TEXTURE_RES, TEXTURE_RES, brightness, textureData);
        Image *image = tex->getImage();
        image->setImage(TEXTURE_RES, TEXTURE_RES, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                        textureData, Image::NO_DELETE, 4);
        image->dirty();
    }
}

void coHSVSelector::setCross(float x, float y)
{
    crossX = x;
    crossY = y;
    float h, s;
    vvToolshed::convertXY2HS(x, y, &h, &s);
    if (myCube)
        myCube->setHS(h, s);
    Matrix m = crossDCS->getMatrix();
    m.setTrans(x * C + A + B - D / 2.0, (((y - 1) * C) - (A + B)) + D / 2.0, 0.0);
    crossDCS->setMatrix(m);
}

vruiTransformNode *coHSVSelector::getDCS()
{
    return myDCS;
}

void coHSVSelector::sendLockMessageLocal()
{
    static char context[100];
    if (myFunctionEditor->getCurrentPin())
    {
        sprintf(context, "%d", myFunctionEditor->getCurrentPin()->getID());
        sendLockMessage(context);
    }
}

int coHSVSelector::hit(vruiHit *hit)
{

    if (coVRCollaboration::instance()->getSyncMode() == coVRCollaboration::MasterSlaveCoupling
        && !coVRCollaboration::instance()->isMaster())
        return ACTION_DONE;

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionB->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        interactionB->setHitByMouse(hit->isMouseHit());
    }

    if (interactionA->wasStarted() || interactionB->wasStarted())
    {
        sendLockMessageLocal();
    }
    if (interactionB->wasStarted())
    {
        coCoord mouseCoord = cover->getPointerMat();
        lastRoll = mouseCoord.hpr[2];
    }

    if (interactionA->isRunning())
    {
        osgUtil::LineSegmentIntersector::Intersection osgHit = dynamic_cast<OSGVruiHit *>(hit)->getHit();

        if (osgHit.drawable.valid())
        {
            float x, y, h, s;
            Vec3 point = osgHit.getLocalIntersectPoint();
            x = (point[0] - (A + B)) / C;
            y = 1 + ((point[1] + (A + B)) / C);
            vvToolshed::convertXY2HS(x, y, &h, &s);
            if (myCube)
                myCube->setHS(h, s);
            vvToolshed::convertHS2XY(h, s, &x, &y);
            setCross(x, y);
            myFunctionEditor->setColor(h, s, brightness);
            static char textColor[100];
            sprintf(textColor, "%f %f %f", h, s, brightness);
            sendOngoingMessage(textColor);
        }
    }
    return ACTION_CALL_ON_MISS;
}

void coHSVSelector::miss()
{
    unregister = true;
}

void coHSVSelector::update()
{
    if (interactionA->wasStopped() || interactionB->wasStopped())
    {
        sendReleaseMessage(NULL);
    }
    if (interactionB->isRunning())
    {
        coCoord mouseCoord = cover->getPointerMat();
        if (lastRoll != mouseCoord.hpr[2])
        {
            float lastValue = brightness;
            if ((lastRoll - mouseCoord.hpr[2]) > 180)
                lastRoll -= 360;
            if ((lastRoll - mouseCoord.hpr[2]) < -180)
                lastRoll += 360;
            brightness -= (lastRoll - mouseCoord.hpr[2]) / 90.0;
            lastRoll = mouseCoord.hpr[2];
            if (brightness < 0.0)
                brightness = 0.0;
            if (brightness > 1.0)
                brightness = 1.0;
            if (lastValue != brightness)
            {
                if (myCube)
                {
                    myCube->setBrightness(brightness);
                }
                vvToolshed::makeColorBoardTexture(TEXTURE_RES, TEXTURE_RES, brightness, textureData);
                Image *image = tex->getImage();
                image->setImage(TEXTURE_RES, TEXTURE_RES, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                                textureData, Image::NO_DELETE, 4);
                image->dirty();

                float h, s;
                vvToolshed::convertXY2HS(crossX, crossY, &h, &s);
                myFunctionEditor->setColor(h, s, brightness);
                static char textColor[100];
                sprintf(textColor, "%f %f %f", h, s, brightness);
                sendOngoingMessage(textColor);
            }
        }
    }
    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()))
        {
            unregister = false;
        }
    }
}

void coHSVSelector::createLists()
{
    color = new Vec4Array(1);
    coord = new Vec3Array(12);
    coordt = new Vec3Array(4);
    coordCross = new Vec3Array(4);
    normal = new Vec3Array(8);
    normalt = new Vec3Array(1);
    texcoord = new Vec2Array(4);

    ushort *vertices = new ushort[8 * 4];

    (*coord)[0].set(0.0, 0.0, 0.0);
    (*coord)[1].set(2 * (A + B) + C, 0.0, 0.0);
    (*coord)[2].set(2 * (A + B) + C, -(2 * (A + B) + C), 0.0);
    (*coord)[3].set(0.0, -(2 * (A + B) + C), 0.0);
    (*coord)[4].set(A, -A, A);
    (*coord)[5].set(A + B + C + B, -A, A);
    (*coord)[6].set(A + B + C + B, -(A + B + C + B), A);
    (*coord)[7].set(A, -(A + B + C + B), A);
    (*coord)[8].set(A + B, -(A + B), A - B);
    (*coord)[9].set(A + B + C, -(A + B), A - B);
    (*coord)[10].set(A + B + C, -(A + B + C), A - B);
    (*coord)[11].set(A + B, -(A + B + C), A - B);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(1.0, 0.0);
    (*texcoord)[2].set(1.0, 1.0);
    (*texcoord)[3].set(0.0, 1.0);

    (*coordt)[3].set(A + B, -(A + B), A - B);
    (*coordt)[2].set(A + B + C, -(A + B), A - B);
    (*coordt)[1].set(A + B + C, -(A + B + C), A - B);
    (*coordt)[0].set(A + B, -(A + B + C), A - B);

    (*coordCross)[3].set(0.0, 0.0, OFFSET);
    (*coordCross)[2].set(D, 0.0, OFFSET);
    (*coordCross)[1].set(D, -(D), OFFSET);
    (*coordCross)[0].set(0.0, -(D), OFFSET);

    (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);

    float isqrtwo = 1.0 / sqrt(2.0);

    (*normal)[0].set(0.0, isqrtwo, isqrtwo);
    (*normal)[1].set(isqrtwo, 0.0, isqrtwo);
    (*normal)[2].set(0.0, -isqrtwo, isqrtwo);
    (*normal)[3].set(-isqrtwo, 0.0, isqrtwo);
    (*normal)[4].set(0.0, -isqrtwo, isqrtwo);
    (*normal)[5].set(-isqrtwo, 0.0, isqrtwo);
    (*normal)[6].set(0.0, isqrtwo, isqrtwo);
    (*normal)[7].set(isqrtwo, 0.0, isqrtwo);

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

    /*ref_ptr<osg::Material> textureMat = new osg::Material();
   textureMat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
   textureMat->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2, 0.2, 0.2, 1.0));
   textureMat->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
   textureMat->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
   textureMat->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
   textureMat->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);*/
    ref_ptr<osg::Material> mtl;

    mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.1, 0.1, 0.1, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.6, 0.6, 0.6, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);

    normalGeostate = new StateSet();
    normalGeostate->setGlobalDefaults();

    ref_ptr<CullFace> cullFace = new CullFace();
    cullFace->setMode(CullFace::BACK);

    normalGeostate->setAttributeAndModes(cullFace.get(), StateAttribute::ON);
    normalGeostate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
    normalGeostate->setMode(GL_BLEND, StateAttribute::ON);
    normalGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
}

osg::ref_ptr<osg::Group> coHSVSelector::createGeodes()
{

    osg::ref_ptr<osg::Group> group = new osg::Group();

    ref_ptr<Geode> geode1 = new Geode();
    ref_ptr<Geode> geode2 = new Geode();

    ref_ptr<Geometry> geoset1 = new Geometry();
    ref_ptr<Geometry> geoset2 = new Geometry();

    geoset1->setColorArray(color.get());
    geoset1->setColorBinding(Geometry::BIND_OVERALL);
    geoset1->setVertexArray(coord.get());
    geoset1->addPrimitiveSet(coordIndex.get());
    geoset1->setNormalArray(normal.get());
    geoset1->setNormalBinding(Geometry::BIND_PER_VERTEX);

    geoset2->setColorArray(color.get());
    geoset2->setColorBinding(Geometry::BIND_OVERALL);
    geoset2->setVertexArray(coordt.get());
    geoset2->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geoset2->setNormalArray(normalt.get());
    geoset2->setNormalBinding(Geometry::BIND_OVERALL);
    geoset2->setTexCoordArray(0, texcoord.get());

    ref_ptr<StateSet> geostate = new StateSet();
    geostate->setGlobalDefaults();

    geostate->setAttributeAndModes(textureMat.get(), StateAttribute::ON);

    // create Texture Object
    tex = new Texture2D();

    tex->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
    tex->setWrap(Texture::WRAP_S, Texture::CLAMP);
    tex->setWrap(Texture::WRAP_T, Texture::CLAMP);

    textureData = new unsigned char[4 * TEXTURE_RES * TEXTURE_RES];

    vvToolshed::makeColorBoardTexture(TEXTURE_RES, TEXTURE_RES, brightness, textureData);

    Image *image = new Image();
    image->setImage(TEXTURE_RES, TEXTURE_RES, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                    textureData, Image::NO_DELETE, 4);
    tex->setImage(image);

    geostate->setTextureAttributeAndModes(0, tex.get(), StateAttribute::ON);

    geostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    geostate->setAttributeAndModes(OSGVruiPresets::getPolyModeFill(), StateAttribute::ON);
    geostate->setMode(GL_BLEND, StateAttribute::ON);
    geostate->setMode(GL_LIGHTING, StateAttribute::ON);

    geode1->addDrawable(geoset1.get());
    geode2->addDrawable(geoset2.get());

    geode1->setStateSet(normalGeostate.get());
    geode2->setStateSet(geostate.get());

    group->setName("HSVSelector");
    group->addChild(geode1.get());
    group->addChild(geode2.get());
    return group.get();
}

void coHSVSelector::createCross()
{
    ref_ptr<Geometry> geoset2 = new Geometry();

    geoset2->setColorArray(color.get());
    geoset2->setColorBinding(Geometry::BIND_OVERALL);
    geoset2->setTexCoordArray(0, texcoord.get());
    geoset2->setVertexArray(coordCross.get());
    geoset2->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geoset2->setNormalArray(normalt.get());
    geoset2->setNormalBinding(Geometry::BIND_OVERALL);

    ref_ptr<StateSet> geostate = new StateSet();
    geostate->setGlobalDefaults();
    geostate->setAttributeAndModes(textureMat.get(), StateAttribute::ON);

    // create Texture Object
    const char *textureName = "crosshair";
    const char *name = NULL;
    std::string look = covise::coCoviseConfig::getEntry("COVER.LookAndFeel");
    if (!look.empty())
    {
        char *fn = new char[strlen(textureName) + strlen(look.c_str()) + 50];
        sprintf(fn, "share/covise/icons/%s/Volume/%s.rgb", look.c_str(), textureName);
        name = coVRFileManager::instance()->getName(fn);
        delete[] fn;
    }
    if (name == NULL)
    {
        char *fn = new char[strlen(textureName) + 50];
        sprintf(fn, "share/covise/icons/Volume/%s.rgb", textureName);
        name = coVRFileManager::instance()->getName(fn);
        delete[] fn;
    }
    //VRUILOG("coHSVSelector::createCross info: " << textureName << " loading Texture " << name)
    if (name)
    {

        OSGVruiTexture *oTex = dynamic_cast<OSGVruiTexture *>(vruiRendererInterface::the()->createTexture(name));
        tex = oTex->getTexture();
        vruiRendererInterface::the()->deleteTexture(oTex);

        if (!tex.valid())
        {
            VRUILOG("coHSVSelector::createCross warn: could not load texture " << name)
            tex = new Texture2D();
        }

        tex->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
        tex->setWrap(Texture::WRAP_S, Texture::CLAMP);
        tex->setWrap(Texture::WRAP_T, Texture::CLAMP);
        geostate->setTextureAttributeAndModes(0, tex.get(), StateAttribute::ON);
    }
    else
    {
        tex = new Texture2D();
    }

    geostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    geostate->setAttributeAndModes(OSGVruiPresets::getPolyModeFill(), StateAttribute::ON);
    geostate->setMode(GL_BLEND, StateAttribute::ON);
    geostate->setMode(GL_LIGHTING, StateAttribute::ON);
    geostate->setRenderingHint(StateSet::TRANSPARENT_BIN);
    geostate->setNestRenderBins(false);

    crosshair = new Geode();
    crosshair->setStateSet(geostate.get());
    crosshair->addDrawable(geoset2.get());
    crosshair->setNodeMask(crosshair->getNodeMask() & ~Isect::Intersection);
}
