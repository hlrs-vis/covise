/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TiledDisplayOSGTexQuadCompositor.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include "TiledDisplayServer.h"

#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/TexEnv>
#include <osg/Geometry>
#include <osg/Projection>

#define COMPOSITOR_TEX_SIZE 1024

using namespace osg;

TiledDisplayOSGTexQuadCompositor::TiledDisplayOSGTexQuadCompositor(int channel, TiledDisplayServer *server)
    : TiledDisplayCompositor(channel, server)
{
    compositorNode = 0;
    osg::Viewport *vp = opencover::coVRConfig::instance()->screens[channel].camera->getViewport();
    dimension = TiledDisplayDimension((unsigned)vp->width(), (unsigned)vp->height());
}

TiledDisplayOSGTexQuadCompositor::~TiledDisplayOSGTexQuadCompositor()
{
}

void TiledDisplayOSGTexQuadCompositor::initSlaveChannel()
{

    cerr << "TiledDisplayOSGTexQuadCompositor::initSlaveChannel info: creating channel " << channel << endl
        //           << coVRConfig::instance()->screens[channel].viewportXMin << " " << coVRConfig::instance()->screens[channel].viewportXMax << " "
        //           << coVRConfig::instance()->screens[channel].viewportYMin << " " << coVRConfig::instance()->screens[channel].viewportYMax << endl
        ;

    // Set ortho projection matrix
    osg::Matrix mat;
    mat.makeIdentity();
    opencover::coVRConfig::instance()->screens[channel].camera->setViewMatrix(mat);

    compositorNode = new Projection(osg::Matrix::identity());
    osg::MatrixTransform *modelViewMatrix = new osg::MatrixTransform;
    modelViewMatrix->setMatrix(osg::Matrix::identity());
    modelViewMatrix->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    compositorNode->addChild(modelViewMatrix);

    float texCoordX = ((float)dimension.width) / ((float)COMPOSITOR_TEX_SIZE);
    float texCoordY = ((float)dimension.height) / ((float)COMPOSITOR_TEX_SIZE);

    compositorTexture = new Texture2D();
    ref_ptr<Image> image = new Image();

#ifdef TILE_ENCODE_JPEG
    image->allocateImage(COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 1, GL_RGB, GL_UNSIGNED_BYTE);
#else
    image->allocateImage(COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 1, GL_BGRA, GL_UNSIGNED_BYTE);
#endif
    image->setInternalTextureFormat(GL_RGBA8);

    compositorTexture->setImage(image.get());
    compositorTexture->setFilter(Texture::MIN_FILTER, Texture::NEAREST);
    compositorTexture->setWrap(Texture::WRAP_S, Texture::CLAMP);
    compositorTexture->setWrap(Texture::WRAP_T, Texture::CLAMP);
    compositorTexture->setSubloadCallback(this);

    ref_ptr<Vec3Array> coord = new Vec3Array(4);

    (*coord)[0].set(-1.0f, -1.0f, 0.0f);
    (*coord)[1].set(1.0f, -1.0f, 0.0f);
    (*coord)[2].set(1.0f, 1.0f, 0.0f);
    (*coord)[3].set(-1.0f, 1.0f, 0.0f);

    ref_ptr<Vec2Array> texcoord = new Vec2Array(4);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(texCoordX, 0.0);
    (*texcoord)[2].set(texCoordX, texCoordY);
    (*texcoord)[3].set(0.0, texCoordY);

    ref_ptr<osg::StateSet> stateSet = new StateSet();
    stateSet->setGlobalDefaults();

    ref_ptr<Material> material = new Material();
    ref_ptr<TexEnv> texEnvModulate = new TexEnv();

    material->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    material->setShininess(Material::FRONT_AND_BACK, 16.0f);
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0.3f, 0.3f, 0.3f, 1.0f));
    material->setDiffuse(Material::FRONT_AND_BACK,
                         Vec4(channel == 1 ? 1.0f : 0.0f,
                              channel == 2 ? 1.0f : 0.0f,
                              channel == 3 ? 1.0f : 0.0f,
                              1.0f));

    //texEnvModulate->setMode(TexEnv::MODULATE);
    texEnvModulate->setMode(TexEnv::REPLACE);

    stateSet->setAttributeAndModes(material.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setMode(GL_CULL_FACE, StateAttribute::OFF | StateAttribute::PROTECTED);
    stateSet->setMode(GL_LIGHTING, StateAttribute::OFF | StateAttribute::PROTECTED);
    stateSet->setMode(GL_BLEND, StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setTextureAttributeAndModes(0, texEnvModulate.get(), StateAttribute::ON | StateAttribute::PROTECTED);
    stateSet->setTextureAttributeAndModes(0, compositorTexture.get(), StateAttribute::ON | StateAttribute::PROTECTED);

    ref_ptr<Geometry> geometry = new Geometry();
    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setTexCoordArray(0, texcoord.get());
    geometry->setUseDisplayList(false);

    ref_ptr<Geode> geode = new Geode();

    geode->setStateSet(stateSet.get());
    geode->addDrawable(geometry.get());

    modelViewMatrix->addChild(geode.get());

    opencover::coVRConfig::instance()->screens[channel].sceneView.get()->setSceneData(compositorNode.get());
}

void TiledDisplayOSGTexQuadCompositor::updateTexturesImplementation()
{
}

void TiledDisplayOSGTexQuadCompositor::load(const osg::Texture2D &texture, osg::State &) const
{
    //cerr << "TiledDisplayOSGTexQuadCompositor::load info: called for server " << channel
    //<< " [" << texture.getTextureWidth() << "x" << texture.getTextureHeight() << "]"
    //<< endl;
    glTexImage2D(GL_TEXTURE_2D, 0, texture.getImage()->getInternalTextureFormat(),
                 COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                 texture.getImage()->getPixelFormat(), texture.getImage()->getDataType(),
                 texture.getImage()->data());
}

void TiledDisplayOSGTexQuadCompositor::subload(const osg::Texture2D &, osg::State &) const
{
    //cerr << "TiledDisplayOSGTexQuadCompositor::subload info: called for server " << channel << endl;
    //if (server->isImageAvailable())
    server->copySubImage(const_cast<TiledDisplayOSGTexQuadCompositor *>(this));
    //cerr << "TiledDisplayOSGTexQuadCompositor::subload info: leaving for server " << channel << endl;
}

void TiledDisplayOSGTexQuadCompositor::setSubTexture(int width, int height, const unsigned char *pixels)
{
#ifdef TILE_ENCODE_JPEG
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
#else
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
#endif
}
