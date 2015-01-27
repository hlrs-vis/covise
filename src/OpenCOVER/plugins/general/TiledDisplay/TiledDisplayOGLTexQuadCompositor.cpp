/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TiledDisplayOGLTexQuadCompositor.h"

#include <cover/coVRPluginSupport.h>

#include "TiledDisplayOGLTexQuadDrawable.h"
#include "TiledDisplayServer.h"

#include <osg/Material>
#include <osg/TexEnv>
#include <osg/Geometry>

using namespace osg;

TiledDisplayOGLTexQuadCompositor::TiledDisplayOGLTexQuadCompositor(int channel, TiledDisplayServer *server)
    : TiledDisplayCompositor(channel, server)
{
    compositorNode = 0;
}

TiledDisplayOGLTexQuadCompositor::~TiledDisplayOGLTexQuadCompositor()
{
}

void TiledDisplayOGLTexQuadCompositor::initSlaveChannel()
{

    cerr << "TiledDisplayOGLTexQuadCompositor::initSlaveChannel info: creating channel " << channel << endl;

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
    stateSet->setTextureAttributeAndModes(0,
                                          texEnvModulate.get(),
                                          StateAttribute::ON | StateAttribute::PROTECTED);

    ref_ptr<TiledDisplayOGLTexQuadDrawable> geometry = new TiledDisplayOGLTexQuadDrawable(this);

    compositorNode = new Geode();
    compositorNode->setStateSet(stateSet.get());
    compositorNode->addDrawable(geometry.get());

    // XXX cover->screens[channel].sceneView->setSceneData(compositorNode.get());
}

void TiledDisplayOGLTexQuadCompositor::setSubTexture(int width, int height, const unsigned char *pixels)
{
#ifdef TILE_ENCODE_JPEG
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
#else
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixels);
#endif
}
