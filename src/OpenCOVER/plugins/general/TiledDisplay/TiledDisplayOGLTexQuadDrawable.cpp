/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include "TiledDisplayOGLTexQuadDrawable.h"
#include "TiledDisplayDimension.h"
#include "TiledDisplayServer.h"
#include "TiledDisplayCompositor.h"

#define COMPOSITOR_TEX_SIZE 1024
#define COMPOSITOR_TEX_TYPE GLubyte

TiledDisplayOGLTexQuadDrawable::TiledDisplayOGLTexQuadDrawable(TiledDisplayCompositor *compositor)
{

    this->compositor = compositor;
    this->server = compositor->getServer();
    this->channel = compositor->getChannel();

    int width, height;
    osg::Viewport *vp = opencover::coVRConfig::instance()->channels[channel].camera->getViewport();

    width = (unsigned)vp->width();
    height = (unsigned)vp->height();
    texture = new GLuint[1];

    dimension = TiledDisplayDimension(width, height);

    cerr << "TiledDisplayOGLTexQuadDrawable::<init> info: creating for server " << channel
         << ", width=" << width << " height=" << height
         << endl;

#ifdef TILE_ENCODE_JPEG
    pixelFormat = GL_RGB;
#else
    pixelFormat = GL_RGBA8;
#endif
    pixelType = GL_UNSIGNED_BYTE;

    pixels = 0;
    texImage = new COMPOSITOR_TEX_TYPE[COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4];
    memset(texImage, 255, COMPOSITOR_TEX_SIZE * COMPOSITOR_TEX_SIZE * 4);

    initPending = true;

    currentState = new osg::StateSet();

    setUseDisplayList(false);
}

TiledDisplayOGLTexQuadDrawable::~TiledDisplayOGLTexQuadDrawable()
{
    cerr << "TiledDisplayOGLTexQuadDrawable::<init> info: destroying for server " << channel << endl;
}

void TiledDisplayOGLTexQuadDrawable::drawImplementation(osg::RenderInfo &renderInfo) const
{

    //cerr << "TiledDisplayOGLTexQuadDrawable::drawImplementation info: called for server " << channel << endl;

    renderInfo.getState()->captureCurrentState(*currentState);
    renderInfo.getState()->pushStateSet(currentState);

    glEnable(GL_TEXTURE_2D);

    if (initPending)
    {
        cerr << "TiledDisplayOGLTexQuadDrawable::drawImplementation info: init for server " << channel << endl;
        glGenTextures(1, texture);
        glBindTexture(GL_TEXTURE_2D, texture[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, pixelFormat, COMPOSITOR_TEX_SIZE, COMPOSITOR_TEX_SIZE, 0,
                     pixelFormat, pixelType, texImage);
        initPending = false;
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, texture[0]);
    }

    server->copySubImage(compositor);

    float texCoordX = ((float)dimension.width) / ((float)COMPOSITOR_TEX_SIZE);
    float texCoordY = ((float)dimension.height) / ((float)COMPOSITOR_TEX_SIZE);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glBegin(GL_QUADS);
    glTexCoord2f(texCoordX, texCoordY);
    glVertex3f(1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, texCoordY);
    glVertex3f(-1.0f, 1.0f, 0.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glTexCoord2f(texCoordX, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    renderInfo.getState()->popStateSet();
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
osg::BoundingBox TiledDisplayOGLTexQuadDrawable::computeBoundingBox() const
#else
osg::BoundingBox TiledDisplayOGLTexQuadDrawable::computeBound() const
#endif
{
    return osg::BoundingBox(-dimension.width, 0.0f, -dimension.height, dimension.width, 0.0f, dimension.height);
}

osg::Object *TiledDisplayOGLTexQuadDrawable::cloneType() const
{
    cerr << "TiledDisplayOGLTexQuadDrawable::cloneType warn: executed" << endl;
    return new TiledDisplayOGLTexQuadDrawable(compositor);
}

osg::Object *TiledDisplayOGLTexQuadDrawable::clone(const osg::CopyOp &) const
{
    cerr << "TiledDisplayOGLTexQuadDrawable::clone warn: executed" << endl;
    return new TiledDisplayOGLTexQuadDrawable(compositor);
}
