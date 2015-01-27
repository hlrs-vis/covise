/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define GL_GLEXT_PROTOTYPES

#include <cover/coVRPluginSupport.h>
#include "ParallelRenderingDefines.h"
#include "ParallelRenderingOGLTexQuadCompositor.h"
#include "ParallelRenderingServer.h"

void writePGM(char *name, unsigned int *image,
              int width, int height)
{

    FILE *outfile;

    outfile = fopen(name, "w");

    fprintf(outfile, "P2\n#\n");
    fprintf(outfile, "%d %d\n%d\n", width, height, 255);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int pos = x + y * width;
            int r = (image[pos] & 0xFF000000) >> 24;
            int g = (image[pos] & 0xFF0000) >> 16;
            int b = (image[pos] & 0xFF00) >> 8;
            int v = (r + g + b) / 3;
            fprintf(outfile, "%d ", v);
        }
        fprintf(outfile, "\n");
    }

    fclose(outfile);
}

ParallelRenderingOGLTexQuadCompositor::ParallelRenderingOGLTexQuadCompositor(int channel)
    : ParallelRenderingCompositor(channel)
{
    texture = new GLuint[1];
    frame = 0;
}

ParallelRenderingOGLTexQuadCompositor::~ParallelRenderingOGLTexQuadCompositor()
{
    delete[] texture;
}

void ParallelRenderingOGLTexQuadCompositor::initSlaveChannel(bool replaceOSG)
{
    texSize = 1;
    osg::Camera *camera = cover->screens[0].camera.get();
    const osg::Viewport *vp = camera->getViewport();
    while (texSize < vp->width() || texSize < vp->height())
        texSize = texSize << 1;

    glPushAttrib(GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);
    GLubyte *texImage = new GLubyte[texSize * texSize * 4];
    memset(texImage, 255, texSize * texSize * 4);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, texture);

    glBindTexture(GL_TEXTURE_2D, texture[0]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texSize,
                 texSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, texImage);

    glBindTexture(GL_TEXTURE_2D, 0);
    delete texImage;

    glPopAttrib();

    if (replaceOSG)
        cover->screens[channel].sceneView.get()->setSceneData(new osg::Group());
}

void ParallelRenderingOGLTexQuadCompositor::setTexture(int width, int height, const unsigned char *pixels)
{
    glPushAttrib(GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture[0]);

#ifdef PARALLELRENDERING_WRITEPGM
    frame++;

    if (frame % 100 == 0)
    {

        char f[24];
        snprintf(f, 24, "image%02d%06d.pgm", channel, frame);
        fprintf(stderr, "saving image [%s]\n", f);
        writePGM(f, (unsigned int *)pixels, width, height);
    }
#endif
    (void)pixels;

    if (width <= texSize && height <= texSize)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixels);

    this->width = width;
    this->height = height;

    glPopAttrib();
}

void ParallelRenderingOGLTexQuadCompositor::render()
{

    glPushAttrib(GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT | GL_VIEWPORT_BIT);

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    glActiveTextureARB(GL_TEXTURE8_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE7_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE6_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE5_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE4_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE3_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE2_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glDisable(GL_TEXTURE_2D);

    glActiveTextureARB(GL_TEXTURE0_ARB);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glEnable(GL_TEXTURE_2D);

    int win = cover->screens[channel].window;
    glViewport((int)(cover->screens[channel].viewportXMin * cover->windows[win].sx),
               (int)(cover->screens[channel].viewportYMin * cover->windows[win].sy),
               (int)((cover->screens[channel].viewportXMax - cover->screens[channel].viewportXMin) * cover->windows[win].sx),
               (int)((cover->screens[channel].viewportYMax - cover->screens[channel].viewportYMin) * cover->windows[win].sy));

    glBindTexture(GL_TEXTURE_2D, texture[0]);

    float texCoordX = ((float)width) / ((float)texSize);
    float texCoordY = ((float)height) / ((float)texSize);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1.0, 1.0, 1.0);

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
    glMatrixMode(GL_TEXTURE);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glBindTexture(GL_TEXTURE_2D, 0);
    glPopAttrib();

    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

    texCoordX = 0;
    texCoordY = 0;
}
