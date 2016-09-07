/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "renderTexture.h"
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <gpu/helper_cuda.h>
#include <gpu/helper_cuda_gl.h>

#include <math.h>
#include <assert.h>


ParticleRenderer::ParticleRenderer()
    : m_texture(0)
    , m_pbo(0)
    , m_vboColor(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
}

void ParticleRenderer::resetPBO()
{
    glDeleteBuffers(1, (GLuint *)&m_pbo);
}


void ParticleRenderer::_drawQuad()
{
    
        glBegin(GL_QUADS);
            glVertex3f(0,0,0);
            glTexCoord2f(0,0);
            glVertex3f(1,0,0);
            glTexCoord2f(1,0);
            glVertex3f(1,1,0);
            glTexCoord2f(1,1);
            glVertex3f(0,1,0);
            glTexCoord2f(0,1);
        glEnd();
}

void ParticleRenderer::display()
{
    _drawQuad();
}

void ParticleRenderer::_initGL()
{


//    _createTexture(32);

}

void ParticleRenderer::_createTexture(int resolution)
{
 /*   unsigned char *data = createGaussianMap(resolution);
    glGenTextures(1, (GLuint *)&m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);*/
}
