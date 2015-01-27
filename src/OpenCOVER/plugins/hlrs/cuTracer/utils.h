/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UTILS_H
#define UTILS_H

#include <GL/gl.h>

#include <cuda.h>
#include <cutil.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

bool glew_init();
void createVBO(GLuint *vbo, unsigned int size);
void deleteVBO(GLuint *vbo);
void renderVBO(GLuint vertex, GLuint velocity, GLuint vorticity,
               unsigned int offset, unsigned int size, unsigned int howmany);

#endif
