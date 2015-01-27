/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <malloc.h>
#include <string.h>
#include <GL/glew.h>

#include "utils.h"

bool glew_init()
{
    return glewInit() == GLEW_OK;
}

void renderVBO(GLuint vertex, GLuint velocity, GLuint vorticity,
               unsigned int offset, unsigned int size, unsigned int howmany)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vertex);
    glVertexPointer(3, GL_FLOAT, 0, 0);

    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, velocity);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);

    if (size > 0)
        glDrawArrays(GL_POINTS, offset * size, size * howmany);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glPopClientAttrib();
    glPopAttrib();
}

void createVBO(GLuint *vbo, unsigned int size)
{
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    printf("createVBO %d: size %d\n", *vbo, (int)size);
}

void deleteVBO(GLuint *vbo)
{
    if (*vbo != 0)
    {
        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);

        *vbo = 0;
    }
}
