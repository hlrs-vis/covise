/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
// Some common definition, table, constants for cuda IsoSurface computation

#ifndef CUDA_ISO_COMMON_H
#define CUDA_ISO_COMMON_H

#include <gpu/cutil.h>
#include <gpu/cutil_math.h>
#include <cuda_gl_interop.h>

struct TypeInfo
{
    char Entry; //entry in array
    char NumOfEdges;
    short NumOfVertexes;
};

__constant__ struct TypeInfo typeMap[11] = {
    { -1, 0, 0 },
    { -1, 0, 0 },
    { -1, 0, 0 },
    { -1, 0, 0 },
    { 0, 6, 4 },
    { 1, 8, 5 },
    { -1, 0, 0 },
    { 2, 12, 6 },
    { -1, 0, 0 },
    { -1, 0, 0 },
    { -1, 0, 0 }
};
////////////////////////////////////////////////////////////////
//
// Device constants
////////////////////////////////////////////////////////////////

__constant__ int thetraFaces[12] = { 0, 2, 1,
                                     0, 1, 3,
                                     0, 3, 2,
                                     1, 2, 3 };

__constant__ int hexaFaces[24] = { 0, 4, 7, 3,
                                   3, 7, 6, 2,
                                   0, 3, 2, 1,
                                   0, 1, 5, 4,
                                   4, 5, 6, 7,
                                   5, 1, 2, 6 };

__constant__ int2 hexaEdgeToVerts[12] = {
    { 0, 1 }, //0
    { 1, 2 },
    { 2, 3 },
    { 3, 0 }, //3
    { 4, 5 },
    { 5, 6 },
    { 6, 7 }, //6
    { 4, 7 },
    { 0, 4 },
    { 1, 5 }, //9
    { 2, 6 },
    { 3, 7 }
};

////////////////////////////////////////////////////////////////
//
// Textures containing look-up tables
////////////////////////////////////////////////////////////////

texture<uint, 1, cudaReadModeElementType> hexaTriTex;
texture<uint, 1, cudaReadModeElementType> hexaNumVertsTex;
texture<uint, 1, cudaReadModeElementType> tetraTriTex;
texture<uint, 1, cudaReadModeElementType> tetraNumVertsTex;
texture<uint, 1, cudaReadModeElementType> pyrTriTex;
texture<uint, 1, cudaReadModeElementType> pyrNumVertsTex;

////////////////////////////////////////////////////////////////
//
// Constants
////////////////////////////////////////////////////////////////

__constant__ float EPSILON = 1.0e-10f;

////////////////////////////////////////////////////////////////
//
// Common host and device functions
////////////////////////////////////////////////////////////////

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Fast mul on G8x / G9x / G100
#define IMUL(a, b) __mul24(a, b)

// Delete VBO
inline void deleteVBO(GLuint *vbo)
{
    if (*vbo != 0)
    {
        //CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));

        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);

        *vbo = 0;
    }
}

// Create VBO
void createVBO(GLuint *vbo, unsigned int size)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void createVBOi(GLuint *VBOi, unsigned int size)
{
    glGenBuffers(1, VBOi);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *VBOi);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
#endif //CUDA_ISO_COMMON_H
