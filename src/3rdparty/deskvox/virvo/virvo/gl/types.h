// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


#ifndef VV_GL_TYPES_H
#define VV_GL_TYPES_H


#include <stddef.h>


// OpenGL 1.0
typedef unsigned int        GLenum;
typedef unsigned char       GLboolean;
typedef unsigned int        GLbitfield;
typedef void                GLvoid;
typedef signed char         GLbyte;
typedef short               GLshort;
typedef int                 GLint;
typedef unsigned char       GLubyte;
typedef unsigned short      GLushort;
typedef unsigned int        GLuint;
typedef int                 GLsizei;
typedef float               GLfloat;
typedef float               GLclampf;
#ifdef OSG_AND_QT_ARE_FIXED
/* Qt 5's qopengl.h checks whether GLdouble is a preprocessor definition and defines it to GLfloat otherwise.
 * This borkage triggered OpenSceneGraph to  #define GLdouble to double in * osg/GL.
 * Don't typedef GLdouble as a work-around, as it is not needed anyway. */
typedef double              GLdouble;
#endif
typedef double              GLclampd;

// OpenGL 1.5
typedef ptrdiff_t           GLintptr;
typedef ptrdiff_t           GLsizeiptr;

// OpenGL 2.0
typedef char                GLchar;

#if 0
// OpenGL 3.0 / GL_ARB_sync
typedef int64_t             GLint64;
typedef uint64_t            GLuint64;
typedef struct __GLsync*    GLsync;
#endif


#endif // VV_GL_TYPES_H
