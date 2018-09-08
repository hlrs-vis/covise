// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2012 University of Stuttgart, 2004-2005 Brown University
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


#ifndef VV_GL_HANDLE_H
#define VV_GL_HANDLE_H


#include "../vvexport.h"
#include "types.h"


namespace virvo
{
namespace gl
{


    // A class that manages the lifetime of an OpenGL object.
#define VV_GL_DEFINE_OBJECT(NAME)                                   \
    class NAME {                                                    \
    private:                                                        \
        GLuint name;                                                \
    public:                                                         \
        /* Construct from the given object */                       \
        explicit NAME(GLuint name = 0) : name(name)                 \
        {                                                           \
        }                                                           \
        /* Destroy the currently held OpenGL object */              \
        ~NAME()                                                     \
        {                                                           \
            reset();                                                \
        }                                                           \
        /* Returns the OpenGL name */                               \
        GLuint get() const                                          \
        {                                                           \
            return name;                                            \
        }                                                           \
        /* Reset with another OpenGL object (of the same type!) */  \
        /* Destroy the currently held OpenGL object */              \
        void reset(GLuint n = 0)                                    \
        {                                                           \
            if (name)                                               \
                destroy();                                          \
            name = n;                                               \
        }                                                           \
        /* Release ownership of the OpenGL object */                \
        GLuint release()                                            \
        {                                                           \
            GLuint n = name; name = 0; return n;                    \
        }                                                           \
        /* Destroy the currently held OpenGL object */              \
        VVAPI void destroy();                                       \
    private:                                                        \
        /* NOT COPYABLE! */                                         \
        NAME(NAME const&);                                          \
        NAME& operator =(NAME const&);                              \
    };


    VV_GL_DEFINE_OBJECT(Buffer)

    VV_GL_DEFINE_OBJECT(Framebuffer)

    VV_GL_DEFINE_OBJECT(Renderbuffer)

    VV_GL_DEFINE_OBJECT(Texture)


#undef VV_GL_DEFINE_OBJECT


    // Creates an OpenGL buffer
    VVAPI GLuint createBuffer();

    // Creates an OpenGL framebuffer object
    VVAPI GLuint createFramebuffer();

    // Creates an OpenGL renderbuffer
    VVAPI GLuint createRenderbuffer();

    // Creates an OpenGL texture
    VVAPI GLuint createTexture();


} // namespace gl
} // namespace virvo


#endif // VV_GL_HANDLE_H
