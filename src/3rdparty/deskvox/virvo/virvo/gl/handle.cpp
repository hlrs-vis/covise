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


#include "handle.h"

#include <GL/glew.h>


namespace gl = virvo::gl;


void gl::Buffer::destroy()
{
    glDeleteBuffers(1, &name);
}


void gl::Framebuffer::destroy()
{
    glDeleteFramebuffers(1, &name);
}


void gl::Renderbuffer::destroy()
{
    glDeleteRenderbuffers(1, &name);
}


void gl::Texture::destroy()
{
    glDeleteTextures(1, &name);
}


GLuint gl::createBuffer()
{
    GLuint n = 0;
    glGenBuffers(1, &n);
    return n;
}


GLuint gl::createFramebuffer()
{
    GLuint n = 0;
    glGenFramebuffers(1, &n);
    return n;
}


GLuint gl::createRenderbuffer()
{
    GLuint n = 0;
    glGenRenderbuffers(1, &n);
    return n;
}


GLuint gl::createTexture()
{
    GLuint n = 0;
    glGenTextures(1, &n);
    return n;
}
