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


#ifndef VV_GL_UTIL_H
#define VV_GL_UTIL_H

#include "light.h"

#include "math/math.h"
#include "vvexport.h"
#include "types.h"


#define VV_GET_GL_ERROR() \
    ::virvo::gl::getError(__FILE__, __LINE__)

#define VV_GET_FRAMEBUFFER_STATUS(TARGET) \
    ::virvo::gl::getFramebufferStatus(TARGET, __FILE__, __LINE__)


namespace virvo
{
namespace gl
{


    // Installs a debug callback function
    VVAPI void enableDebugCallback();

    // Returns the OpenGL error code.
    VVAPI GLenum getError(char const* file, int line);

    // Returns the framebuffer status
    VVAPI GLenum getFramebufferStatus(GLenum target, char const* file, int line);

    // Render the 2D texture into the current draw buffer using the given blend function
    VVAPI void blendTexture(GLuint texture, GLenum sfactor, GLenum dfactor);

    // Render the 2D texture into the current draw buffer
    // Assuming premultiplied color values.
    VVAPI void blendTexture(GLuint texture);

    // Render pixels into the current draw buffer using the given blend function
    VVAPI void blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels, GLenum sfactor, GLenum dfactor);

    // Render pixels into the current draw buffer
    // Assuming premultiplied color values.
    VVAPI void blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels);

    // Prepare the stencil buffer for interlaced stereo rendering.
    // If w == 0 || h == 0 uses the current viewport.
    VVAPI void renderInterlacedStereoStencilBuffer(bool lines = true);

    // Sets the modelview matrix
    VVAPI void setModelviewMatrix(mat4 const& mv);

    // Returns the current modelview matrix
    VVAPI mat4 getModelviewMatrix();

    // Sets the projection matrix
    VVAPI void setProjectionMatrix(mat4 const& pr);

    // Returns the current projection matrix
    VVAPI mat4 getProjectionMatrix();

    // Sets the viewport
    VVAPI void setViewport(recti const& vp);

    // Returns the current viewport
    VVAPI recti getViewport();

    // Returns the OpenGL light specified by l
    VVAPI light getLight(GLenum l);


} // namespace gl
} // namespace virvo


#endif
