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


#include "util.h"


#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>

#include <stdio.h>
#include <stdarg.h>


namespace gl = virvo::gl;
using virvo::mat4;
using virvo::recti;


#ifndef APIENTRY
#define APIENTRY
#endif


#ifdef __APPLE__

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#endif // __APPLE__


#if !defined(NDEBUG) && defined(GL_KHR_debug)

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static char const* GetDebugTypeString(GLenum type)
{
    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:
        return "error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "deprecated behavior detected";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "undefined behavior detected";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "portablility warning";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "performance warning";
    case GL_DEBUG_TYPE_OTHER:
        return "other";
    case GL_DEBUG_TYPE_MARKER:
        return "marker";
    }

    return "{unknown type}";
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void APIENTRY DebugCallback( GLenum /*source*/,
                                    GLenum type,
                                    GLuint /*id*/,
                                    GLenum /*severity*/,
                                    GLsizei /*length*/,
                                    const GLchar* message,
                                    GLvoid* /*userParam*/
                                    )
{
    fprintf(stderr, "GL %s: %s\n", GetDebugTypeString(type), message);

    if (type == GL_DEBUG_TYPE_ERROR)
    {
#ifdef _WIN32
        if (IsDebuggerPresent())
            DebugBreak();
#endif
    }
}

#endif // !NDEBUG && GL_KHR_debug


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::enableDebugCallback()
{
#if !defined(NDEBUG) && defined(GL_KHR_debug)
    if (GLEW_KHR_debug)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback((GLDEBUGPROC)DebugCallback, 0);
    }
#endif
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
GLenum gl::getError(char const* file, int line)
{
    GLenum err = glGetError();

    if (err != GL_NO_ERROR)
        fprintf(stderr, "%s(%d) : GL error: %s\n", file, line, gluErrorString(err));

    return err;
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static char const* GetFramebufferStatusString(GLenum status)
{
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        return "framebuffer complete";
    case GL_FRAMEBUFFER_UNDEFINED:
        return "framebuffer undefined";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        return "framebuffer incomplete attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        return "framebuffer incomplete missing attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        return "framebuffer incomplete draw buffer";
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        return "framebuffer incomplete read buffer";
    case GL_FRAMEBUFFER_UNSUPPORTED:
        return "framebuffer unsupported";
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        return "framebuffer incomplete multisample";
#ifdef GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
        return "framebuffer incomplete layer targets";
#endif
    default:
        return "{unknown status}";
    }
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
GLenum gl::getFramebufferStatus(GLenum target, char const* file, int line)
{
    GLenum status = glCheckFramebufferStatus(target);

    if (status != GL_FRAMEBUFFER_COMPLETE)
        fprintf(stderr, "%s(%d) : GL framebuffer error: %s\n", file, line, GetFramebufferStatusString(status));

    return status;
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static void DrawFullScreenQuad()
{
    glPushAttrib(GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glActiveTexture(GL_TEXTURE0);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glPopAttrib();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendTexture(GLuint texture, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glActiveTexture(GL_TEXTURE0);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glDepthMask(GL_FALSE);

    DrawFullScreenQuad();

    glPopAttrib();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendTexture(GLuint texture)
{
    blendTexture(texture, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels, GLenum sfactor, GLenum dfactor)
{
    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_PIXEL_MODE_BIT);

    GLint viewport[4];

    glGetIntegerv(GL_VIEWPORT, &viewport[0]);

    glWindowPos2i(viewport[0], viewport[1]);

    GLfloat scaleX = static_cast<GLfloat>(viewport[2]) / srcW;
    GLfloat scaleY = static_cast<GLfloat>(viewport[3]) / srcH;

    glPixelZoom(scaleX, scaleY);

    glEnable(GL_BLEND);
    glBlendFunc(sfactor, dfactor);

    glDrawPixels(srcW, srcH, format, type, pixels);

    glPopAttrib();
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::blendPixels(GLsizei srcW, GLsizei srcH, GLenum format, GLenum type, const GLvoid* pixels)
{
    blendPixels(srcW, srcH, format, type, pixels, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}


//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
void gl::renderInterlacedStereoStencilBuffer(bool lines)
{
    static const GLubyte kPatternLines[32*(32/8)] = {
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
        0xFF, 0xFF, 0xFF, 0xFF,
        0x00, 0x00, 0x00, 0x00,
    };

    static const GLubyte kPatternCheckerBoard[32*(32/8)] = {
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
        0xAA, 0xAA, 0xAA, 0xAA,
        0x55, 0x55, 0x55, 0x55,
    };

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_LIGHTING);

    glColorMask(0, 0, 0, 0);
    glDepthMask(0);

    glClearStencil(0);
    glClear(GL_STENCIL_BUFFER_BIT);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glEnable(GL_POLYGON_STIPPLE);
    if (lines)
        glPolygonStipple(kPatternLines);
    else
        glPolygonStipple(kPatternCheckerBoard);

    glEnable(GL_STENCIL_TEST);
    glStencilMask(0xFFFFFFFF);
    glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
    glStencilFunc(GL_ALWAYS, 1, 0xFFFFFFFF);

    DrawFullScreenQuad();

    glPopAttrib();
    glPopClientAttrib();
}


void gl::setModelviewMatrix(mat4 const& mv)
{

    GLint old;
    glGetIntegerv(GL_MATRIX_MODE, &old);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mv.data());

    glMatrixMode(old);

}


mat4 gl::getModelviewMatrix()
{
    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    return mat4(m);
}


void gl::setProjectionMatrix(mat4 const& mv)
{

    GLint old;
    glGetIntegerv(GL_MATRIX_MODE, &old);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(mv.data());

    glMatrixMode(old);

}

mat4 gl::getProjectionMatrix()
{
    GLfloat m[16];
    glGetFloatv(GL_PROJECTION_MATRIX, m);
    return mat4(m);
}

void gl::setViewport(recti const& vp)
{
    glViewport(vp.x, vp.y, vp.w, vp.h);
}


recti gl::getViewport()
{
    GLint v[4];
    glGetIntegerv(GL_VIEWPORT, v);
    return recti(v);
}

gl::light gl::getLight(GLenum l)
{
    gl::light result;
    glGetLightfv(l, GL_AMBIENT,                reinterpret_cast< GLfloat* >(&result.ambient));
    glGetLightfv(l, GL_DIFFUSE,                reinterpret_cast< GLfloat* >(&result.diffuse));
    glGetLightfv(l, GL_SPECULAR,               reinterpret_cast< GLfloat* >(&result.specular));
    glGetLightfv(l, GL_POSITION,               reinterpret_cast< GLfloat* >(&result.position));
    glGetLightfv(l, GL_SPOT_DIRECTION,         reinterpret_cast< GLfloat* >(&result.spot_direction));
    glGetLightfv(l, GL_SPOT_CUTOFF,            reinterpret_cast< GLfloat* >(&result.spot_cutoff));
    glGetLightfv(l, GL_CONSTANT_ATTENUATION,   reinterpret_cast< GLfloat* >(&result.constant_attenuation));
    glGetLightfv(l, GL_LINEAR_ATTENUATION,     reinterpret_cast< GLfloat* >(&result.linear_attenuation));
    glGetLightfv(l, GL_QUADRATIC_ATTENUATION,  reinterpret_cast< GLfloat* >(&result.quadratic_attenuation));
    return result;
}


