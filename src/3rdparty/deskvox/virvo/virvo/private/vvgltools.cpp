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

#include <iostream>

#include <string.h>
#include <cstdlib>
#include <GL/glew.h>

#include "vvopengl.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvgltools.h"

#include <sstream>


#ifdef __APPLE__

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#endif // __APPLE__


using namespace std;
using namespace virvo;


namespace
{
  bool debugAbortOnGLError = false;
  bool debugPrintBacktrace = false;

  /** Callback function for GL errors.
    If the extension GL_ARB_debug_output is available, this callback
    function will be called automatically if a GL error is generated
    */
#ifndef WINAPI
#define WINAPI
#endif
  void WINAPI debugCallback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/,
      GLsizei /*length*/, GLchar const* message, const GLvoid* /*userParam*/)
  {
    std::cerr << "GL error: " << message << std::endl;
    if(debugPrintBacktrace)
      vvToolshed::printBacktrace();

    if(debugAbortOnGLError)
      abort();
  }
}

  void WINAPI debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
      GLsizei length, GLchar const* message, GLvoid* userParam)
  {
    return debugCallback(source, type, id, severity, length, message, (const GLvoid*)userParam);
  }

//============================================================================
// Method Definitions
//============================================================================

bool vvGLTools::enableGLErrorBacktrace(bool printBacktrace, bool abortOnError)
{
// As of January 2011, only freeglut supports glutInitContextFlags
// with GLUT_DEBUG. This may be outdated in the meantime as may be
// those checks!

  debugPrintBacktrace = printBacktrace;
  debugAbortOnGLError = abortOnError;

#ifdef GL_ARB_debug_output
#if !defined(_WIN32)
  glewInit();
  if (glDebugMessageCallbackARB != NULL)
  {
    cerr << "Init callback function for GL_ARB_debug_output extension" << endl;
    glDebugMessageCallbackARB(debugCallback, NULL);
    return true;
  }
  else
#endif
  {
    cerr << "glDebugMessageCallbackARB not available" << endl;
    return false;
  }
#else
  cerr << "Consider installing GLEW >= 1.5.7 for extension GL_ARB_debug_output" << endl;
  return false;
#endif // GL_ARB_debug_output
}

//----------------------------------------------------------------------------
/** Check OpenGL for errors.
    @param  error string if there was an error, otherwise return NULL
*/
void vvGLTools::printGLError(const char* msg)
{
  const GLenum err = glGetError();
  if(err != GL_NO_ERROR)
  {
    const char* str = (const char*)gluErrorString(err);
    cerr << "GL error: " << msg << ", " << str << endl;
  }
}

vvGLTools::GLInfo vvGLTools::getGLInfo()
{
  GLInfo result;

  result.vendor = (const char*)glGetString(GL_VENDOR);
  result.renderer = (const char*)glGetString(GL_RENDERER);
  result.version = (const char*)glGetString(GL_VERSION);

  return result;
}

//----------------------------------------------------------------------------
/** Checks OpenGL for a specific OpenGL version.
    @param major OpenGL major version to check
    @param minor OpenGL minor version to check
    @param release OpenGL release version to check
    @return true if version is supported
*/
bool vvGLTools::isGLVersionSupported(int major, int minor, int release)
{
  (void)release;
  // Get version string from OpenGL:
  const GLubyte* verstring = glGetString(GL_VERSION);
  if (!verstring) return false;

  int ver[3] = { 0, 0, 0 };
  int idx = 0;
  for (const GLubyte *p = verstring;
      *p && *p != ' ' && idx < 3;
      ++p)
  {
    if (*p == '.')
    {
      ++idx;
    }
    else if (*p >= '0' && *p <= '9')
    {
      ver[idx] *= 10;
      ver[idx] += *p-'0';
    }
    else
      return false;
  }

  vvDebugMsg::msg(3, "GL version ", ver[0], ver[1], ver[2]);

  if(ver[0] < major)
    return false;
  if(ver[0] > major)
    return true;

  if(ver[1] < minor)
    return false;
  if(ver[1] >= minor)
    return true;

  return false;
}

//----------------------------------------------------------------------------
/** Checks OpenGL for a specific extension.
    @param extension OpenGL extension to check for (e.g. "GL_EXT_bgra")
    @return true if extension is supported
*/
bool vvGLTools::isGLextensionSupported(const char* extension)
{
  // Check requested extension name for existence and for spaces:
  const GLubyte* where = (GLubyte*)strchr(extension, ' ');
  if (where || *extension=='\0') return false;

  // Get extensions string from OpenGL:
  const GLubyte* extensions = glGetString(GL_EXTENSIONS);
  if (!extensions) return false;

  // Parse OpenGL extensions string:
  const GLubyte* start = extensions;
  for (;;)
  {
    where = (GLubyte*)strstr((const char*)start, extension);
    if (!where) return false;
    const GLubyte* terminator = where + strlen(extension);
    if (where==start || *(where - 1)==' ')
      if (*terminator==' ' || *terminator=='\0')
        return true;
    start = terminator;
  }
}

//----------------------------------------------------------------------------
/** Display the OpenGL extensions which are supported by the system at
  run time.
  @param style display style
*/
void vvGLTools::displayOpenGLextensions(const DisplayStyle style)
{
  char* extCopy;                                  // local copy of extensions string for modifications

  const char* extensions = (const char*)glGetString(GL_EXTENSIONS);

  switch (style)
  {
    default:
    case CONSECUTIVE:
      cerr << extensions << endl;
      break;
    case ONE_BY_ONE:
      extCopy = new char[strlen(extensions) + 1];
      strcpy(extCopy, extensions);
      for (int i=0; i<(int)strlen(extCopy); ++i)
        if (extCopy[i] == ' ') extCopy[i] = '\n';
      cerr << extCopy << endl;
      delete[] extCopy;
      break;
  }
}

//----------------------------------------------------------------------------
/** Check for some specific OpenGL extensions.
  Displays the status of volume rendering related extensions, each on a separate line.
*/
void vvGLTools::checkOpenGLextensions()
{
  const char* status[2] = {"supported", "not found"};

  cerr << "GL_EXT_texture3D...............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_texture3D")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_texture_edge_clamp......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_texture_edge_clamp")) ? status[0] : status[1]) << endl;

  cerr << "GL_SGI_texture_color_table.....";
  cerr << ((vvGLTools::isGLextensionSupported("GL_SGI_texture_color_table")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_paletted_texture........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_paletted_texture")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_blend_equation..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_blend_equation")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_shared_texture_palette..";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_shared_texture_palette")) ? status[0] : status[1]) << endl;

  cerr << "GL_EXT_blend_minmax............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_multitexture............";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_multitexture")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader...........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader2..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader2")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_texture_shader3..........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_texture_shader3")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_texture_env_combine.....";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_texture_env_combine")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_register_combiners.......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_register_combiners")) ? status[0] : status[1]) << endl;

  cerr << "GL_NV_register_combiners2......";
  cerr << ((vvGLTools::isGLextensionSupported("GL_NV_register_combiners2")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_fragment_program........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_fragment_program")) ? status[0] : status[1]) << endl;

  cerr << "GL_ATI_fragment_shader.........";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ATI_fragment_shader")) ? status[0] : status[1]) << endl;

  cerr << "GL_ARB_imaging.................";
  cerr << ((vvGLTools::isGLextensionSupported("GL_ARB_imaging")) ? status[0] : status[1]) << endl;
}


//----------------------------------------------------------------------------
/** Draw view aligned quad. If no vertex coordinates are specified,
    these default to: (-1.0f, -1.0f) (1.0f, 1.0f). No multi texture coordinates
    supported.
*/
void vvGLTools::drawQuad(float x1, float y1, float x2, float y2)
{
  glBegin(GL_QUADS);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glNormal3f(0.0f, 0.0f, 1.0f);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x1, y1);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x2, y1);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(x2, y2);

    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(x1, y2);
  glEnd();
}

//----------------------------------------------------------------------------
/** Query the color specificied using glClearColor (rgba)
*/
vec4 vvGLTools::queryClearColor()
{
  GLfloat tmp[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, tmp);
  return vec4(tmp[0], tmp[1], tmp[2], tmp[3]);
}

std::string virvo::gltools::lastError(const std::string& file, int line)
{
  std::stringstream out;
  const GLenum err = glGetError();
  if(err != GL_NO_ERROR)
  {
    std::string str(reinterpret_cast<const char*>(gluErrorString(err)));
    out << file << ":" << line << ": OpenGL error: " << str;
  }
  return out.str();
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
