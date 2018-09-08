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

#include <GL/glew.h>
#include "vvcgprogram.h"
#include "vvdebugmsg.h"
#include "vvglslprogram.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <locale>

using std::string;
using std::cerr;
using std::endl;

vvShaderFactory::vvShaderFactory()
{
  _shaderDir = string();
  for(int i=0;i<3;i++)
  {
    _shaderName[i] = string();
    _fileString[i] = string();
  }

  glewInit();
}

void vvShaderFactory::setDefines(const std::string &defines)
{
  _defines = defines;
}

bool vvShaderFactory::loadFragmentLibrary(const std::string &file)
{
  _libFileString.clear();
  if (!file.empty())
  {
    if(_shaderDir.empty())
      _shaderDir = getShaderDir();

    try
    {
      _libFileString = vvToolshed::file2string(_shaderDir + "/vv_" + file + ".fsh");
    } 
    catch (std::exception &e)
    {
      std::cerr << e.what() << std::endl;
      return false;
    }
  }

  return true;
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& name)
{
  return createProgram(name, name, name);
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& vert, const std::string& geom, const std::string& frag)
{
  if(vert.empty() && geom.empty() && frag.empty())
    return NULL;

  _shaderName[0].clear();
  _shaderName[1].clear();
  _shaderName[2].clear();

  if(_shaderDir.empty())
    _shaderDir = getShaderDir();

  vvShaderProgram *program = NULL;

  if(!isSupported("glsl"))
  {
    vvDebugMsg::msg(0, "vvShaderFactory::createProgram: no GLSL support!");
  }
  else
  {
    if(!vert.empty()) _shaderName[0] = "vv_" + vert + ".vsh";
    if(!geom.empty()) _shaderName[1] = "vv_" + geom + ".gsh";
    if(!frag.empty()) _shaderName[2] = "vv_" + frag + ".fsh";
    bool loaded = loadFileStrings();

    if(loaded)
    {
      vvDebugMsg::msg(2, "GLSL-shaders found:");

      for(int i=0;i<3;i++)
      {
        if(_fileString[i].length() > 0)
        {
          vvDebugMsg::msg(2, _shaderName[i].c_str());

          // prepend defines and fragment shader library, but keep #version on first line
          std::string version;
          std::string::size_type lineend = _fileString[i].find_first_of("\n\r");
          if (lineend != std::string::npos)
          {
            version = _fileString[i].substr(0, lineend);
            if (version.find("#version") == std::string::npos)
            {
              version.clear();
            }
            else
            {
              _fileString[i] = _fileString[i].substr(lineend);
            }
          }

          if (i == 2)
            _fileString[i] = version + _defines + _libFileString + _fileString[i];
          else
            _fileString[i] = version + _defines + _fileString[i];
        }
      }

      program = new vvGLSLProgram(_fileString[0], _fileString[1], _fileString[2], _geoShaderArgs);
      if(!program->isValid())
      {
        delete program;
        program = NULL;
      }
    }
  }
#ifdef HAVE_CG
  if(!program)
  {
    if(!isSupported("cg"))
    {
      vvDebugMsg::msg(0, "vvShaderFactory::createProgram: no CG support!");
    }
    else
    {
      if(!vert.empty()) _shaderName[0] = "vv_" + vert + ".vert.cg";
      if(!geom.empty()) _shaderName[1] = "vv_" + geom + ".geom.cg";
      if(!frag.empty()) _shaderName[2] = "vv_" + frag + ".frag.cg";

      bool loaded = loadFileStrings();

      if(loaded)
      {
        cerr << "CG-shaders found: ";

        for(int i=0;i<3;i++)
          if(_fileString[i].length() > 0)
            cerr << _shaderName[i] << " ";
        cerr << endl;

        program = new vvCgProgram(_fileString[0], _fileString[1], _fileString[2], _geoShaderArgs);
        if(!program->isValid())
        {
          delete program;
          program = NULL;
        }
      }
    }
  }
#endif

  if(!program)
  {
    string errmsg = "No supported shaders with name " + vert + " " + geom + " or " + frag + " found!";
    vvDebugMsg::msg(0, errmsg.c_str());
  }

  return program;
}

vvShaderProgram* vvShaderFactory::createProgram(const std::string& vert, const std::string& geom, const std::string& frag,
                                                const vvShaderProgram::GeoShaderArgs& geoShaderArgs)
{
  if (geom.empty())
  {
    vvDebugMsg::msg(0, "vvShaderFactory::createProgram(): Geometry shader args specified but no geometry shader supplied");
  }
  _geoShaderArgs = geoShaderArgs;
  return createProgram(vert, geom, frag);
}

bool vvShaderFactory::loadFileStrings()
{
  try
  {
    for (size_t i = 0; i < 3; ++i)
    {
      if (_shaderName[i].empty())
        _fileString[i] = "";
      else
        _fileString[i] = vvToolshed::file2string(_shaderDir + _shaderName[i]);
    }

    return true;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return false;
  }
}

const string vvShaderFactory::getShaderDir()
{
  string result;

  const char* shaderEnv = "VV_SHADER_PATH";
  if (getenv(shaderEnv))
  {
    cerr << "Environment variable " << shaderEnv << " found: " << getenv(shaderEnv) << endl;
    result = getenv(shaderEnv);
  }
  else
  {
#define STRINGIFY(x) #x
#ifdef VIRVO_SHADER_DIR
	  result = STRINGIFY(VIRVO_SHADER_DIR);
#else
    cerr << "Warning: you should set the environment variable " << shaderEnv << " to point to your shader directory" << endl;
    static char shaderDir[256];
#ifdef _WIN32
    const char* primaryWin32ShaderDir = "..\\..\\..\\virvo\\shader";
    vvToolshed::getProgramDirectory(shaderDir, 256);
    strcat(shaderDir, primaryWin32ShaderDir);
    cerr << "Trying shader path: " << shaderDir << endl;
    if (!vvToolshed::isDirectory(shaderDir))
    {
       vvToolshed::getProgramDirectory(shaderDir, 256);
    }
    cerr << "Using shader path: " << shaderDir << endl;
    result = shaderDir;
#else
    const char* deskVoxShaderPath = "/..";
#ifdef SHADERDIR
    result = SHADERDIR;
#else
    vvToolshed::getProgramDirectory(shaderDir, 256);
    strcat(shaderDir, deskVoxShaderPath);
    result = shaderDir;
#endif
#endif
#endif
  }
#ifdef _WIN32
  result += "\\";
#else
  result += "/";
#endif

  return result;
}

bool vvShaderFactory::isSupported(const std::string& lang)
{
  std::string str = lang;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "cg")
  {
#ifdef HAVE_CG
    return true;
#else
    return false;
#endif
  }
  else if (str == "glsl")
  {
#if defined GL_VERSION_1_1 || defined GL_VERSION_1_2 \
 || defined GL_VERSION_1_3 || defined GL_VERSION_1_4 \
 || defined GL_VERSION_1_5 || defined GL_VERSION_2_0 \
 || defined GL_VERSION_3_0
    // Assume that even compilers that support higher gl versions
    // will know at least one of those listed here.
    return glCreateProgram && glDeleteProgram && glUniform1f;
#else
    return false;
#endif
  }
  else
  {
    return false;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
