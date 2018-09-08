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

#include "vvdebugmsg.h"
#include "vvglslprogram.h"
#include "vvtoolshed.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

#include <cassert>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;
using std::string;

namespace
{
  enum TextureType
  {
    TEXTURE_1D = 0,
    TEXTURE_2D,
    TEXTURE_3D
  };

  struct Texture
  {
    GLint       _id;
    GLint       _unit;
    TextureType _type;
    GLint       _uniform;
    Texture() : _id(-1), _unit(-1), _type(TEXTURE_1D), _uniform(-1) {}
  };
}

typedef std::map<std::string, GLint> ParaMap;
typedef std::map<std::string, Texture*> TextureMap;
typedef TextureMap::iterator TextureIterator;

struct vvGLSLData
{
  vvGLSLData()
    : programId(0)
    , nTexture(0)
  {

  }

  GLuint      programId;
  GLuint      shaderId[3];
  ParaMap     parameterMaps;
  TextureMap  textureNameMaps; ///< maps of texturename on texture unit
  GLuint      nTexture;        ///< counter for texture units
};

namespace
{
  Texture* getTexture(vvGLSLData* data, const string& parameterName, const string& parameterType)
  {
    TextureIterator texIterator = data->textureNameMaps.find(parameterName);
    if (texIterator != data->textureNameMaps.end())
    {
      return texIterator->second;
    }
    else
    {
      Texture* newTex = new Texture;
      newTex->_uniform = glGetUniformLocation(data->programId, parameterName.c_str());
      if(newTex->_uniform == -1)
      {
        string errmsg;
        errmsg = parameterType + "(" + parameterName
                 + ") does not correspond to an active uniform variable in program";
        vvDebugMsg::msg(1, errmsg.c_str());
      }
      else
      {
        newTex->_unit = data->nTexture++;
      }
      data->textureNameMaps[parameterName] = newTex;
      return newTex;
    }
  }

  GLint getUniform(vvGLSLData* data, const string& parameterName, const string& parameterType)
  {
    if (data->parameterMaps.find(parameterName) != data->parameterMaps.end())
    {
      return data->parameterMaps[parameterName];
    }
    else
    {
      const GLint uniform = glGetUniformLocation(data->programId, parameterName.c_str());
      if (uniform == -1)
      {
        string errmsg;
        errmsg = parameterType + "(" + parameterName
                 + ") does not correspond to an active uniform variable in program";
        vvDebugMsg::msg(1, errmsg.c_str());
      }

      data->parameterMaps[parameterName] = uniform;
      return uniform;
    }
  }
}

vvGLSLProgram::vvGLSLProgram(const string& vert, const string& geom, const string& frag)
  : vvShaderProgram(vert, geom, frag)
  , _data(new vvGLSLData)
{
  for(int i=0; i<3;i++)
    _data->shaderId[i] = 0;

  _shadersLoaded = loadShaders();
  if (!_shadersLoaded)
  {
    vvDebugMsg::msg(1, "vvGLSLProgram::vvGLSLProgram() Loading Shaders failed!");
  }
}

vvGLSLProgram::vvGLSLProgram(const string& vert, const string& geom, const string& frag,
                             const vvShaderProgram::GeoShaderArgs& geoShaderArgs)
  : vvShaderProgram(vert, geom, frag, geoShaderArgs)
  , _data(new vvGLSLData)
{
  for(int i=0; i<3;i++)
    _data->shaderId[i] = 0;

  _shadersLoaded = loadShaders();
  if(!_shadersLoaded)
  {
    vvDebugMsg::msg(1, "vvGLSLProgram::vvGLSLProgram() Loading Shaders failed!");
  }
}

vvGLSLProgram::~vvGLSLProgram()
{
  for (TextureIterator it = _data->textureNameMaps.begin();
       it != _data->textureNameMaps.end(); ++it)
  {
    delete it->second;
  }

  disable();
  if (_data->programId)
  {
    glDeleteProgram(_data->programId);
  }
  delete _data;
}

bool vvGLSLProgram::loadShaders()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::loadShaders()");

  _data->programId = glCreateProgram();

  for(int i=0;i<3;i++)
  {
    if(_fileStrings[i].empty())
      continue;

    switch(i)
    {
    case 0:
      _data->shaderId[i] = glCreateShader(GL_VERTEX_SHADER);
      vvDebugMsg::msg(2, "glCreateShader(GL_VERTEX_SHADER)");
      break;
    case 1:
    {
      _data->shaderId[i] = glCreateShader(GL_GEOMETRY_SHADER_EXT);

      GLenum inputType = VV_TRIANGLES;
      switch (_geoShaderArgs.inputType)
      {
      case VV_POINTS:
        inputType = GL_POINTS;
        break;
      case VV_LINES:
        inputType = GL_LINES;
        break;
      case VV_LINES_ADJACENCY:
        inputType = GL_LINES_ADJACENCY_EXT;
        break;
      case VV_TRIANGLES:
        inputType = GL_TRIANGLES;
        break;
      case VV_TRIANGLES_ADJACENCY:
        inputType = GL_TRIANGLES_ADJACENCY_EXT;
        break;
      default:
        vvDebugMsg::msg(0, "Invalid input type for geometry shader. Supported: GL_POINTS, GL_LINES, GL_LINES_ADJACENCY_EXT, GL_TRIANGLES, GL_TRIANGLES_ADJACENCY_EXT");
        break;
      }

      GLenum outputType = VV_TRIANGLE_STRIP;
      switch (_geoShaderArgs.outputType)
      {
      case VV_POINTS:
        outputType = GL_POINTS;
        break;
      case VV_LINE_STRIP:
        outputType = GL_LINE_STRIP;
        break;
      case VV_TRIANGLE_STRIP:
        outputType = GL_TRIANGLE_STRIP;
        break;
      default:
        vvDebugMsg::msg(0, "Invalid output type for geometry shader. Supported: GL_POINTS, GL_LINE_STRIP, GL_TRIANGLE_STRIP");
      }

      glProgramParameteriEXT(_data->programId, GL_GEOMETRY_INPUT_TYPE_EXT, inputType);
      glProgramParameteriEXT(_data->programId, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputType);

      int maxVertices = 0;
      glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxVertices);

      // negative value indicates that user desires the maximum
      if (_geoShaderArgs.numOutputVertices < 0)
      {
        _geoShaderArgs.numOutputVertices = maxVertices;
      }

      if (_geoShaderArgs.numOutputVertices <= maxVertices)
      {
        glProgramParameteriEXT(_data->programId, GL_GEOMETRY_VERTICES_OUT_EXT, _geoShaderArgs.numOutputVertices);
      }
      else
      {
        vvDebugMsg::msg(0, "Invalid number of output vertices for geometry shader. Supported: ", maxVertices);
      }

      vvDebugMsg::msg(2, "glCreateShader(GL_GEOMETRY_SHADER_EXT)");
      break;
    }
    case 2:
      _data->shaderId[i] = glCreateShader(GL_FRAGMENT_SHADER);
      vvDebugMsg::msg(2, "glCreateShader(GL_FRAGMENT_SHADER)");
      break;
    }

    GLint size = (GLint)_fileStrings[i].size();
    const char* code = _fileStrings[i].c_str();
    glShaderSource(_data->shaderId[i], 1, (const GLchar**)&code, &size);
    glCompileShader(_data->shaderId[i]);

    GLint compiled;
    glGetShaderiv(_data->shaderId[i], GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
      GLint length;
      std::vector<GLchar> compileLog;
      glGetShaderiv(_data->shaderId[i], GL_INFO_LOG_LENGTH, &length);
      if (length < 0)
      {
        VV_LOG(0) << "glCompileShader failed, cannot obtain log" << std::endl;
        return false;
      }
      compileLog.resize(static_cast<size_t>(length));
      glGetShaderInfoLog(_data->shaderId[i], length, &length, &compileLog[0]);
      vvDebugMsg::msg(0, "glCompileShader failed: " , &compileLog[0]);
      return false;
    }

    glAttachShader(_data->programId, _data->shaderId[i]);
  }

  glLinkProgram(_data->programId);

  GLint linked;
  glGetProgramiv(_data->programId, GL_LINK_STATUS, &linked);
  if (!linked)
  {
    GLint length;
    std::vector<GLchar> linkLog;
    glGetProgramiv(_data->programId, GL_INFO_LOG_LENGTH, &length);
    linkLog.resize(length);
    glGetProgramInfoLog(_data->programId, length, &length, &linkLog[0]);
    vvDebugMsg::msg(0, "glLinkProgram failed: ", &linkLog[0]);
    return false;
  }

  _shadersLoaded = true;

  vvGLTools::printGLError("Leaving vvGLSLProgram::loadShaders()");

  return true;
}

void vvGLSLProgram::enable()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::enableProgram()");

  if(_shadersLoaded)
  {
    GLint activeTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
    for(TextureMap::iterator i = _data->textureNameMaps.begin(); i != _data->textureNameMaps.end(); ++i)
    {
      Texture* tex = i->second;
      if(tex->_unit == -1)
      {
        vvDebugMsg::msg(2, "texture unit invalid for ", i->first.c_str());
        continue;
      }
      if(tex->_id == -1)
      {
        vvDebugMsg::msg(2, "texture name invalid for ", i->first.c_str());
        continue;
      }

      glActiveTexture(GL_TEXTURE0+tex->_unit);
      switch(tex->_type)
      {
      case TEXTURE_1D:
        glBindTexture(GL_TEXTURE_1D, tex->_id);
        break;
      case TEXTURE_2D:
        glBindTexture(GL_TEXTURE_2D, tex->_id);
        break;
      case TEXTURE_3D:
        glBindTexture(GL_TEXTURE_3D, tex->_id);
        break;
      default:
        assert("invalid texture type" == NULL);
        break;
      }
    }
    glUseProgramObjectARB(_data->programId);
    glActiveTexture(activeTexture);
  }
  else
  {
    cerr << "vvGLSLProgram::enableProgram() Can't enable Programm: shaders not successfully loaded!" << endl;
  }

  vvGLTools::printGLError("Leaving vvGLSLProgram::enableProgram()");
}

void vvGLSLProgram::disable()
{
  vvGLTools::printGLError("Enter vvGLSLProgram::disableProgram()");

  glUseProgramObjectARB(0);

  vvGLTools::printGLError("Leaving vvGLSLProgram::disableProgram()");
}

void vvGLSLProgram::setParameter1f(const string& parameterName, const float& f1)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter1f");
  if(uniform != -1)
    glUniform1f(uniform, f1);
}

void vvGLSLProgram::setParameter1i(const string& parameterName, const int& i1)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter1i");
  if(uniform != -1)
    glUniform1i(uniform, i1);
}

void vvGLSLProgram::setParameter3f(const string& parameterName, const float* array)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter3f");
  if(uniform != -1)
    glUniform3fv(uniform, 1, array);
}

void vvGLSLProgram::setParameter3f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter3f");
  if(uniform != -1)
    glUniform3f(uniform, f1, f2, f3);
}

void vvGLSLProgram::setParameter4f(const string& parameterName, const float* array)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter4f");
  if(uniform != -1)
    glUniform4fv(uniform, 1, array);
}

void vvGLSLProgram::setParameter4f(const string& parameterName,
                            const float& f1, const float& f2, const float& f3, const float& f4)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameter4f");
  if(uniform != -1)
    glUniform4f(uniform, f1, f2, f3, f4);
}

void vvGLSLProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameterArray1i");
  if(uniform != -1)
    glUniform1iv(uniform, count, array);
}

void vvGLSLProgram::setParameterArray1f(const string& parameterName, const float* array, int count)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameterArray1f");
  if(uniform != -1)
    glUniform1fv(uniform, count, array);
}

void vvGLSLProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameterArray3f");
  if(uniform != -1)
    glUniform3fv(uniform, count, array);
}

void vvGLSLProgram::setParameterMatrix4f(const string& parameterName, const float* mat)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameterMatrix4f");
  if(uniform != -1)
    glUniformMatrix4fv(uniform, 1, GL_FALSE, mat);
}

void vvGLSLProgram::setParameterMatrix4f(const string& parameterName, virvo::mat4 const& mat)
{
  const GLint uniform = getUniform(_data, parameterName, "setParameterMatrix4f");
  if(uniform != -1)
    glUniformMatrix4fv(uniform, 1, GL_FALSE, mat.data());
}

void vvGLSLProgram::setParameterTex1D(const string& parameterName, const unsigned int& ui)
{
  Texture* tex = getTexture(_data, parameterName, "setParameterTex1D");
  if(tex->_uniform != -1)
  {
    tex->_type = TEXTURE_1D;
    tex->_id = ui;
    glUniform1i(tex->_uniform, tex->_unit);
    GLint activeTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glBindTexture(GL_TEXTURE_1D, ui);
    glActiveTexture(activeTexture);
  }
}

void vvGLSLProgram::setParameterTex2D(const string& parameterName, const unsigned int& ui)
{
  Texture* tex = getTexture(_data, parameterName, "setParameterTex2D");
  if(tex->_uniform != -1)
  {
    tex->_type = TEXTURE_2D;
    tex->_id = ui;
    glUniform1i(tex->_uniform, tex->_unit);
    GLint activeTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glBindTexture(GL_TEXTURE_2D, ui);
    glActiveTexture(activeTexture);
  }
}

void vvGLSLProgram::setParameterTex3D(const string& parameterName, const unsigned int& ui)
{
  Texture* tex = getTexture(_data, parameterName, "setParameterTex3D");
  if(tex->_uniform != -1)
  {
    tex->_type = TEXTURE_3D;
    tex->_id = ui;
    glUniform1i(tex->_uniform, tex->_unit);
    GLint activeTexture = GL_TEXTURE0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
    glActiveTexture(GL_TEXTURE0+tex->_unit);
    glBindTexture(GL_TEXTURE_3D, ui);
    glActiveTexture(activeTexture);
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
