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

#include "vvcgprogram.h"

#ifdef HAVE_CG

#include "vvopengl.h"
#include "vvdebugmsg.h"

#include "private/vvgltools.h"

#include <iostream>

using std::cerr;
using std::endl;
using std::string;

#include <Cg/cg.h>
#include <Cg/cgGL.h>

typedef std::map<std::string, CGparameter> ParameterMap;
typedef ParameterMap::iterator ParameterIterator;

struct vvCgProgram::CGdata
{
  ParameterIterator initParameter(const string& parameterName)
  {
    // check if already initialized
    ParameterIterator paraIterator = cgParameterNameMaps.find(parameterName);
    if(paraIterator != cgParameterNameMaps.end())
      return paraIterator;

    CGparameter paraFirst = 0;
    for(int i=0;i<3;i++)
    {
      if(shaderId[i]==0)
        continue;

      CGparameter param = cgGetNamedParameter(shaderId[i], parameterName.c_str());

      if(param != NULL && paraFirst == 0)
      {
        paraFirst = param;
      }
      else if(param != NULL && paraFirst != 0)
      {
        cgConnectParameter(paraFirst, param);
      }
    }

    cgParameterNameMaps[parameterName] = paraFirst;

    if(paraFirst == 0)
    {
      string errmsg = "cgParameter (" + parameterName + ")not found!";
      vvDebugMsg::msg(2, errmsg.c_str());
      return cgParameterNameMaps.end();
    }

    return cgParameterNameMaps.find(parameterName);
  }

  CGcontext program;
  CGprofile profile[3];
  CGprogram shaderId[3];

  ParameterMap cgParameterNameMaps;
};

namespace
{
  void cgErrorHandler(CGcontext context, CGerror error, void*)
  {
    if(error != CG_NO_ERROR)
      cerr << cgGetErrorString(error) << " (" << static_cast<int>(error) << ")" << endl;
    for(GLenum glerr = glGetError(); glerr != GL_NO_ERROR; glerr = glGetError())
    {
      cerr << "GL error: " << gluErrorString(glerr) << endl;
    }
    if(context && error==CG_COMPILER_ERROR)
    {
       if(const char *listing = cgGetLastListing(context))
       {
          cerr << "last listing:" << endl;
          cerr << listing << endl;
       }
    }
  }

  CGGLenum toCgEnum(const int i)
  {
    CGGLenum result;
    switch(i)
    {
    case 0:
      result = CG_GL_VERTEX;
      break;
  #if CG_VERSION_NUM >= 2000
    case 1:
      result = CG_GL_GEOMETRY;
      break;
  #endif
    case 2:
      result = CG_GL_FRAGMENT;
      break;
    default:
      vvDebugMsg::msg(0, "toCgEnum() unknown ShaderType!");
      result = CG_GL_FRAGMENT;
      break;
    }
    return result;
  }
}

vvCgProgram::vvCgProgram(const string& vert, const string& geom, const string& frag)
: vvShaderProgram(vert, geom, frag)
{
  _data = new CGdata();

  for(int i=0; i<3;i++)
  {
    _data->shaderId[i] = 0;
    _data->profile[i] = CGprofile(0);
  }

  _shadersLoaded = loadShaders();
  if(_shadersLoaded)
  {
     vvDebugMsg::msg(0, "vvCgProgram::vvCgProgram() Loading Shaders failed!");
  }
}

vvCgProgram::vvCgProgram(const string& vert, const string& geom, const string& frag,
                         const vvShaderProgram::GeoShaderArgs& geoShaderArgs)
: vvShaderProgram(vert, geom, frag, geoShaderArgs)
{
  _data = new CGdata();

  for(int i=0; i<3;i++)
  {
    _data->shaderId[i] = 0;
    _data->profile[i] = CGprofile(0);
  }

  _shadersLoaded = loadShaders();
  if(_shadersLoaded)
  {
     vvDebugMsg::msg(0, "vvCgProgram::vvCgProgram() Loading Shaders failed!");
  }
}

vvCgProgram::~vvCgProgram()
{
  disable();
  if (_data->program)
  {
    cgDestroyContext(_data->program);
  }
  delete _data;
}

bool vvCgProgram::loadShaders()
{
  cgSetErrorHandler(cgErrorHandler, NULL);
  _data->program = cgCreateContext();

  if (_data->program == NULL)
  {
    vvDebugMsg::msg(0, "Can't create Cg context");
  }

  for(int i=0;i<3;i++)
  {
    if(_fileStrings[i].length() == 0)
      continue;

    _data->profile[i] = cgGLGetLatestProfile(toCgEnum(i));
    cgGLSetOptimalOptions(_data->profile[i]);
    _data->shaderId[i] = cgCreateProgram( _data->program, CG_SOURCE, _fileStrings[i].c_str(), _data->profile[i], NULL, NULL);

    if (_data->shaderId[i] == NULL)
    {
      vvDebugMsg::msg(0, "Couldn't load cg-shader!");
      return false;
    }
  }
  return true;
}

void vvCgProgram::enable()
{
  for(int i=0;i<3;i++)
  {
    if(_data->shaderId[i] == 0)
      continue;

    cgGLLoadProgram(_data->shaderId[i]);
    cgGLEnableProfile(_data->profile[i]);
    cgGLBindProgram(_data->shaderId[i]);
  }
}

void vvCgProgram::disable()
{
  for(int i=0;i<3;i++)
  {
    if(_data->profile[i] == 0)
      continue;

    cgGLDisableProfile(_data->profile[i]);
  }
}

void vvCgProgram::setParameter1f(const string& parameterName, const float& f1)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter1f(it->second, f1);
}

void vvCgProgram::setParameter1i(const string& parameterName, const int& i1)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter1i(it->second, i1);
}

void vvCgProgram::setParameter3f(const string& parameterName, const float* array)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter3fv(it->second, array);
}

void vvCgProgram::setParameter3f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter3f(it->second, f1, f2, f3);
}

void vvCgProgram::setParameter4f(const string& parameterName, const float* array)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter4fv(it->second, array);
}

void vvCgProgram::setParameter4f(const string& parameterName,
                          const float& f1, const float& f2, const float& f3, const float& f4)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetParameter4f(it->second, f1, f2, f3, f4);
}

void vvCgProgram::setParameterArray1i(const string& parameterName, const int* array, const int& count)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
  {
    // transform integers to floats because CG doesn't support uniform integers
    float* floats(new float[count]);
    for(int i=0;i<count;i++)
      floats[i] = float(array[i]);
    cgGLSetParameterArray1f(it->second, 0, count, floats);
    delete [] floats;
  }
}

void vvCgProgram::setParameterArray1f(const string& parameterName, const float* array, int count)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgGLSetParameterArray1f(it->second, 0, count, array);
}

void vvCgProgram::setParameterArray3f(const string& parameterName, const float* array, const int& count)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgGLSetParameterArray3f(it->second, 0, 3*count, array);
}

void vvCgProgram::setParameterMatrix4f(const string& parameterName, const float* mat)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
    cgSetMatrixParameterfr(it->second, mat);
}

void vvCgProgram::setParameterMatrix4f(const string& parameterName, virvo::mat4 const& mat)
{
  setParameterMatrix4f(parameterName, mat.data());
}

void vvCgProgram::setParameterTex1D(const string& parameterName, const unsigned int& ui)
{
  ParameterIterator it = _data->initParameter(parameterName);
  if(it->second != 0)
  {
    cgGLSetTextureParameter(it->second, ui);
    cgGLEnableTextureParameter(it->second);
  }
}

void vvCgProgram::setParameterTex2D(const string& parameterName, const unsigned int& ui)
{
  setParameterTex1D(parameterName, ui);
}

void vvCgProgram::setParameterTex3D(const string& parameterName, const unsigned int& ui)
{
  setParameterTex1D(parameterName, ui);
}

#endif // ifdef HAVE_CG

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
