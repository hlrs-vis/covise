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

#ifndef VV_GLSLPROGRAM_H
#define VV_GLSLPROGRAM_H

#include "math/forward.h"
#include "vvshaderprogram.h"
#include "vvopengl.h"

#include <vector>
#include <map>

struct vvGLSLData;

/** Wrapper Class for OpenGL Shading Language
  This class loads a combination of up to three shaders
  (vertex-, geometry- and fragment-shaders) and
  manages all interaction with it.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvGLSLProgram : public vvShaderProgram
{
public:
  /** Creates a vvGLSLProgram and tries to attach the given shaders source code.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
   */
  vvGLSLProgram(const std::string& vert, const std::string& geom, const std::string& frag);

  /** Creates a vvGLSLProgram and tries to attach the given shaders source code.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
    @param geoShaderArgs parameters for the geometry shader
   */
  vvGLSLProgram(const std::string& vert, const std::string& geom, const std::string& frag,
                const vvShaderProgram::GeoShaderArgs& geoShaderArgs);

  /// Deactivates and deletes shader program that was generated in this class
  ~vvGLSLProgram();

  void enable();   ///< enables program with loaded shaders
  void disable();  ///< disables program with its shaders

  /**
    Set uniform parameter functions. Use parameters' names only.
    Parameters' ids are checked and connected between programs automatically.
   */
  void setParameter1f(const std::string& parameterName, const float& f1);
  void setParameter1i(const std::string& parameterName, const int& i1);

  void setParameter3f(const std::string& parameterName, const float* array);
  void setParameter3f(const std::string& parameterName,
                              const float& f1, const float& f2, const float& f3);

  void setParameter4f(const std::string& parameterName, const float* array);
  void setParameter4f(const std::string& parameterName,
                              const float& f1, const float& f2, const float& f3, const float& f4);

  void setParameterArray1i(const std::string& parameterName, const int* array, const int& count);

  void setParameterArray1f(const std::string& parameterName, const float* array, int count);
  void setParameterArray3f(const std::string& parameterName, const float* array, const int& count);

  void setParameterMatrix4f(const std::string& parameterName, const float* mat);
  void setParameterMatrix4f(const std::string& parameterName, virvo::mat4 const& mat);

  void setParameterTex1D(const std::string& parameterName, const unsigned int& ui);
  void setParameterTex2D(const std::string& parameterName, const unsigned int& ui);
  void setParameterTex3D(const std::string& parameterName, const unsigned int& ui);

  void disableTexture1D(const std::string& parameterName = NULL);
  void disableTexture2D(const std::string& parameterName = NULL);
  void disableTexture3D(const std::string& parameterName = NULL);

private:
  vvGLSLData* _data;

  bool loadShaders();     ///< Initializes, compiles, and links a shader program
  void deleteProgram();   ///< deletes program with all shaders and frees memory
};
#endif // VV_GLSLPROGRAM_H

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
