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

#ifndef VV_CGPROGRAM_H
#define VV_CGPROGRAM_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CG

#include "vvexport.h"
#include "vvshaderprogram.h"

#include <map>

/** Wrapper Class for Cg Shading Language
  This class loads a combination of up to three shaders
  (vertex-, geometry- and fragment-shaders) and
  manages all interaction with it.

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvCgProgram : public vvShaderProgram
{
public:
  vvCgProgram();  ///< trivial constructor

  /**
    Creates a vvCgProgram and tries to attach the given shaders source codes.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
   */
  vvCgProgram(const std::string& vert, const std::string& geom, const std::string& frag);

  /**
    Creates a vvCgProgram and tries to attach the given shaders source codes.
    @param vert Filestring of desired vertex-shader or emtpy/NULL.
    @param geom Filestring of desired geometry-shader or emtpy/NULL.
    @param frag Filestring of desired fragment-shader or emtpy/NULL.
   */
  vvCgProgram(const std::string& vert, const std::string& geom, const std::string& frag,
              const vvShaderProgram::GeoShaderArgs& geoShaderArgs);

  /// Deactivates and deletes shader program that was generated in this class
  ~vvCgProgram();

  void enable();
  void disable();

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

private:
  bool loadShaders();

  // CG data
  struct CGdata;
  CGdata *_data;
};

#endif // HAVE_CG

#endif // VV_CGPGROGRAM_H

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
