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

#ifndef VV_TEXREND_H
#define VV_TEXREND_H

#include <vector>

#include <boost/scoped_ptr.hpp>

// Virvo:
#include "vvexport.h"
#include "vvrenderer.h"
#include "vvopengl.h"

class vvShaderFactory;
class vvShaderProgram;
class vvVolDesc;

//============================================================================
// Class Definitions
//============================================================================

/** Volume rendering engine using a texture-based algorithm.
  Textures can be drawn as planes or spheres. In planes mode a rendering
  quality can be given (determining the number of texture slices used), and
  the texture normal can be set according to the application's needs.<P>
  The data points are located at the grid as follows:<BR>
  The outermost data points reside at the very edge of the drawn object,
  the other values are evenly distributed inbetween.
  Make sure you define HAVE_CG in your compiler if you want to use Nvidia Cg.
  @author Juergen Schulze (schulze@cs.brown.de)
  @author Martin Aumueller
  @author Stefan Zellmann
  @see vvRenderer
*/

class VIRVOEXPORT vvTexRend : public vvRenderer
{
  public:
    enum ErrorType                                /// Error Codes
    {
      OK = 0,                                     ///< no error
      TRAM_ERROR,                                 ///< not enough texture memory
      TEX_SIZE_UNKNOWN,                           ///< size of 3D texture is unknown
      NO3DTEX,                                    ///< 3D textures not supported on this hardware
      UNSUPPORTED                                 ///< general error code
    };
    enum FeatureType                              /// Rendering features
    {
      VV_POST_CLASSIFICATION,                     ///< post-classification, needs shaders
      VV_MIP                                      ///< maximum intensity projection
    };
    enum SliceOrientation                         /// Slice orientation for planar 3D textures
    {
      VV_VARIABLE = 0,                            ///< choose automatically
      VV_VIEWPLANE,                               ///< parallel to view plane
      VV_CLIPPLANE,                               ///< parallel to clip plane
      VV_VIEWDIR,                                 ///< perpendicular to viewing direction
      VV_OBJECTDIR,                               ///< perpendicular to line eye-object
      VV_ORTHO                                    ///< as in orthographic projection
    };

    static const int NUM_PIXEL_SHADERS;           ///< number of pixel shaders used
  private:
    std::vector<std::vector<float> > rgbaTF;      ///< density to RGBA conversion table, as created by TF [0..1]
    std::vector<std::vector<uint8_t> > rgbaLUT;   ///< final RGBA conversion table, as transferred to graphics hardware (includes opacity and gamma correction)
    float  lutDistance;                           ///< slice distance for which LUT was computed
    virvo::vector< 3, size_t >   texels;          ///< width, height and depth of volume, including empty space [texels]
    float texMin[3];                              ///< minimum texture value of object [0..1] (to prevent border interpolation)
    float texMax[3];                              ///< maximum texture value of object [0..1] (to prevent border interpolation)
    size_t   textures;                            ///< number of textures stored in TRAM
    size_t   texelsize;                           ///< number of bytes/voxel transferred to OpenGL (depending on rendering mode)
    GLint internalTexFormat;                      ///< internal texture format (parameter for glTexImage...)
    GLenum texFormat;                             ///< texture format (parameter for glTexImage...)
    GLuint* texNames;                             ///< names of texture slices stored in TRAM
    std::vector<GLuint> pixLUTName;               ///< names for transfer function textures
    bool extTex3d;                                ///< true = 3D texturing supported
    bool extNonPower2;                            ///< true = NonPowerOf2 textures supported
    bool extMinMax;                               ///< true = maximum/minimum intensity projections supported
    bool extPixShd;                               ///< true = Nvidia pixel shader support (requires GeForce FX)
    bool extBlendEquation;                        ///< true = support for blend equation extension
    bool arbMltTex;                               ///< true = ARB multitexture support
    bool usePreIntegration;                       ///< true = pre-integrated rendering is actually used
    int maxClipPlanes;                            ///< maximum number of OpenGL clip planes
    ptrdiff_t minSlice, maxSlice;                 ///< min/maximum slice to render [0..numSlices-1], -1 for no slice constraints
    SliceOrientation _sliceOrientation;           ///< slice orientation for planar 3d textures
    size_t _lastFrame;                            ///< last frame rendered

    boost::scoped_ptr<vvShaderFactory> _shaderFactory; ///< Factory for shader-creation
    boost::scoped_ptr<vvShaderProgram> _shader;   ///< shader performing intersection test on gpu

    virvo::vec3 _eye;                             ///< the current eye position

    void setupClassification();
    ErrorType makeTextures(bool newTex);

    void enableShader (vvShaderProgram* shader) const;
    void disableShader(vvShaderProgram* shader) const;

    vvShaderProgram* initShader();

    void removeTextures();
    ErrorType updateTextures3D(ssize_t, ssize_t, ssize_t, ssize_t, ssize_t, ssize_t, bool);
    void setGLenvironment() const;
    void unsetGLenvironment() const;
    void renderTex3DPlanar(virvo::mat4 const& mv);
    void updateLUT(float dist);
    size_t getLUTSize(virvo::vector< 3, size_t >& size) const;
    size_t getPreintTableSize() const;
    void enableTexture(GLenum target) const;
    void disableTexture(GLenum target) const;
    void initLight(vvShaderProgram* pixelShader, virvo::mat4 const& mv, virvo::vec3 const& normal, float sliceThickness);

    int  getCurrentShader() const;
    void setCurrentShader(int);
    size_t getTextureSize(size_t sz) const;
  public:
    vvTexRend(vvVolDesc*, vvRenderState);
    virtual ~vvTexRend();
    void  renderVolumeGL();
    void  updateTransferFunction();
    void  updateVolumeData();
    void  updateVolumeData(size_t, size_t, size_t, size_t, size_t, size_t);
    void  activateClippingPlanes();
    void  deactivateClippingPlanes();
    void  setNumLights(int);
    bool  instantClassification() const;
    void  setViewingDirection(virvo::vec3f const& vd);
    void  setObjectDirection(virvo::vec3f const& od);
    bool checkParameter(ParameterType param, vvParam const& value) const;
    virtual void setParameter(ParameterType param, const vvParam& value);
    virtual vvParam getParameter(ParameterType param) const;
    static bool isSupported(FeatureType);
    void renderQualityDisplay() const;
    void printLUT(size_t chan=0) const;
    void setTexMemorySize(size_t);
    size_t getTexMemorySize() const;
    uint8_t* getHeightFieldData(float[4][3], size_t&, size_t&);
    float getManhattenDist(float[3], float[3]) const;
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
