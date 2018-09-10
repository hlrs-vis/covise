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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <GL/glew.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <cfloat>
#include <cstring>

#include "vvopengl.h"
#include "vvdynlib.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvdebugmsg.h"
#include "vvtoolshed.h"
#include "vvtexrend.h"
#include "vvtextureutil.h"
#include "vvprintgl.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvvoldesc.h"
#include "vvvecmath.h"

#include "gl/util.h"

#include "private/vvgltools.h"
#include "private/vvlog.h"

using namespace std;
using namespace virvo;

enum {
  Shader1Chan = 0,
  Shader2DTF = 8,
  ShaderPreInt = 11,
  ShaderLighting = 12,
  ShaderMultiTF = 13,
  ShaderRGB = 14,
  NumPixelShaders
};


//----------------------------------------------------------------------------
const int vvTexRend::NUM_PIXEL_SHADERS = NumPixelShaders;

static virvo::BufferPrecision mapBitsToBufferPrecision(int bits)
{
    switch (bits)
    {
    case 8:
        return virvo::Byte;
    case 16:
        return virvo::Short;
    case 32:
        return virvo::Float;
    default:
        assert(!"unknown bit size");
        return virvo::Byte;
    }
}

static PixelFormat mapBufferPrecisionToFormat(virvo::BufferPrecision bp)
{
    switch (bp)
    {
    case virvo::Byte:
        return virvo::PF_RGBA8;
    case virvo::Short:
        return virvo::PF_RGBA16F;
    case virvo::Float:
        return virvo::PF_RGBA32F;
    default:
        assert(!"unknown format");
        return virvo::PF_UNSPECIFIED;
    }
}

//----------------------------------------------------------------------------
/** Constructor.
  @param vd                      volume description
  @param renderState             object describing the render state
  @param geom                    render geometry (default: automatic)
  @param displayNames            names of x-displays (host:display.screen) for multi-gpu rendering
  @param numDisplays             # displays for multi-gpu rendering
  @param multiGpuBufferPrecision precision of the offscreen buffer used for multi-gpu rendering
*/
vvTexRend::vvTexRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvTexRend::vvTexRend()");

  glewInit();

  if (this->_useOffscreenBuffer)
    setRenderTarget( virvo::FramebufferObjectRT::create( mapBufferPrecisionToFormat(this->_imagePrecision), virvo::PF_DEPTH24_STENCIL8) );

  if (vvDebugMsg::isActive(2))
  {
#ifdef _WIN32
    cerr << "_WIN32 is defined" << endl;
#elif WIN32
    cerr << "_WIN32 is not defined, but should be if running under Windows" << endl;
#endif

#ifdef HAVE_CG
    cerr << "HAVE_CG is defined" << endl;
#else
    cerr << "Tip: define HAVE_CG for pixel shader support" << endl;
#endif

    cerr << "Compiler knows OpenGL versions: ";
#ifdef GL_VERSION_1_1
    cerr << "1.1";
#endif
#ifdef GL_VERSION_1_2
    cerr << ", 1.2";
#endif
#ifdef GL_VERSION_1_3
    cerr << ", 1.3";
#endif
#ifdef GL_VERSION_1_4
    cerr << ", 1.4";
#endif
#ifdef GL_VERSION_1_5
    cerr << ", 1.5";
#endif
#ifdef GL_VERSION_2_0
    cerr << ", 2.0";
#endif
#ifdef GL_VERSION_2_1
    cerr << ", 2.1";
#endif
#ifdef GL_VERSION_3_0
    cerr << ", 3.0";
#endif
#ifdef GL_VERSION_3_1
    cerr << ", 3.1";
#endif
#ifdef GL_VERSION_3_2
    cerr << ", 3.2";
#endif
#ifdef GL_VERSION_3_3
    cerr << ", 3.3";
#endif
#ifdef GL_VERSION_4_0
    cerr << ", 4.0";
#endif
#ifdef GL_VERSION_4_1
    cerr << ", 4.1";
#endif
#ifdef GL_VERSION_4_2
    cerr << ", 4.2";
#endif
    cerr << endl;
  }

  _shaderFactory.reset(new vvShaderFactory());
  _shaderFactory->loadFragmentLibrary("texrend");

  rendererType = TEXREND;
  texNames = NULL;
  _sliceOrientation = VV_VARIABLE;
  minSlice = maxSlice = -1;
  rgbaTF.resize(1);
  rgbaLUT.resize(1);
  usePreIntegration = false;
  textures = 0;

  setCurrentShader(_currentShader);

  _lastFrame = std::numeric_limits<size_t>::max();
  lutDistance = -1.0;
  _isROIChanged = true;

  // Find out which OpenGL extensions are supported:
  extTex3d  = vvGLTools::isGLextensionSupported("GL_EXT_texture3D") || vvGLTools::isGLVersionSupported(1,2,1);
  arbMltTex = vvGLTools::isGLextensionSupported("GL_ARB_multitexture") || vvGLTools::isGLVersionSupported(1,3,0);

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax") || vvGLTools::isGLVersionSupported(1,4,0);
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation") || vvGLTools::isGLVersionSupported(1,1,0);
  extPixShd = vvShaderFactory::isSupported("cg") || vvShaderFactory::isSupported("glsl");

  extNonPower2 = vvGLTools::isGLextensionSupported("GL_ARB_texture_non_power_of_two") || vvGLTools::isGLVersionSupported(2,0,0);

  // Store number of supported OpenGL clip planes
  glGetIntegerv(GL_MAX_CLIP_PLANES, &maxClipPlanes);

  _shader.reset(initShader());
  // Can only use post-classification if shaders
  // are supported and can be loaded/were found
  if(!extPixShd || !_shader)
    setParameter(vvRenderer::VV_POST_CLASSIFICATION, false);

  setupClassification();

  if (_postClassification)
  {
    pixLUTName.resize(vd->tf.size());
    glGenTextures(pixLUTName.size(), &pixLUTName[0]);
  }

  cerr << "Rendering algorithm: ";
  if (_postClassification)
    cerr << "VV_PIX_SHD, vv_shader" << std::setw(2) << std::setfill('0') << (_currentShader+1);
  else
    cerr << "VV_RGBA";
  cerr << endl;

  textures = 0;

  if (_postClassification)
  {
    makeTextures(true);      // we only have to do this once for non-RGBA textures
  }
  updateTransferFunction();
  _shader.reset(initShader());
}

//----------------------------------------------------------------------------
/// Destructor
vvTexRend::~vvTexRend()
{
  vvDebugMsg::msg(1, "vvTexRend::~vvTexRend()");

  if (_postClassification)
  {
    glDeleteTextures(pixLUTName.size(), &pixLUTName[0]);
  }
  removeTextures();
}


//------------------------------------------------
/** Initialize texture parameters for a voxel type
  @param vt voxeltype
*/
void vvTexRend::setupClassification()
{
  if (_postClassification)
  {
    if (vd->getChan() == 1)
    {
      texelsize=1;
      internalTexFormat = GL_LUMINANCE;
      texFormat = GL_LUMINANCE;
    }
    else if (vd->getChan() == 2)
    {
      texelsize=2;
      internalTexFormat = GL_LUMINANCE_ALPHA;
      texFormat = GL_LUMINANCE_ALPHA;
    }
    else if (vd->getChan() == 3)
    {
      texelsize=3;
      internalTexFormat = GL_RGB;
      texFormat = GL_RGB;
    }
    else
    {
      texelsize=4;
      internalTexFormat = GL_RGBA;
      texFormat = GL_RGBA;
    }
  }
  else
  {
    internalTexFormat = GL_RGBA;
    texFormat = GL_RGBA;
    texelsize=4;
  }
}

//----------------------------------------------------------------------------
/// Remove all textures from texture memory.
void vvTexRend::removeTextures()
{
  vvDebugMsg::msg(1, "vvTexRend::removeTextures()");

  if (textures > 0)
  {
    glDeleteTextures(textures, texNames);
    delete[] texNames;
    texNames = NULL;
    textures = 0;
  }
}

//----------------------------------------------------------------------------
/// Generate textures for all rendering modes.
vvTexRend::ErrorType vvTexRend::makeTextures(bool newTex)
{
  ErrorType err = OK;

  vvDebugMsg::msg(2, "vvTexRend::makeTextures()");

  virvo::vector< 3, ssize_t > vox = _paddingRegion.max - _paddingRegion.min;
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] == 0 || vox[1] == 0 || vox[2] == 0)
    return err;

  // Compute texture dimensions (perhaps must be power of 2):
  texels[0] = getTextureSize(vox[0]);
  texels[1] = getTextureSize(vox[1]);
  texels[2] = getTextureSize(vox[2]);

  updateTextures3D(0, 0, 0, texels[0], texels[1], texels[2], newTex);
  vvGLTools::printGLError("vvTexRend::makeTextures");

  if (_postClassification)
  {
    updateTransferFunction();
    updateLUT(1.f);
  }
  return err;
}

void vvTexRend::setTexMemorySize(size_t newSize)
{
  if (_texMemorySize == newSize)
    return;

  _texMemorySize = newSize;
}

size_t vvTexRend::getTexMemorySize() const
{
  return _texMemorySize;
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvTexRend::updateTransferFunction()
{
  virvo::vector< 3, size_t > size;

  vvDebugMsg::msg(1, "vvTexRend::updateTransferFunction()");
  if (_postClassification)
  {
     if (_preIntegration && arbMltTex && !(getParameter(VV_CLIP_MODE) && (_clipSingleSlice || _clipOpaque)))
       usePreIntegration = true;
     else
       usePreIntegration = false;
  }
  else
  {
     if (_preIntegration &&
           arbMltTex && 
           !(getParameter(VV_CLIP_MODE) && (_clipSingleSlice || _clipOpaque)) &&
           (_postClassification && (_currentShader==Shader1Chan || _currentShader==ShaderPreInt)))
     {
        usePreIntegration = true;
        if(_currentShader==Shader1Chan)
           _currentShader = ShaderPreInt;
     }
     else
     {
        usePreIntegration = false;
        if(_currentShader==ShaderPreInt)
           _currentShader = Shader1Chan;
     }
  }

  // Generate arrays from pins:
  size_t total = getLUTSize(size);
  if (vd->tf.size() != rgbaTF.size())
    rgbaTF.resize(vd->tf.size());
  for (size_t i=0; i<vd->tf.size(); ++i) {
    if (rgbaTF[i].size() != total*4) // reserve space for TF as 4 floats/entry (RGBA)
       rgbaTF[i].resize(total*4);
    vd->computeTFTexture(i, size[0], size[1], size[2], &rgbaTF[i][0]);
  }

  if(!instantClassification())
    updateLUT(1.0f);                                // generate color/alpha lookup table
  else
    lutDistance = -1.;                              // invalidate LUT
}

//----------------------------------------------------------------------------
// see parent in vvRenderer
void vvTexRend::updateVolumeData()
{
  vvRenderer::updateVolumeData();

  makeTextures(true);
}

//----------------------------------------------------------------------------
void vvTexRend::updateVolumeData(size_t offsetX, size_t offsetY, size_t offsetZ,
                                 size_t sizeX, size_t sizeY, size_t sizeZ)
{
  updateTextures3D(offsetX, offsetY, offsetZ, sizeX, sizeY, sizeZ, false);
}

//----------------------------------------------------------------------------
/**
   Method to create a new 3D texture or update parts of an existing 3D texture.
   @param offsetX, offsetY, offsetZ: lower left corner of texture
   @param sizeX, sizeY, sizeZ: size of texture
   @param newTex: true: create a new texture
                  false: update an existing texture
*/
vvTexRend::ErrorType vvTexRend::updateTextures3D(ssize_t offsetX, ssize_t offsetY, ssize_t offsetZ,
                                                 ssize_t sizeX, ssize_t sizeY, ssize_t sizeZ, bool newTex)
{
  ErrorType err = OK;
  vvDebugMsg::msg(1, "vvTexRend::updateTextures3D()");

  if (!extTex3d) return NO3DTEX;

  size_t texSize = sizeX * sizeY * sizeZ * texelsize;
  VV_LOG(1) << "3D Texture width     = " << sizeX << std::endl;
  VV_LOG(1) << "3D Texture height    = " << sizeY << std::endl;
  VV_LOG(1) << "3D Texture depth     = " << sizeZ << std::endl;
  VV_LOG(1) << "3D Texture size (KB) = " << texSize / 1024 << std::endl;

  if (vd->frames != textures)
    newTex = true;

  if (newTex)
  {
    VV_LOG(2) << "Creating texture names. # of names: " << vd->frames << std::endl;

    removeTextures();
    textures  = vd->frames;
    delete[] texNames;
    texNames = new GLuint[textures];
    glGenTextures(vd->frames, texNames);
  }

  VV_LOG(2) << "Transferring textures to TRAM. Total size [KB]: " << vd->frames * texSize / 1024 << std::endl;

  vec3i first(offsetX, offsetY, offsetZ);
  vec3i last = first + vec3i(sizeX, sizeY, sizeZ);

  PixelFormat pf = PF_R8;

  // Texrend uses 8-bit per color component for rendering!
  if (!_postClassification) // pre-classification
    pf = PF_RGBA8;
  else if (vd->getChan() == 1)
    pf = PF_R8;
  else if (vd->getChan() == 2)
    pf = PF_RG8;
  else if (vd->getChan() == 3)
    pf = PF_RGB8;
  else if (vd->getChan() == 4)
    pf = PF_RGBA8;
  else
    cerr << "Cannot determine pixel format: unsupported number of channels." << endl;
    // TODO: out..

  // Generate sub texture contents:
  TextureUtil tu(vd);
  for (size_t f = 0; f < vd->frames; f++)
  {
    TextureUtil::Pointer texData = NULL;

    if (_postClassification)
    {
      texData = tu.getTexture(first,
          last,
          pf,
          TextureUtil::All,
          f);
    }
    else
    {
      // Compute RGBA texture with indirection from rgbaLUT
      texData = tu.getTexture(first,
          last,
          &(rgbaLUT[0])[0], // TODO: why rgbaLUT[0]?
          1/*bytes per RGBA channel*/,
          f);
    }

    if (newTex)
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glPixelStorei(GL_UNPACK_ALIGNMENT,1);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
        texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);
      GLint glWidth;
      glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);

      if (glWidth==texels[0])
      {
        glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
          texFormat, GL_UNSIGNED_BYTE, &texData[0]);
      }
      else
      {
        vvGLTools::printGLError("Tried to accommodate 3D textures");

        cerr << "Insufficient texture memory for 3D texture(s)." << endl;
        err = TRAM_ERROR;
      }
    }
    else
    {
      glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
      glTexSubImage3D(GL_TEXTURE_3D_EXT, 0, offsetX, offsetY, offsetZ,
        sizeX, sizeY, sizeZ, texFormat, GL_UNSIGNED_BYTE, &texData[0]);
    }
  }

  return err;
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexRend::setGLenvironment() const
{
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment()");

  // Save current GL state:
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);

  if (glBlendFuncSeparate)
  {
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }
  else
  {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }

  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glDepthMask(GL_FALSE);

  switch (_mipMode)
  {
    // alpha compositing
    case 0: glBlendEquation(GL_FUNC_ADD); break;
    case 1: glBlendEquation(GL_MAX); break;   // maximum intensity projection
    case 2: glBlendEquation(GL_MIN); break;   // minimum intensity projection
    default: break;
  }
  vvDebugMsg::msg(3, "vvTexRend::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvTexRend::unsetGLenvironment() const
{
  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment()");

  glPopAttrib();

  vvDebugMsg::msg(3, "vvTexRend::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
/** Render a volume entirely if probeSize=0 or a cubic sub-volume of size probeSize.
  @param mv        model-view matrix
*/
#define USE_ARRAYS
void vvTexRend::renderTex3DPlanar(mat4 const& mv)
{
  vec3 vissize, vissize2;                         // full and half object visible sizes
  vvVector3 isect[6];                             // intersection points, maximum of 6 allowed when intersecting a plane and a volume [object space]
  vec3 farthest;                                  // volume vertex farthest from the viewer
  vec3 delta;                                     // distance vector between textures [object space]
  vec3 normal;                                    // normal vector of textures
  vec3 origin;                                    // origin (0|0|0) transformed to object space
  vec3 normClipPoint;                             // normalized point on clipping plane
  vec3 clipPosObj;                                // clipping plane position in object space w/o position
  vec3 probePosObj;                               // probe midpoint [object space]
  vec3 probeSizeObj;                              // probe size [object space]
  vec3 probeTexels;                               // number of texels in each probe dimension
  vec3 probeMin, probeMax;                        // probe min and max coordinates [object space]
  vec3 texSize;                                   // size of 3D texture [object space]
  float     maxDist;                              // maximum length of texture drawing path
  size_t    numSlices;

  vvDebugMsg::msg(3, "vvTexRend::renderTex3DPlanar()");

  if (!extTex3d) return;                          // needs 3D texturing extension

  // determine visible size and half object size as shortcut
  virvo::vector< 3, ssize_t > minVox = _visibleRegion.min;
  virvo::vector< 3, ssize_t > maxVox = _visibleRegion.max;
  for (size_t i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], ssize_t(0));
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  vec3 minCorner = vd->objectCoords(minVox);
  vec3 maxCorner = vd->objectCoords(maxVox);
  vissize = maxCorner - minCorner;
  vec3 center = aabb(minCorner, maxCorner).center();

  for (size_t i=0; i<3; ++i)
  {
    texSize[i] = vissize[i] * (float)texels[i] / (float)vd->vox[i];
    vissize2[i]   = 0.5f * vissize[i];
  }
  vec3f pos = vd->pos + center;

  if (_isROIUsed)
  {
    vec3f size = vd->getSize();
    vec3f size2 = size * 0.5f;
    // Convert probe midpoint coordinates to object space w/o position:
    probePosObj = roi_pos_;
    probePosObj -= pos;                        // eliminate object position from probe position

    // Compute probe min/max coordinates in object space:
    probeMin = probePosObj - (roi_size_ * size) * 0.5f;
    probeMax = probePosObj + (roi_size_ * size) * 0.5f;

    // Constrain probe boundaries to volume data area:
    for (size_t i=0; i<3; ++i)
    {
      if (probeMin[i] > size2[i] || probeMax[i] < -size2[i])
      {
        vvDebugMsg::msg(3, "probe outside of volume");
        return;
      }
      if (probeMin[i] < -size2[i]) probeMin[i] = -size2[i];
      if (probeMax[i] >  size2[i]) probeMax[i] =  size2[i];
      probePosObj[i] = (probeMax[i] + probeMin[i]) *0.5f;
    }

    // Compute probe edge lengths:
    for (size_t i=0; i<3; ++i)
      probeSizeObj[i] = probeMax[i] - probeMin[i];
  }
  else                                            // probe mode off
  {
    probeSizeObj = vd->getSize();
    probeMin = minCorner;
    probeMax = maxCorner;
    probePosObj = center;
  }

  // Initialize texture counters
  if (_isROIUsed)
  {
    probeTexels = vec3f(0.0f, 0.0f, 0.0f);
    for (size_t i=0; i<3; ++i)
    {
      probeTexels[i] = texels[i] * probeSizeObj[i] / texSize[i];
    }
  }
  else                                            // probe mode off
  {
    probeTexels = vec3f( (float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2] );
  }

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);

  // Calculate inverted modelview matrix:
  mat4 invMV = inverse(mv);

  // Find eye position (object space):
  vec3f eye = getEyePosition();

  // Get projection matrix:
  vvMatrix pm = gl::getProjectionMatrix();
  bool isOrtho = pm.isProjOrtho();

  getObjNormal(normal, origin, eye, invMV, isOrtho);

  // compute number of slices to draw
  float depth = fabs(normal[0]*probeSizeObj[0]) + fabs(normal[1]*probeSizeObj[1]) + fabs(normal[2]*probeSizeObj[2]);
  size_t minDistanceInd = 0;
  if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=1;
  if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
    minDistanceInd=2;
  float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

  float sliceDistance = voxelDistance / _quality;
  if(_isROIUsed && _quality < 2.0)
  {
    // draw at least twice as many slices as there are samples in the probe depth.
    sliceDistance = voxelDistance * 0.5f;
  }
  numSlices = 2*(size_t)ceilf(depth/sliceDistance*.5f);

  if (numSlices < 1)                              // make sure that at least one slice is drawn
    numSlices = 1;
  // don't render an insane amount of slices
  {
    virvo::vector< 3, ssize_t > sz = maxVox - minVox;
    ssize_t maxV = std::max(sz[0], sz[1]);
    maxV = std::max(maxV, sz[2]);
    ssize_t lim = maxV * 10. * std::max(_quality, 1.f);
    if (numSlices > lim)
    {
      numSlices = lim;
      VV_LOG(1) << "Limiting number of slices to " << numSlices << std::endl;
    }
  }

  VV_LOG(3) << "Number of textures to render: " << numSlices << std::endl;

  float thickness = sliceDistance/voxelDistance;
  // Use alpha correction in indexed mode: adapt alpha values to number of textures:
  if (instantClassification())
  {

    // just tolerate slice distance differences imposed on us
    // by trying to keep the number of slices constant
    if(lutDistance/thickness < 0.88 || thickness/lutDistance < 0.88)
    {
      updateLUT(thickness);
    }
  }

  delta = normal;
  delta *= vec3f(sliceDistance);

  // Compute farthest point to draw texture at:
  farthest = delta;
  farthest *= vec3f((float)(numSlices - 1) * -0.5f);
  farthest += probePosObj; // will be vd->pos if no probe present

  if (getParameter(VV_CLIP_MODE))                     // clipping plane present?
  {
    // Adjust numSlices and set farthest point so that textures are only
    // drawn up to the clipPoint. (Delta may not be changed
    // due to the automatic opacity correction.)
    // First find point on clipping plane which is on a line perpendicular
    // to clipping plane and which traverses the origin:
    vec3 temp = delta * vec3(-0.5f);
    farthest += temp;                          // add a half delta to farthest
    clipPosObj = getParameter(VV_CLIP_PLANE_POINT);
    clipPosObj -= pos;
    temp = probePosObj;
    temp += normal;
    /* auto */ virvo::hit_record< ray, plane3 > hr = intersect
    (
        ray(probePosObj, probePosObj - temp),
        plane3(normal, clipPosObj)
    );
    if (hr.hit)
    {
        normClipPoint = hr.pos;
    }
    else
    {
        normClipPoint = vec3(0.0f, 0.0f, 0.0f);
    }
    maxDist = length( farthest - vec3(normClipPoint) );
    numSlices = (size_t)( maxDist / length(delta) ) + 1;
    temp = delta;
    temp *= vec3( ((float)(1 - static_cast<ptrdiff_t>(numSlices))) );
    farthest = normClipPoint;
    farthest += temp;
    if (_clipSingleSlice)
    {
      // Compute slice position:
      temp = delta;
      temp *= vec3f( ((float)(numSlices-1)) );
      farthest += temp;
      numSlices = 1;

      // Make slice opaque if possible:
      if (instantClassification())
      {
        updateLUT(0.0f);
      }
    }
  }

  vec3 texPoint;                                  // arbitrary point on current texture
  int drawn = 0;                                  // counter for drawn textures
  vec3 deltahalf = delta * 0.5f;

  // Relative viewing position
  vec3 releye = eye - pos;

  // Volume render a 3D texture:
  if(_postClassification && _shader)
  {
    enableShader(_shader.get());
    _shader->setParameterTex3D("pix3dtex", texNames[vd->getCurrentFrame()]);
    initLight(_shader.get(), mv, normal, thickness);
  }
  else
  {
    enableTexture(GL_TEXTURE_3D_EXT);
    glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);
  }

  // volume section mode:
  size_t start = minSlice==-1 ? 0 : minSlice;
  size_t stop = maxSlice==-1 ? numSlices : maxSlice;
  texPoint = farthest + (float)start*delta;

  GLint activeTexture = GL_TEXTURE0;                                                                                                                                                                                                                                                      
  glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE, &activeTexture);
  glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
  glEnableClientState(GL_VERTEX_ARRAY);
  glClientActiveTextureARB(GL_TEXTURE0_ARB);
  glEnableClientState(GL_TEXTURE_COORD_ARRAY);

  const size_t BatchSize = 100;
  std::vector<GLint> firsts;
  firsts.reserve(BatchSize);
  std::vector<GLsizei> counts;
  counts.reserve(BatchSize);
  std::vector<GLfloat> vc, tc0, tc1;

  vc.reserve(BatchSize*3*6);
  tc0.reserve(BatchSize*3*6);
  if (usePreIntegration)
  {
    glClientActiveTextureARB(GL_TEXTURE1_ARB);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTextureARB(GL_TEXTURE0_ARB);
    tc1.reserve(BatchSize*3*6);
  }

  glColor4f(1.0, 1.0, 1.0, 1.0);
  glNormal3f(normal[0], normal[1], normal[2]);
  size_t vertCount = 0;
  for (size_t i=start; i<stop; ++i)                     // loop thru all drawn textures
  {
    for (; i<stop && firsts.size()<BatchSize; ++i)
    {
      // Search for intersections between texture plane (defined by texPoint and
      // normal) and texture object (0..1):
      size_t isectCnt = isect->isectPlaneCuboid(normal, texPoint, probeMin, probeMax);

      texPoint += delta;

      if (isectCnt<3) continue;                     // at least 3 intersections needed for drawing

      counts.push_back(isectCnt);
      firsts.push_back(vertCount);
      vertCount += isectCnt;
      // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
      // and non-overlapping triangles:
      isect->cyclicSort(isectCnt, normal);
      for (size_t j=0; j<isectCnt; ++j)
      {
        for (int k=0; k<3; ++k)
          vc.push_back(isect[j][k]);
      }

      // Generate vertices in texture coordinates:
      if(usePreIntegration)
      {
        for (size_t j=0; j<isectCnt; ++j)
        {
          vec3 front, back;

          if(isOrtho)
          {
            back = vec3(isect[j]) - deltahalf;
          }
          else
          {
            vec3 v = vec3(isect[j]) - deltahalf;
            /* auto */ virvo::hit_record< ray, plane3 > hr = intersect
            (
                ray(releye, releye - vec3(isect[j])),
                plane3(normal, v)
            );
            back = hr.pos;
          }

          if(isOrtho)
          {
            front = vec3(isect[j]) + deltahalf;
          }
          else
          {
            vec3 v = vec3(isect[j]) + deltahalf;
            /* auto */ virvo::hit_record< ray, plane3 > hr = intersect
            (
                ray(releye, releye - vec3(isect[j])),
                plane3(normal, v)
            );
            front = hr.pos;
          }

            vec3 tex_coord_back
            (
                       (back[0] - minCorner[0]) / vissize[0],
                1.0f - (back[1] - minCorner[1]) / vissize[1],
                1.0f - (back[2] - minCorner[2]) / vissize[2]
            );
            std::copy( &tex_coord_back[0], &tex_coord_back[0] + 3, std::back_inserter(tc0) );

            vec3 tex_coord_front
            (
                       (front[0] - minCorner[0]) / vissize[0],
                1.0f - (front[1] - minCorner[1]) / vissize[1],
                1.0f - (front[2] - minCorner[2]) / vissize[2]
            );
            std::copy( &tex_coord_front[0], &tex_coord_front[0] + 3, std::back_inserter(tc1) );
        }
      }
      else
      {
        for (size_t j=0; j<isectCnt; ++j)
        {
            vec3 tex_coord
            (
                       (isect[j][0] - minCorner[0]) / vissize[0],
                1.0f - (isect[j][1] - minCorner[1]) / vissize[1],
                1.0f - (isect[j][2] - minCorner[2]) / vissize[2]
            );
            std::copy( &tex_coord[0], &tex_coord[0] + 3, std::back_inserter(tc0) );
        }
      }
    }
    --i; // last loop increased i by 1 too much

    if (vertCount != 0)
    {
      glVertexPointer(3, GL_FLOAT, 0, &vc[0]);
      glClientActiveTextureARB(GL_TEXTURE0_ARB);
      glTexCoordPointer(3, GL_FLOAT, 0, &tc0[0]);
      if (usePreIntegration)
      {
        glClientActiveTextureARB(GL_TEXTURE1_ARB);
        glTexCoordPointer(3, GL_FLOAT, 0, &tc1[0]);
      }

      glMultiDrawArrays(GL_TRIANGLE_FAN, &firsts[0], &counts[0], firsts.size());
      drawn += firsts.size();
    }

    vertCount = 0;
    firsts.clear();
    counts.clear();
    vc.clear();
    tc0.clear();
    tc1.clear();
  }
  glClientActiveTextureARB(activeTexture);
  glPopClientAttrib();

  vvDebugMsg::msg(3, "Number of textures drawn: ", drawn);

  if (_postClassification && _shader)
  {
    disableShader(_shader.get());
  }
  else
  {
    disableTexture(GL_TEXTURE_3D_EXT);
  }
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvTexRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL()");

  vvGLTools::printGLError("enter vvTexRend::renderVolumeGL()");

  activateClippingPlanes();

  virvo::vector< 3, ssize_t > vox = _paddingRegion.max - _paddingRegion.min;
  for (size_t i = 0; i < 3; ++i)
  {
    vox[i] = std::min(vox[i], vd->vox[i]);
  }

  if (vox[0] * vox[1] * vox[2] == 0)
    return;

  setGLenvironment();

  // Determine texture object extensions:
  for (size_t i = 0; i < 3; ++i)
  {
    // padded borders for (trilinear) interpolation
    size_t paddingLeft = size_t(abs(ptrdiff_t(_visibleRegion.min[i] - _paddingRegion.min[i])));
    size_t paddingRight = size_t(abs(ptrdiff_t(_visibleRegion.max[i] - _paddingRegion.max[i])));
    // a voxels size
    const float vsize = 1.0f / (float)texels[i];
    // half a voxels size
    const float vsize2 = 0.5f / (float)texels[i];
    if (paddingLeft == 0)
    {
      texMin[i] = vsize2;
    }
    else
    {
      texMin[i] = vsize * (float)paddingLeft;
    }

    texMax[i] = (float)vox[i] / (float)texels[i];
    if (paddingRight == 0)
    {
      texMax[i] -= vsize2;
    }
    else
    {
      texMax[i] -= vsize * (float)paddingRight;
    }
  }

  // allow for using raw volume data from vvVolDesc for textures without re-shuffeling
  std::swap(texMin[2], texMax[2]);
  std::swap(texMin[1], texMax[1]);

  // Get OpenGL modelview matrix:
  mat4 mv = gl::getModelviewMatrix();

  renderTex3DPlanar(mv);

  unsetGLenvironment();

  if (_fpsDisplay)
  {
    // Make sure rendering is done to measure correct time.
    // Since this operation is costly, only do it if necessary.
    glFinish();
  }

  deactivateClippingPlanes();

  vvDebugMsg::msg(3, "vvTexRend::renderVolumeGL() done");
}

//----------------------------------------------------------------------------
/** Activate the previously set clipping planes.
*/
void vvTexRend::activateClippingPlanes()
{
  vvDebugMsg::msg(3, "vvTexRend::activateClippingPlanes()");

  typedef vvRenderState::ParameterType PT;

  for ( PT act_id = VV_CLIP_OBJ_ACTIVE0, obj_id = VV_CLIP_OBJ0;
        act_id != VV_CLIP_OBJ_ACTIVE_LAST && obj_id != VV_CLIP_OBJ_LAST;
        act_id = PT(act_id + 1), obj_id = PT(obj_id + 1))
  {
    int i = act_id - VV_CLIP_OBJ_ACTIVE0;
    if (getParameter(act_id))
    {
      if (boost::shared_ptr<vvClipPlane> plane = boost::dynamic_pointer_cast<vvClipPlane>(getParameter(obj_id).asClipObj()))
      {
        if (i < maxClipPlanes && i != getParameter(VV_FOCUS_CLIP_OBJ).asInt())
        {
          // Generate OpenGL compatible clipping plane parameters:
          // normal points into opposite direction
          GLdouble planeEq[4];
          planeEq[0] = -plane->normal.x;
          planeEq[1] = -plane->normal.y;
          planeEq[2] = -plane->normal.z;
          planeEq[3] =  plane->offset;
          glClipPlane(GL_CLIP_PLANE0 + i, planeEq);
          glEnable(GL_CLIP_PLANE0 + i);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
/** Deactivate the clipping planes.
 */
void vvTexRend::deactivateClippingPlanes()
{
  vvDebugMsg::msg(3, "vvTexRend::deactivateClippingPlanes()");
  for (int i = 0; i < maxClipPlanes; ++i)
    glDisable(GL_CLIP_PLANE0 + i);
}

//----------------------------------------------------------------------------
/** Set number of lights in the scene.
  Fixed material characteristics are used with each setting.
  @param numLights  number of lights in scene (0=ambient light only)
*/
void vvTexRend::setNumLights(const int numLights)
{
  const float ambient[]  = {0.5f, 0.5f, 0.5f, 1.0f};
  const float pos0[] = {0.0f, 10.0f, 10.0f, 0.0f};
  const float pos1[] = {0.0f, -10.0f, -10.0f, 0.0f};

  vvDebugMsg::msg(1, "vvTexRend::setNumLights()");

  // Generate light source 1:
  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_POSITION, pos0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);

  // Generate light source 2:
  glEnable(GL_LIGHT1);
  glLightfv(GL_LIGHT1, GL_POSITION, pos1);
  glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);

  // At least 2 lights:
  if (numLights >= 2)
    glEnable(GL_LIGHT1);
  else
    glDisable(GL_LIGHT1);

  // At least one light:
  if (numLights >= 1)
    glEnable(GL_LIGHT0);
  else                                            // no lights selected
    glDisable(GL_LIGHT0);
}

//----------------------------------------------------------------------------
/// @return true if classification is done in no time
bool vvTexRend::instantClassification() const
{
  vvDebugMsg::msg(3, "vvTexRend::instantClassification()");
  return _postClassification;
}

//----------------------------------------------------------------------------
/// Returns the number of entries in the RGBA lookup table.
size_t vvTexRend::getLUTSize(virvo::vector< 3, size_t >& size) const
{
  size_t x, y, z;

  vvDebugMsg::msg(3, "vvTexRend::getLUTSize()");
  if (usePreIntegration)
  {
    x = y = getPreintTableSize();
    z = 1;
  }
  else
  {
    x = 256;
    if (vd->getChan() == 2 && vd->tf.size() == 1)
    {
       y = x;
       z = 1;
    }
    else
       y = z = 1;
  }

  size[0] = x;
  size[1] = y;
  size[2] = z;

  return x * y * z;
}

//----------------------------------------------------------------------------
/// Returns the size (width and height) of the pre-integration lookup table.
size_t vvTexRend::getPreintTableSize() const
{
  vvDebugMsg::msg(1, "vvTexRend::getPreintTableSize()");
  return 256;
}

//----------------------------------------------------------------------------
/** Update the color/alpha look-up table.
 Note: glColorTableSGI can have a maximum width of 1024 RGBA entries on IR2 graphics!
 @param dist  slice distance relative to 3D texture sample point distance
              (1.0 for original distance, 0.0 for all opaque).
*/
void vvTexRend::updateLUT(const float dist)
{
  vvDebugMsg::msg(3, "Generating texture LUT. Slice distance = ", dist);

  vec4 corr;                                      // gamma/alpha corrected RGBA values [0..1]
  virvo::vector< 3, size_t > lutSize;             // number of entries in the RGBA lookup table
  lutDistance = dist;

  size_t total = getLUTSize(lutSize);
  if (rgbaTF.size() != rgbaLUT.size())
  {
    rgbaLUT.resize(rgbaTF.size());
  }
  if (pixLUTName.size() > rgbaLUT.size())
  {
    glDeleteTextures(pixLUTName.size()-rgbaLUT.size(), &pixLUTName[rgbaLUT.size()]);
    pixLUTName.resize(rgbaLUT.size());
  }
  else if (pixLUTName.size() < rgbaLUT.size())
  {
    pixLUTName.resize(rgbaLUT.size());
    glGenTextures(pixLUTName.size()-rgbaLUT.size(), &pixLUTName[rgbaLUT.size()]);
  }
 
  for (size_t chan=0; chan<rgbaTF.size(); ++chan)
  {
    assert(total*4 == rgbaTF[chan].size());
    if (rgbaLUT[chan].size() != rgbaTF[chan].size())
      rgbaLUT[chan].resize(rgbaTF[chan].size());

    if (usePreIntegration)
    {
      vd->tf[chan].makePreintLUTCorrect(getPreintTableSize(), &rgbaLUT[chan][0], dist);
    }
    else
    {
      for (size_t i=0; i<total; ++i)
      {
        // Gamma correction:
        if (_gammaCorrection)
        {
          corr[0] = gammaCorrect(rgbaTF[chan][i * 4],     VV_RED);
          corr[1] = gammaCorrect(rgbaTF[chan][i * 4 + 1], VV_GREEN);
          corr[2] = gammaCorrect(rgbaTF[chan][i * 4 + 2], VV_BLUE);
          corr[3] = gammaCorrect(rgbaTF[chan][i * 4 + 3], VV_ALPHA);
        }
        else
        {
          corr[0] = rgbaTF[chan][i * 4];
          corr[1] = rgbaTF[chan][i * 4 + 1];
          corr[2] = rgbaTF[chan][i * 4 + 2];
          corr[3] = rgbaTF[chan][i * 4 + 3];
        }

        // Opacity correction:
        // for 0 distance draw opaque slices
        if (dist<=0.0 || (getParameter(VV_CLIP_MODE) && _clipOpaque)) corr[3] = 1.0f;
        else if (_opacityCorrection) corr[3] = 1.0f - powf(1.0f - corr[3], dist);

        // Convert float to uint8_t and copy to rgbaLUT array:
        for (size_t c=0; c<4; ++c)
        {
          rgbaLUT[chan][i * 4 + c] = uint8_t(corr[c] * 255.0f);
        }
      }
    }

    // Copy LUT to graphics card:
    vvGLTools::printGLError("enter updateLUT()");
    if (_postClassification)
    {
      glBindTexture(GL_TEXTURE_2D, pixLUTName[chan]);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lutSize[0], lutSize[1], 0,
          GL_RGBA, GL_UNSIGNED_BYTE, &rgbaLUT[chan][0]);
    }
  }

  if (!_postClassification)
    makeTextures(false);// this mode doesn't use a hardware LUT, so every voxel has to be updated

  vvGLTools::printGLError("leave updateLUT()");
}

//----------------------------------------------------------------------------
/** Set user's viewing direction.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the user is inside the volume.
  @param vd  viewing direction in object coordinates
*/
void vvTexRend::setViewingDirection(vec3f const& vd)
{
  vvDebugMsg::msg(3, "vvTexRend::setViewingDirection()");
  viewDir = vd;
}

//----------------------------------------------------------------------------
/** Set the direction from the viewer to the object.
  This information is needed to correctly orientate the texture slices
  in 3D texturing mode if the viewer is outside of the volume.
  @param vd  object direction in object coordinates
*/
void vvTexRend::setObjectDirection(vec3f const& od)
{
  vvDebugMsg::msg(3, "vvTexRend::setObjectDirection()");
  objDir = od;
}


bool vvTexRend::checkParameter(ParameterType param, vvParam const& value) const
{
  switch (param)
  {
  case VV_SLICEINT:

    {
      virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

      if (mode == virvo::Nearest || mode == virvo::Linear)
      {
        return true;
      }
    }

    return false;

  case VV_CLIP_OBJ0:
  case VV_CLIP_OBJ1:
  case VV_CLIP_OBJ2:
  case VV_CLIP_OBJ3:
  case VV_CLIP_OBJ4:
  case VV_CLIP_OBJ5:
  case VV_CLIP_OBJ6:
  case VV_CLIP_OBJ7:
  {
    boost::shared_ptr<vvClipObj> obj = value.asClipObj();
    if (!boost::dynamic_pointer_cast<vvClipPlane>(obj))
      return false;

    int i = static_cast<int>(param) - static_cast<int>(VV_CLIP_OBJ0);

    if (i >= maxClipPlanes)
      return false;

    return true;
  }

  default:

    return vvRenderer::checkParameter(param, value);

  }
}


//----------------------------------------------------------------------------
// see parent
void vvTexRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvTexRend::setParameter()");
  switch (param)
  {
    case vvRenderer::VV_POST_CLASSIFICATION:
      vvRenderer::setParameter(param, newValue);
      setupClassification();
      break;
    case vvRenderer::VV_GAMMA:
      // fall-through
    case vvRenderer::VV_GAMMA_CORRECTION:
      vvRenderer::setParameter(param, newValue);
      updateTransferFunction();
      break;
    case vvRenderer::VV_SLICEINT:
      if (_interpolation != static_cast< virvo::tex_filter_mode >(newValue.asInt()))
      {
        _interpolation = static_cast< virvo::tex_filter_mode >(newValue.asInt());
        for (size_t f = 0; f < vd->frames; ++f)
        {
          glBindTexture(GL_TEXTURE_3D_EXT, texNames[f]);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
          glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
        }
        updateTransferFunction();
      }
      break;
    case vvRenderer::VV_MIN_SLICE:
      minSlice = newValue;
      break;
    case vvRenderer::VV_MAX_SLICE:
      maxSlice = newValue;
      break;
    case vvRenderer::VV_SLICEORIENT:
      _sliceOrientation = (SliceOrientation)newValue.asInt();
      break;
    case vvRenderer::VV_PREINT:
      vvRenderer::setParameter(param, newValue);
      updateTransferFunction();
      disableShader(_shader.get());
      _shader.reset(initShader());
      break;
    case vvRenderer::VV_BINNING:
      vd->_binning = (vvVolDesc::BinningType)newValue.asInt();
      break;
    case vvRenderer::VV_OFFSCREENBUFFER:
    case vvRenderer::VV_USE_OFFSCREEN_BUFFER:
      {
        bool fbo = static_cast<bool>(newValue);

        this->_useOffscreenBuffer = fbo;

        if (fbo)
          setRenderTarget( virvo::FramebufferObjectRT::create() );
        else
          setRenderTarget( virvo::NullRT::create() );
      }
      break;
    case vvRenderer::VV_IMG_SCALE:
      //_imageScale = newValue;
      break;
    case vvRenderer::VV_IMG_PRECISION:
    case vvRenderer::VV_IMAGE_PRECISION:
      {
        virvo::BufferPrecision bp = mapBitsToBufferPrecision(static_cast<int>(newValue));

        this->_imagePrecision = bp;

//      setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH32F_STENCIL8) );
        setRenderTarget( virvo::FramebufferObjectRT::create(mapBufferPrecisionToFormat(bp), virvo::PF_DEPTH24_STENCIL8) );
      }
      break;
    case vvRenderer::VV_LIGHTING:
      vvRenderer::setParameter(param, newValue);
      disableShader(_shader.get());
      _shader.reset(initShader());
      break;
    case vvRenderer::VV_PIX_SHADER:
      setCurrentShader(newValue);
      break;
    case vvRenderer::VV_PADDING_REGION:
      vvRenderer::setParameter(param, newValue);
      makeTextures(true);
      break;
    case vvRenderer::VV_CHANNEL_WEIGHTS:
      vvRenderer::setParameter(param, newValue);
      disableShader(_shader.get());
      _shader.reset(initShader());
      break;
    default:
      vvRenderer::setParameter(param, newValue);
      break;
  }
}

//----------------------------------------------------------------------------
// see parent for comments
vvParam vvTexRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvTexRend::getParameter()");

  switch (param)
  {
    case vvRenderer::VV_MIN_SLICE:
      return minSlice;
    case vvRenderer::VV_MAX_SLICE:
      return maxSlice;
    case vvRenderer::VV_SLICEORIENT:
      return (int)_sliceOrientation;
    case vvRenderer::VV_BINNING:
      return (int)vd->_binning;
    case vvRenderer::VV_PIX_SHADER:
      return getCurrentShader();
    default:
      return vvRenderer::getParameter(param);
  }
}

//----------------------------------------------------------------------------
/** Return true if a feature is supported.
 */
bool vvTexRend::isSupported(const FeatureType feature)
{
  vvDebugMsg::msg(3, "vvTexRend::isSupported()");
  switch(feature)
  {
    case VV_MIP: return true;
    case VV_POST_CLASSIFICATION: return vvShaderFactory::isSupported("cg") || vvShaderFactory::isSupported("glsl");
    default: assert(0); break;
  }
  return false;
}

//----------------------------------------------------------------------------
/** Return the currently used pixel shader [0..numShaders-1].
 */
int vvTexRend::getCurrentShader() const
{
  vvDebugMsg::msg(3, "vvTexRend::getCurrentShader()");
  return _currentShader;
}

//----------------------------------------------------------------------------
/** Set the currently used pixel shader [0..numShaders-1].
 */
void vvTexRend::setCurrentShader(const int shader)
{
  vvDebugMsg::msg(3, "vvTexRend::setCurrentShader()");
  if(shader >= NUM_PIXEL_SHADERS || shader < 0)
    _currentShader = ShaderMultiTF;
  else
    _currentShader = shader;

  disableShader(_shader.get());
  _shader.reset(initShader());
}

//----------------------------------------------------------------------------
/// inherited from vvRenderer, only valid for planar textures
void vvTexRend::renderQualityDisplay() const
{
  const int numSlices = int(_quality * 100.0f);
  vvPrintGL printGL;
  vec4f clearColor = vvGLTools::queryClearColor();
  vec4f fontColor( 1.0f - clearColor[0], 1.0f - clearColor[1], 1.0f - clearColor[2], 1.0f );
  printGL.setFontColor(fontColor);
  printGL.print(-0.9f, 0.9f, "Textures: %d", numSlices);
}

//----------------------------------------------------------------------------
void vvTexRend::enableTexture(const GLenum target) const
{
  if (!_postClassification)
    glEnable(target);
}

//----------------------------------------------------------------------------
void vvTexRend::disableTexture(const GLenum target) const
{
  if (!_postClassification)
    glDisable(target);
}

//----------------------------------------------------------------------------
void vvTexRend::enableShader(vvShaderProgram* shader) const
{
  vvGLTools::printGLError("Enter vvTexRend::enableShader()");

  if(!shader)
    return;

  shader->enable();

  if (_postClassification)
  {
    if (_currentShader == ShaderMultiTF)
    {
      for (size_t chan=0; chan < pixLUTName.size(); ++chan)
      {
        std::stringstream str;
        str << "pixLUT" << chan;
        shader->setParameterTex2D(str.str().c_str(), pixLUTName[chan]);
      }

      if (_useChannelWeights)
      {
        shader->setParameterArray1f("channelWeights", &vd->channelWeights[0], vd->channelWeights.size());
      }
    }
    else
    {
      shader->setParameterTex2D("pixLUT", pixLUTName[0]);
    }

    if (_channel4Color != NULL)
    {
      shader->setParameter3f("chan4color", _channel4Color[0], _channel4Color[1], _channel4Color[2]);
    }
    if (_opacityWeights != NULL)
    {
      shader->setParameter4f("opWeights", _opacityWeights[0], _opacityWeights[1], _opacityWeights[2], _opacityWeights[3]);
    }

    shader->setParameter1i("preintegration", usePreIntegration ? 1 : 0);
  }

  vvGLTools::printGLError("Leaving vvTexRend::enableShader()");
}

//----------------------------------------------------------------------------
void vvTexRend::disableShader(vvShaderProgram* shader) const
{
  vvGLTools::printGLError("Enter vvTexRend::disableShader()");

  if (shader)
  {
    shader->disable();
  }

  vvGLTools::printGLError("Leaving vvTexRend::disableShader()");
}

//----------------------------------------------------------------------------
/** @return Pointer of initialized ShaderProgram or NULL
 */
vvShaderProgram* vvTexRend::initShader()
{
  vvGLTools::printGLError("Enter vvTexRend::initShader()");

  std::ostringstream fragName;
  if (_postClassification)
  {
    fragName << "shader" << std::setw(2) << std::setfill('0') << (_currentShader+1);
  }

  std::stringstream defines;
  defines << "#define NUM_CHANNELS " << vd->getChan() << std::endl;
  if (_lighting)
  {
    defines << "#define LIGHTING 1" << std::endl;
  }
  if (usePreIntegration)
  {
    defines << "#define PREINTEGRATION 1" << std::endl;
  }
  if (_useChannelWeights)
  {
    defines << "#define CHANNEL_WEIGHTS 1" << std::endl;
  }

  _shaderFactory->setDefines(defines.str());

  // intersection on CPU, try to create fragment program
  vvShaderProgram* shader = _shaderFactory->createProgram("", "", fragName.str());

  vvGLTools::printGLError("Leave vvTexRend::initShader()");

  return shader;
}

//----------------------------------------------------------------------------
void vvTexRend::printLUT(size_t chan) const
{
  virvo::vector< 3, size_t > lutEntries;

  size_t total = getLUTSize(lutEntries);
  for (size_t i=0; i<total; ++i)
  {
    cerr << "#" << i << ": ";
    for (size_t c=0; c<4; ++c)
    {
      cerr << int(rgbaLUT[chan][i * 4 + c]);
      if (c<3) cerr << ", ";
    }
    cerr << endl;
  }
}

uint8_t* vvTexRend::getHeightFieldData(float points[4][3], size_t& width, size_t& height)
{
  GLint viewport[4];
  uint8_t *pixels, *data, *result=NULL;
  size_t numPixels;
  size_t index;
  float sizeX, sizeY;
  vec3 size, size2;
  vec3 texcoord[4];

  std::cerr << "getHeightFieldData" << endl;

  glGetIntegerv(GL_VIEWPORT, viewport);

  width = size_t(ceil(getManhattenDist(points[0], points[1])));
  height = size_t(ceil(getManhattenDist(points[0], points[3])));

  numPixels = width * height;
  pixels = new uint8_t[4*numPixels];

  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  size = vd->getSize();
  for (size_t i = 0; i < 3; ++i)
    size2[i]   = 0.5f * size[i];

  for (size_t j = 0; j < 4; j++)
    for (size_t k = 0; k < 3; k++)
  {
    texcoord[j][k] = (points[j][k] + size2[k]) / size[k];
    texcoord[j][k] = texcoord[j][k] * (texMax[k] - texMin[k]) + texMin[k];
  }

  enableTexture(GL_TEXTURE_3D_EXT);
  glBindTexture(GL_TEXTURE_3D_EXT, texNames[vd->getCurrentFrame()]);

  if (glIsTexture(texNames[vd->getCurrentFrame()]))
    std::cerr << "true" << endl;
  else
    std::cerr << "false" << endl;

  sizeX = 2.0f * float(width)  / float(viewport[2] - 1);
  sizeY = 2.0f * float(height) / float(viewport[3] - 1);

  std::cerr << "SizeX: " << sizeX << endl;
  std::cerr << "SizeY: " << sizeY << endl;
  std::cerr << "Viewport[2]: " << viewport[2] << endl;
  std::cerr << "Viewport[3]: " << viewport[3] << endl;

  std::cerr << "TexCoord1: " << texcoord[0][0] << " " << texcoord[0][1] << " " << texcoord[0][2] << endl;
  std::cerr << "TexCoord2: " << texcoord[1][0] << " " << texcoord[1][1] << " " << texcoord[1][2] << endl;
  std::cerr << "TexCoord3: " << texcoord[2][0] << " " << texcoord[2][1] << " " << texcoord[2][2] << endl;
  std::cerr << "TexCoord4: " << texcoord[3][0] << " " << texcoord[3][1] << " " << texcoord[3][2] << endl;

  glBegin(GL_QUADS);
  glTexCoord3f(texcoord[0][0], texcoord[0][1], texcoord[0][2]);
  glVertex3f(-1.0, -1.0, -1.0);
  glTexCoord3f(texcoord[1][0], texcoord[1][1], texcoord[1][2]);
  glVertex3f(sizeX, -1.0, -1.0);
  glTexCoord3f(texcoord[2][0], texcoord[2][1], texcoord[2][2]);
  glVertex3f(sizeX, sizeY, -1.0);
  glTexCoord3f(texcoord[3][0], texcoord[3][1], texcoord[3][2]);
  glVertex3f(-1.0, sizeY, -1.0);
  glEnd();

  glFinish();
  glReadBuffer(GL_BACK);

  data = new uint8_t[texelsize * numPixels];
  memset(data, 0, texelsize * numPixels);
  glReadPixels(viewport[0], viewport[1], width, height,
    GL_RGB, GL_UNSIGNED_BYTE, data);

  std::cerr << "data read" << endl;

  if (vd->getChan() == 1 && (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4))
  {
    result = new uint8_t[numPixels];
    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
    {
      index = y * width + x;
      if (_postClassification)
        result[index] = data[texelsize*index];
      else
        assert(0);
      std::cerr << "Result: " << index << " " << (int) (result[index]) << endl;
    }
  }
  else if (vd->bpc == 1 || vd->bpc == 2 || vd->bpc == 4)
  {
    result = new uint8_t[vd->getChan() * numPixels];

    for (size_t y = 0; y < height; y++)
      for (size_t x = 0; x < width; x++)
    {
      index = (y * width + x) * vd->getChan();
      for (int c = 0; c < vd->getChan(); c++)
      {
        result[index + c] = data[index + c];
        std::cerr << "Result: " << index+c << " " << (int) (result[index+c]) << endl;
      }
    }
  }

  std::cerr << "result read" << endl;

  disableTexture(GL_TEXTURE_3D_EXT);

  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

  return result;
}

float vvTexRend::getManhattenDist(float p1[3], float p2[3]) const
{
  float dist = 0;

  for (size_t i=0; i<3; ++i)
  {
    dist += float(fabs(p1[i] - p2[i])) / float(vd->getSize()[i] * vd->vox[i]);
  }

  std::cerr << "Manhattan Distance: " << dist << endl;

  return dist;
}

void vvTexRend::initLight(vvShaderProgram* shader, mat4 const& mv, vec3 const& normal, float sliceThickness)
{
  if (_postClassification)
  {
    shader->setParameter1i("lighting", _lighting ? 1 : 0);

    if (_currentShader == ShaderLighting || _currentShader == ShaderMultiTF || _currentShader == Shader2DTF)
    {
      // Local illumination based on blinn-phong shading.
      gl::light l = gl::getLight(GL_LIGHT0);

      // transform pos in eye coords to object coords
      vec4 lposobj = inverse(mv) * l.position;
      vec3 lpos = lposobj.xyz();

      // Viewing direction.
      vec3 V = normal;

      shader->setParameter3f("V", V.x, V.y, V.z);
      shader->setParameter3f("lpos", lpos.x, lpos.y, lpos.z);
      shader->setParameter1f("constAtt", l.constant_attenuation);
      shader->setParameter1f("linearAtt", l.linear_attenuation);
      shader->setParameter1f("quadAtt", l.quadratic_attenuation);

      shader->setParameter1f("threshold", 1.f - powf(1.f-0.1f, sliceThickness));
    }
  }
}

size_t vvTexRend::getTextureSize(size_t sz) const
{
  if (extNonPower2)
    return sz;

  return vvToolshed::getTextureSize(sz);
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
