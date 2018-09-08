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

#ifndef VV_RENDERER_H
#define VV_RENDERER_H

#include "math/math.h"
#include "texture/forward.h"

#include "vvmacros.h"
#include "vvparam.h"
#include "vvinttypes.h"
#include "vvrendertarget.h"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/static_assert.hpp>

class vvClipObj;
class vvStopwatch;
class vvVolDesc;

//============================================================================
// Class Definition
//============================================================================

class VIRVOEXPORT vvRenderState
{
public:
  enum { NUM_CLIP_OBJS = 8 };

  enum IbrMode                                  ///< how to determine ibr depth per pixel values
  {
    // single-pass
    VV_ENTRANCE = 0,                            ///< first opaque voxel
    VV_EXIT,                                    ///< last opaque voxel
    VV_MIDPOINT,                                ///< midpoint of entrance and exit
    VV_THRESHOLD,                               ///< accumulate up to some threshold
    VV_PEAK,                                    ///< peak opacity
    VV_GRADIENT,                                ///< max gradient

    // two-pass
    VV_REL_THRESHOLD,                           ///< some threshold, weighted by max opacity
    VV_EN_EX_MEAN,                              ///< Mean of entrance and exit
    VV_NONE
  };


  enum ParameterType                            ///<  Names for rendering parameters
  {
    VV_QUALITY = 0,
    VV_VISIBLE_REGION,
    VV_PADDING_REGION,
    VV_MIP_MODE,
    VV_ALPHA_MODE,
    VV_POST_CLASSIFICATION,
    VV_LEAPEMPTY,                               ///< empty space leaping
    VV_CLIP_PERIMETER,
    VV_BOUNDARIES,
    VV_ORIENTATION,
    VV_PALETTE,
    VV_QUALITY_DISPLAY,
    VV_CHANNEL_WEIGHTS,
    VV_CLIP_MODE,
    VV_CLIP_PLANE_POINT,
    VV_CLIP_PLANE_NORMAL,
    VV_CLIP_COLOR,
    VV_CLIP_SPHERE_CENTER,
    VV_CLIP_SPHERE_RADIUS,
    VV_CLIP_SINGLE_SLICE,
    VV_CLIP_OPAQUE,
    VV_IS_ROI_USED,
    VV_ROI_POS,
    VV_ROI_SIZE,
    VV_SPHERICAL_ROI,
    VV_BRICK_SIZE,
    VV_MAX_BRICK_SIZE,
    VV_BRICK_TEXEL_OVERLAP,
    VV_SHOW_BRICKS,
    VV_COMPUTE_BRICK_SIZE,
    VV_TEX_MEMORY_SIZE,
    VV_FPS_DISPLAY,
    VV_GAMMA_CORRECTION,
    VV_GAMMA,
    VV_OPACITY_WEIGHTS,
    VV_BOUND_COLOR,
    VV_PROBE_COLOR,
    VV_USE_OFFSCREEN_BUFFER,
    VV_IMAGE_SCALE,
    VV_IMAGE_PRECISION,
    VV_SHOW_TEXTURE,
    VV_USE_IBR,
    VV_IBR_DEPTH_RANGE,
    VV_IBR_MODE,
    VV_IBR_DEPTH_PREC,                          ///< precision of generated pixel depth values (8, 16 or 32 bit)
    VV_CODEC,                                   ///< id of image/video codec for remote rendering
    VV_IBR_SYNC,                                ///< display each image as it is sent

    VV_OPCORR,                                  ///< opacity correction on/off
    VV_SLICEINT,                                ///< interpolation within slice
    VV_WARPINT,                                 ///< interpolation during warp (shear-warp)
    VV_INTERSLICEINT,                           ///< interpolation between slices
    VV_PREINT,                                  ///< pre-integration on/off
    VV_MIN_SLICE,                               ///< minimum slice index to render
    VV_MAX_SLICE,                               ///< maximum slice index to render
    VV_BINNING,                                 ///< binning type (linear, iso-value, opacity)
    VV_SLICEORIENT,                             ///< slice orientation for planer 3d textures
    VV_ISECT_TYPE,                              ///< proxy geometry using different shaders or the gpu
    VV_TERMINATEEARLY,                          ///< terminate rays early
    VV_OFFSCREENBUFFER,                         ///< offscreen buffer on/off
    VV_IMG_SCALE,                               ///< downsample img by reducing img resolution [0..1]
    VV_IMG_PRECISION,                           ///< render to high-res target to minimize slicing rounding error
    VV_LIGHTING,
    VV_PIX_SHADER,

    VV_FOCUS_CLIP_OBJ,                          ///< clip object that is currently manipulated

    VV_CLIP_OBJ0,
    VV_CLIP_OBJ1,
    VV_CLIP_OBJ2,
    VV_CLIP_OBJ3,
    VV_CLIP_OBJ4,
    VV_CLIP_OBJ5,
    VV_CLIP_OBJ6,
    VV_CLIP_OBJ7,
    VV_CLIP_OBJ_LAST,

    VV_CLIP_OBJ_ACTIVE0,
    VV_CLIP_OBJ_ACTIVE1,
    VV_CLIP_OBJ_ACTIVE2,
    VV_CLIP_OBJ_ACTIVE3,
    VV_CLIP_OBJ_ACTIVE4,
    VV_CLIP_OBJ_ACTIVE5,
    VV_CLIP_OBJ_ACTIVE6,
    VV_CLIP_OBJ_ACTIVE7,
    VV_CLIP_OBJ_ACTIVE_LAST,

    VV_CLIP_OUTLINE0,
    VV_CLIP_OUTLINE1,
    VV_CLIP_OUTLINE2,
    VV_CLIP_OUTLINE3,
    VV_CLIP_OUTLINE4,
    VV_CLIP_OUTLINE5,
    VV_CLIP_OUTLINE6,
    VV_CLIP_OUTLINE7,
    VV_CLIP_OUTLINE_LAST
  };

  BOOST_STATIC_ASSERT( VV_CLIP_OBJ_LAST - VV_CLIP_OBJ0 == NUM_CLIP_OBJS );
  BOOST_STATIC_ASSERT( VV_CLIP_OBJ_ACTIVE_LAST - VV_CLIP_OBJ_ACTIVE0 == NUM_CLIP_OBJS );
  BOOST_STATIC_ASSERT( VV_CLIP_OUTLINE_LAST - VV_CLIP_OUTLINE0 == NUM_CLIP_OBJS );

  virtual bool checkParameter(ParameterType param, vvParam const& value) const;
  virtual void setParameter(ParameterType param, const vvParam& value);
  virtual vvParam getParameter(ParameterType param) const;

protected:
  float _quality;                               ///< rendering image quality (0=minimum, 1=sampling rate, >1=oversampling)
  int   _mipMode;                               ///< min/maximum intensity projection (0=off, 1=max, 2=min)
  int	  _alphaMode;                             ///< calculation for alpha value (0=max of channel weights*values, 1=weighted avg)
  bool  _postClassification;                    ///< true = use post-classification transfer function lookup, false = use pre-classification
  bool  _emptySpaceLeaping;                     ///< true = don't render bricks without contribution
  bool  _boundaries;                            ///< true = display volume boundaries
  bool  _orientation;                           ///< true = display object orientation
  bool  _palette;                               ///< true = display transfer function palette
  bool  _qualityDisplay;                        ///< true = display rendering quality level
  bool _useChannelWeights;                      ///< true = mind channel weights
  bool _clipMode;                               ///< true = use clipping plane 0
  bool      _clipPlanePerimeter;                ///< true = render line around clipping plane
  vvColor   _clipPlaneColor;                    ///< clipping plane boundary color (R,G,B in [0..1])
  bool  _clipSingleSlice;                       ///< true = use single slice in clipping mode
  bool  _clipOpaque;                            ///< true = make single slice opaque
  bool  _isROIUsed;                             ///< true = use roi
  bool  _isROIChanged;                          ///< true = roi related values have been changed
  virvo::vec3f roi_pos_;                        ///< object space coordinates of ROI midpoint [mm]
  virvo::vec3f roi_size_;                       ///< size of roi in each dimension [0..1]
  bool  _sphericalROI;                          ///< true = use sphere rather than cube for roi

  virvo::vector< 3, size_t > _brickSize;        ///< last bricksize in x/y/z
  virvo::vector< 3, size_t > _maxBrickSize;     ///< max allowed bricksize in x/y/z
  size_t   _brickTexelOverlap;                  /*!  overlap needed for performing calculations at brick borders
                                                     max value: min(brickSize[d])/2-1 */
  bool  _showBricks;                            ///< true = show brick boundarys
  bool  _computeBrickSize;                      ///< true = calculate brick size
  size_t   _texMemorySize;                      ///< size of texture memory
  bool  _fpsDisplay;                            ///< true = show frame rate, might be costly if e. g. glFinish is called
  bool  _gammaCorrection;                       ///< true = gamma correction on
  virvo::vec4f gamma_;                          ///< gamma correction value: 0=red, 1=green, 2=blue, 3=4th channel
  bool  _useOpacityWeights;                        ///< true = for multi-channel data sets only: allow weighted opacities in channels
  vvColor _boundColor;                          ///< boundary color (R,G,B in [0..1])
  vvColor _probeColor;                          ///< probe boundary color (R,G,B in [0..1])
  bool  _useOffscreenBuffer;                    ///< render target for image downscaling
  float _imageScale;                            ///< undersampling by downscaling rendered img [0..1]
  virvo::BufferPrecision _imagePrecision;       /*!  render to high-res offscreen buffer (32 bit float) to minimize rounding error
                                                     caused by adding up contribution of to many slices */
  int _currentShader;                           ///< ID of currently used fragment shader
  bool _lighting;                               ///< use local illumination
  bool _showTexture;                            ///< true = show texture mapping, if applicable, added by Han, Feb 2008
  bool _useIbr;                                 ///< use ibr in rayrenderer
  IbrMode _ibrMode;
  virvo::basic_aabb< ssize_t > _visibleRegion;  ///< part of the vd that is visible and thus rendered
  virvo::basic_aabb< ssize_t > _paddingRegion;  ///< padding region for interpolation
  bool _opacityCorrection;                      ///< true = opacity correction on
  virvo::tex_filter_mode _interpolation;
  bool _earlyRayTermination;                    ///< terminate ray marching when enough alpha was gathered
  bool _preIntegration;                         ///< true = try to use pre-integrated rendering (planar 3d textures)
  int _depthPrecision;                          ///< number of bits in depth buffer for image based rendering
  virvo::vec2f depth_range_;

  boost::shared_ptr<vvClipObj> _clipObjs[NUM_CLIP_OBJS];
  int _focusClipObj;                            ///< clip object that is currently manipulated
  bool _clipObjsActive[NUM_CLIP_OBJS];
  bool _clipOutlines[NUM_CLIP_OBJS];
public:
  vvRenderState();
};

/** Abstract volume rendering class.
  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvSoftVR
  @see vvTexRend
  @see vvVolTex
  @see vvRenderVP
*/
class VIRVOEXPORT vvRenderer : public vvRenderState
{

  VV_NOT_COPYABLE(vvRenderer)

  public:
    enum RendererType                             /// Current renderer
    {
      INVALID = -2,                               ///< no renderer
      GENERIC = -1,                               ///< generic renderer (does nothing)
      TEXREND = 0,                                ///< texture based renderer
      SOFTPAR,                                    ///< software based renderer for parallel projection
      SOFTPER,                                    ///< software based renderer for perspective projection
      SOFTSW,                                     ///< software based shear-warp renderer
      RAYREND,                                    ///< ray casting renderer
      RENDERVP,                                   ///< Phil Lacroute's VolPack renderer
      VOLPACK,                                    ///< Phil Lacroute's VolPack renderer or no rendering for perspective projection
      SIMIAN,                                     ///< Joe Kniss's Simian renderer
      IMGREND,                                    ///< 2D image renderer
      UNKNOWN,                                    ///< unknown renderer
      STINGRAY,                                   ///< Imedia's Stingray renderer
      VIRTEXREND,                                 ///< virtualized texture memory using bricking + out-of-core
      REMOTE_IBR,                                 ///< image based remote renderer
      REMOTE_IMAGE,                               ///< image based remote renderer
      SERBRICKREND,                               ///< serial brick renderer
      PARBRICKREND,                               ///< parallel brick renderer
      NUM_RENDERERS                               ///< keep last
    };

    enum BasicColorType                           /// basic colors
    {
      VV_RED = 0,
      VV_GREEN,
      VV_BLUE,
      VV_ALPHA
    };

  protected:
    RendererType rendererType;                    ///< currently used renderer type
    vvVolDesc* vd;                                ///< volume description
    float      _channel4Color[3];                 ///< weights for visualization of 4th channel in RGB mode
    float      _opacityWeights[4];                ///< opacity weights for alpha blending with 4 channels
    virvo::vec3f viewDir;                         ///< user's current viewing direction [object coordinates]
    virvo::vec3f objDir;                          ///< direction from viewer to object [object coordinates]

    virtual void init();                   		  ///< initialization routine

    void getObjNormal(virvo::vec3& normal,
                      virvo::vec3& origin,
                      virvo::vec3 const& eye,
                      virvo::mat4 const& invMV,
                      bool isOrtho = false) const;

    void getShadingNormal(virvo::vec3& normal,
                          virvo::vec3& origin,
                          virvo::vec3 const& eye,
                          virvo::mat4 const& invMV,
                          bool isOrtho = false) const;
    void calcProbeDims(virvo::vec3& probePosObj, virvo::vec3& probeSizeObj,
        virvo::vec3& probeMin, virvo::vec3& probeMax) const;

    // Class Methods:
  public:                                         // public methods will be inherited as public
    vvRenderer(vvVolDesc*, vvRenderState);
    virtual ~vvRenderer();

    // Static methods:
    static float adaptQuality(float, float, float, float);

    // Public methods that should be redefined by subclasses:
    virtual RendererType getRendererType() const;
    virtual void  renderVolumeGL();
    virtual void  renderVolumeRGB(int, int, uchar*);
    virtual void  renderMultipleVolume();
    virtual void  updateTransferFunction();
    virtual void  updateVolumeData();
    virtual size_t getNumFrames();
    virtual size_t getCurrentFrame();
    virtual void  setCurrentFrame(size_t);
    virtual float getLastRenderTime() const;
    virtual void  setPosition( virvo::vec3f const& p );
    virtual virvo::vec3f getPosition() const;
    virtual void  renderCoordinates() const;
    virtual void  renderPalette() const;
    virtual void  renderQualityDisplay() const;
    virtual void  renderFPSDisplay() const;
    virtual void  drawBoundingBox( virvo::vec3f const& oSize, virvo::vec3f const& oPos, const vvColor& color ) const;
    virtual void  drawPlanePerimeter( virvo::vec3f const& oSize, virvo::vec3f const& oPos,
        virvo::vec3f const& oPlane, virvo::vec3f const& oNorm, const vvColor& ) const;
    virtual void  drawSpherePerimeter( virvo::vec3 const& oSize, virvo::vec3 const& oPos,
        virvo::vec3 const& oCenter, float radius, const vvColor& color ) const;
    virtual bool  instantClassification() const;
    virtual void  setViewingDirection( virvo::vec3f const& vd );
    virtual void  setObjectDirection( virvo::vec3f const& od );
    virtual void  setROIEnable(bool);
    virtual void  setSphericalROI(bool sphericalROI);
    virtual bool  isROIEnabled() const;
    virtual void  setProbePosition( virvo::vec3f const& pos );
    virtual virvo::vec3f  getProbePosition() const;
    virtual void  setProbeSize( virvo::vec3f const& newSize );
    virtual virvo::vec3f  getProbeSize() const;
    virtual virvo::vec3f  getEyePosition() const;
    virtual bool  isInVolume( virvo::vec3f const& point ) const;
    virtual float getAlphaValue(float, float, float);
    virtual void  setParameter(ParameterType param, const vvParam& value);
    virtual vvParam getParameter(ParameterType param) const;
    virtual void  profileStart();
    virtual void  profileStop();
    virtual void  setGamma(BasicColorType, float);
    virtual float getGamma(BasicColorType);
    virtual float gammaCorrect(float, BasicColorType);
    virtual void  setChannel4Color(BasicColorType, float);
    virtual float getChannel4Color(BasicColorType);
    virtual void  setOpacityWeight(BasicColorType, float);
    virtual float getOpacityWeight(BasicColorType);

    // added by Han Kim Feb. 2008
    virtual void setVolDesc(vvVolDesc*);
    virtual vvVolDesc* getVolDesc();

    virtual virvo::cartesian_axis< 3 > getPrincipalViewingAxis(virvo::mat4 const& mv, float& zx, float& zy, float& zz) const;

    // Returns the current render target
    virvo::RenderTarget const* getRenderTarget() const {
      return renderTarget_.get();
    }

    // Returns the current render target
    virvo::RenderTarget* getRenderTarget() {
      return renderTarget_.get();
    }

    // Sets a new render-target
    virtual void setRenderTarget(virvo::RenderTarget* rt) {
      renderTarget_.reset(rt);
    }

    // Prepare this renderer to render a new frame
    virtual bool beginFrame(unsigned clearMask);

    // Called when a new frame has been rendered
    virtual bool endFrame();

    // Resize the current render-target
    virtual bool resize(int w, int h);

    // Displays the render-target
    virtual bool present() const;

    void renderBoundingBox() const;
    void renderROI() const;
    void renderClipObjs() const;
    void renderHUD() const;
    void renderOpaqueGeometry() const;

    // Renders and displays a single frame in one go
    void renderFrame();

    // Renders and displays a single frame in one go
    void renderFrame(int w, int h);

private:

    // The current render target
    boost::scoped_ptr<virvo::RenderTarget> renderTarget_;

    // For fps display
    boost::scoped_ptr<vvStopwatch> stopwatch_;


    float		_lastRenderTime;                   ///< time it took to render the previous frame (seconds)
    float		_lastComputeTime;
    float		_lastPlaneSortingTime;
    float		_lastGLdrawTime;

};

template<class A>
void serialize(A& a, vvRenderState::ParameterType& pt, unsigned /*version*/)
{
    a & static_cast<unsigned>(pt);
}

template<class A>
void serialize(A& a, vvRenderer::RendererType& rt, unsigned /*version*/)
{
    a & static_cast<unsigned>(rt);
}

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
