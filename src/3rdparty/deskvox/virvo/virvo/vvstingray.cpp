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

#ifdef USE_STINGRAY

// Cg:
#ifdef HAVE_CG
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#endif

// StingRay:
#include <StingRayCaveRenderer.hpp>

// Virvo:
#include "vvdebugmsg.h"
#include "vvclock.h"
#include "vvstingray.h"

//----------------------------------------------------------------------------
/** Constructor.
  @param vd  volume description
  @param m   render geometry (default: automatic)
*/
vvStingray::vvStingray(vvVolDesc* vd, vvRenderState renderState) : vvRenderer(vd, renderState)
{
  vvDebugMsg::msg(1, "vvStingray::vvStingray()");
  cerr << "vvStingray::vvStingray() called." << endl;

  rendererType = STINGRAY;
  interpolation = true;
  mip = 0;
  gammaCorrection = false;
  rgbaTF  = new float[4096 * 4];
  rgbaLUT = new uchar[4096 * 4];

  //****************** initialize Stingray renderer
  // if mip==0 then alpha blending should be used,
  // if mip==1 maximum intensity projection should be use
  //
  // if interpolation==true: linear interpolation
  // if false: nearest neighbor
  _stingRay = new fvStingRayCaveRenderer();
  unsigned char* test = new unsigned char[32 * 32 * 32];
  memset(test, 128, 32*32*32);
  //  _stingRay->initializeRenderer(2, 2, fvStingRayCaveRenderer::F_ISOSURFACE,
  //    fvStingRayCaveRenderer::F_PERSPECTIVE, vd->getRaw(), vd->vox[0], vd->vox[1], vd->vox[2]);
  _stingRay->initializeRenderer(2, 2, fvStingRayCaveRenderer::F_ISOSURFACE,
    fvStingRayCaveRenderer::F_PERSPECTIVE, test, 32, 32, 32);

  updateTransferFunction();
  cerr << "vvStingray::vvStingray() done" << endl;
}

//----------------------------------------------------------------------------
/// Destructor
vvStingray::~vvStingray()
{
  delete _stingRay;
}

//----------------------------------------------------------------------------
/// Update transfer function from volume description.
void vvStingray::updateTransferFunction()
{
  int i, c;
  float* rgba;
  int lutEntries;

  vvDebugMsg::msg(1, "vvStingray::updateTransferFunction()");

  lutEntries = 256;
  rgba = new float[4 * lutEntries];

  // Generate arrays from pins:
  vd->tf.computeTFTexture(lutEntries, 1, 1, rgba);

  // Copy RGBA values to internal array:
  for (i=0; i<lutEntries; ++i)
  {
    for (c=0; c<4; ++c)
    {
      rgbaTF[i * 4 + c] = rgba[i * 4 + c];
    }
  }

  delete[] rgba;
}

//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvStingray::setGLenvironment()
{
  vvDebugMsg::msg(3, "vvStingray::setGLenvironment()");

  // Save current GL state:
  glGetBooleanv(GL_CULL_FACE, &glsCulling);
  glGetBooleanv(GL_BLEND, &glsBlend);
  glGetBooleanv(GL_COLOR_MATERIAL, &glsColorMaterial);
  glGetIntegerv(GL_BLEND_SRC, &glsBlendSrc);
  glGetIntegerv(GL_BLEND_DST, &glsBlendDst);
  glGetBooleanv(GL_LIGHTING, &glsLighting);
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glGetIntegerv(GL_BLEND_EQUATION_EXT, &glsBlendEquation);
  glGetBooleanv(GL_DEPTH_TEST, &glsDepthTest);
  glGetIntegerv(GL_DEPTH_FUNC, &glsDepthFunc);

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);

  vvDebugMsg::msg(3, "vvStingray::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvStingray::unsetGLenvironment()
{
  vvDebugMsg::msg(3, "vvStingray::unsetGLenvironment()");

  if (glsCulling==(uchar)true) glEnable(GL_CULL_FACE);
  else glDisable(GL_CULL_FACE);

  if (glsBlend==(uchar)true) glEnable(GL_BLEND);
  else glDisable(GL_BLEND);

  if (glsColorMaterial==(uchar)true) glEnable(GL_COLOR_MATERIAL);
  else glDisable(GL_COLOR_MATERIAL);

  if (glsDepthTest==(uchar)true) glEnable(GL_DEPTH_TEST);
  else glDisable(GL_DEPTH_TEST);

  if (glsLighting==(uchar)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);

  glDepthFunc(glsDepthFunc);
  glBlendFunc(glsBlendSrc, glsBlendDst);
  glMatrixMode(glsMatrixMode);

  vvDebugMsg::msg(3, "vvStingray::unsetGLenvironment() done");
}

//----------------------------------------------------------------------------
/** Render the volume onto currently selected drawBuffer.
 Viewport size in world coordinates is -1.0 .. +1.0 in both x and y direction
*/
void vvStingray::renderVolumeGL()
{
  GLint viewPort[4];
  GLfloat glsRasterPos[4];                        // current raster position (glRasterPos)
  vvVector3 size;
  static vvStopwatch* sw = new vvStopwatch();     // stop watch for performance measurements
  static int imgSize[2] =                         // width and height of Stingray window
  {
    0,0
  };
  unsigned char* image=NULL;

  vvDebugMsg::msg(3, "vvStingray::renderVolumeGL()");
  cerr << "vvStingray::renderVolumeGL() called" << endl;

  sw->start();

  // Draw boundary lines (must be done before setGLenvironment()):
  size = vd->getSize();
  if (_renderState._boundaries)
  {
    drawBoundingBox(&size, &vd->pos, boundColor);
  }
  if (_renderState.getParameter(VV_CLIP_MODE))
  {
    drawPlanePerimeter(&size, &vd->pos, &_renderState._clipPoint,
      &_renderState._clipNormal, _renderState._clipColor);
  }

  setGLenvironment();

  // Update image size:
  glGetIntegerv(GL_VIEWPORT, viewPort);
  if (viewPort[2]!=imgSize[0] || viewPort[3]!=imgSize[1] || !image)
  {
    imgSize[0] = viewPort[2];
    imgSize[1] = viewPort[3];
    //delete[] image;
    //image = new unsigned char[imgSize[0] * imgSize[1] * 4];
    _stingRay->setImageSize(imgSize[0], imgSize[1]);
    cerr << "New image size: " << imgSize[0] << " x " << imgSize[1] << endl;
  }
  image = _stingRay->renderVolume();

  // Save and set viewing matrix states:
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Store raster position:
  glGetFloatv(GL_CURRENT_RASTER_POSITION, glsRasterPos);

  // Draw stingray image:
  glRasterPos2f(-1.0f,-1.0f);                     // pixmap origin is bottom left corner of output window
  glDrawPixels(viewPort[2], viewPort[3], GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)image);

  // Restore raster position:
  glRasterPos4fv(glsRasterPos);

  // Restore matrix states:
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);

  /**************************************************************************
   * This function can be used with a definition of
   * setBoundingBox( const double boxWidth, boxHeight, boxDepth )
   **************************************************************************/
  // _stingRay->setBoundingBox()

  /**************************************************************************
   * This function can be used with a definition of
   * setInterpolationType( teInterpolationType eInterpolationType )
   * either F_LINEAR of F_NEAREST_NEIGHBOR
   **************************************************************************/
  // _stingRay->setInterpolationType( fvStingRayCaveRenderer::F_LINEAR )

  vvRenderer::renderVolumeGL();

  _lastRenderTime = sw->getTime();

  unsetGLenvironment();
}

//----------------------------------------------------------------------------
/// @return true if classification is done in no time
bool vvStingray::instantClassification() const
{
  vvDebugMsg::msg(3, "vvStingray::instantClassification()");
  return true;
}

//----------------------------------------------------------------------------
/** Update the color/alpha look-up table.
 Note: glColorTableSGI can have a maximum width of 1024 RGBA entries on IR2 graphics!
 @param dist  slice distance relative to 3D texture sample point distance
              (1.0 for original distance, 0.0 for all opaque).
*/
void vvStingray::updateLUT(float dist)
{
  vvDebugMsg::msg(1, "Generating texture LUT. Slice distance = ", dist);
  //cerr << "updateLUT called" << endl;
  float corr[4];                                  // gamma/alpha corrected RGBA values [0..1]
  int lutEntries;                                 // number of entries in the RGBA lookup table
  int i,c;

  lutEntries = 256;
  assert(lutEntries <= 4096);                     // rgbaTF and rgbaLUT are limited to this size

  // Copy LUT entries while gamma and opacity correcting:
  for (i=0; i<lutEntries; ++i)
  {
    // Gamma correction:
    if (gammaCorrection)
    {
      corr[0] = gammaCorrect(rgbaTF[i * 4],     VV_RED);
      corr[1] = gammaCorrect(rgbaTF[i * 4 + 1], VV_GREEN);
      corr[2] = gammaCorrect(rgbaTF[i * 4 + 2], VV_BLUE);
    }
    else
    {
      corr[0] = rgbaTF[i * 4];
      corr[1] = rgbaTF[i * 4 + 1];
      corr[2] = rgbaTF[i * 4 + 2];
    }

    corr[3] = rgbaTF[i * 4 + 3];

    // Convert float to uchar and copy to rgbaLUT array:
    for (c=0; c<4; ++c)
    {
      rgbaLUT[i * 4 + c] = (uchar)(corr[c] * 255.0f);
    }
  }

  // Copy LUT to graphics card:
  _stingRay->setLUT(rgbaLUT);
}
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
