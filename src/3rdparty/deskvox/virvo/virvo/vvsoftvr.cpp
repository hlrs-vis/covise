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
using std::cerr;
using std::endl;

#include "vvplatform.h"

#include <string.h>

#ifndef VV_REMOTE_RENDERING
#include "vvopengl.h"
#endif

#include <assert.h>
#include <math.h>
#include "gl/util.h"
#include "private/vvlog.h"
#include "vvdebugmsg.h"
#include "vvsoftimg.h"
#include "vvsoftvr.h"
#include "vvclock.h"
#include "vvimage.h"
#include "vvvoldesc.h"
#include "vvtoolshed.h"
#include "vvvecmath.h"

#include "private/vvgltools.h"

namespace gl = virvo::gl;
using virvo::mat4;
using virvo::vec3;
using virvo::vec4;

//----------------------------------------------------------------------------
/// Constructor.
vvSoftVR::vvSoftVR(vvVolDesc* vd, vvRenderState rs) : vvRenderer(vd, rs)
    , principal(virvo::cartesian_axis< 3 >::X)
{
   int i;

   vvDebugMsg::msg(1, "vvSoftVR::vvSoftVR()");

   // Initialize variables:
   xClipNormal = vec3(0.0f, 0.0f, 1.0f);
   xClipDist = 0.0f;
   numProc = vvToolshed::getNumProcessors();
   len[0] = len[1] = len[2] = 0;
   compression = false;
   multiprocessing = false;
   sliceInterpol = true;
   warpInterpol = true;
   sliceBuffer = true;
   bilinLookup = false;
   opCorr = false;
   earlyRayTermination = 0;
   oldQuality = 1.f;
   quality = 1.f;
   _timing = false;
   _size = vd->getSize();

   //  setWarpMode(SOFTWARE);     // initialize warp mode
   setWarpMode(TEXTURE);

   // Create output image size:
   outImg = NULL;
   vWidth = vHeight = -1;
   setOutputImageSize();

   // Generate x and y axis representations:
   for (i=0; i<3; ++i)
   {
      raw[i] = NULL;
      rle[i] = NULL;
      rleStart[i] = NULL;
   }
   findAxisRepresentations();
   //  encodeRLE();

   if (vd->getBPV() != 1)
   {
      cerr << "Shear-warp renderer can only display 8 bit scalar datasets." << endl;
   }

   // Generate color LUTs:
   updateTransferFunction();
}


//----------------------------------------------------------------------------
/// Destructor.
vvSoftVR::~vvSoftVR()
{
   int i;

   vvDebugMsg::msg(1, "vvSoftVR::~vvSoftVR()");

   delete outImg;
   delete intImg;
   for (i=0; i<3; ++i)
      delete[] raw[i];
}


//----------------------------------------------------------------------------
// See description in superclass.
void vvSoftVR::renderVolumeGL()
{
#ifndef VV_REMOTE_RENDERING
   GLint matrixMode;                              // current OpenGL matrix mode
   vvStopwatch* sw = NULL;                        // stop watch
                                                  // rendering times
   float preparation=0.0f, compositing=0.0f, warp=0.0f, total=0.0f;
   bool result;

   vvDebugMsg::msg(3, "vvSoftPer::renderVolumeGL()");

   if (vd->getBPV() != 1) return;                      // TODO: should work with all color depths

   if (_timing)
   {
      sw = new vvStopwatch();
      sw->start();
   }

   if(oldQuality != _quality)
   {
     setQuality(_quality);
     oldQuality = _quality;
   }


   // Memorize current OpenGL matrix mode and modelview matrix
   // because prepareRendering modifies the modelview matrix:
   glGetIntegerv(GL_MATRIX_MODE, &matrixMode);
   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();

   result = prepareRendering();

   // Undo translation:
   glMatrixMode(GL_MODELVIEW);
   glPopMatrix();
   glMatrixMode(matrixMode);                      // restore matrix mode

   if (!result)
   {
      delete sw;
      return;
   }

   if (_timing)
   {
      preparation = sw->getTime();
   }

   compositeVolume();

   if (_timing)
   {
      compositing = sw->getTime() - preparation;
   }

   if (vvDebugMsg::isActive(3))
      compositeOutline();

   if (_boundaries)
                                                  // draw back boundaries
      drawBoundingBox(_size, vd->pos, _boundColor /*FIXME:, false*/);

   if (warpMode==SOFTWARE)
   {
      outImg->warp(ivWarp, intImg);
      if (vvDebugMsg::isActive(3))
         outImg->overlay(intImg);
      outImg->draw();
   }
   else
   {
      intImg->warpTex(iwWarp);
      if (vvDebugMsg::isActive(3))
         intImg->draw();
   }

   if (_boundaries)
                                                  // draw front boundaries
      drawBoundingBox(_size, vd->pos, _boundColor /*FIXME:, true*/);

   if (_timing)
   {
      total = sw->getTime();
      warp = total - compositing - preparation;
      cerr << "Times [ms]: prep=" << (preparation*1000.0f) <<
         ", comp=" << (compositing*1000.0f) << ", warp=" <<
         (warp * 1000.0f) << ", total=" << (total*1000.0f) << endl;
      delete sw;
   }

   vvRenderer::renderVolumeGL();                  // draw coordinate axes
#endif
}


//----------------------------------------------------------------------------
/** Render the outline of the volume to the intermediate image.
  The shear transformation matrices have to be computed before calling this method.
*/
void vvSoftVR::compositeOutline()
{
   vec3 vertex[8];
   int vert[12][2] =                              // volume edge point indices
   {
      { 0,1 },
      { 1,2 },
      { 2,3 },
      { 3,0 },
      { 4,5 },
      { 5,6 },
      { 6,7 },
      { 7,4 },
      { 0,4 },
      { 1,5 },
      { 2,6 },
      { 3,7 }
   };
   // color components (RGB) for lines
   uchar col[12][3] =
   {
      { 255,0,0 },
      { 0,255,0 },
      { 0,0,255 },
      {0,255,255},
      { 255,0,255 },
      { 255,255,0 },
      {127,0,0},
      { 0,127,0 },
      { 0,0,127 },
      { 0,127,127 },
      { 127,0,127 },
      { 127,127,0 }
   };
   int i;
   int x1,y1,x,y;

   vvDebugMsg::msg(3, "vvSoftPar::compositeOutline()");

   // Compute vertices:
   for (i=0; i<8; ++i)
   {
      // Generate volume corners:
      vertex[i][0] = (float)(((i+1)/2) % 2);
      vertex[i][1] = (float)((i/2) % 2);
      vertex[i][2] = (float)((i/4) % 2);
      vertex[i] -= 0.5f;                        // vertices become -0.5 or +0.5
      vertex[i] *= _size;                     // vertices are scaled to correct object space coordinates
      vertex[i] = ( oiShear * vec4(vertex[i], 1.0f) ).xyz();
   }

   // Draw lines:
   for (i=0; i<12; ++i)
   {
      x  = (int)vertex[vert[i][0]][0];
      y  = (int)vertex[vert[i][0]][1];
      x1 = (int)vertex[vert[i][1]][0];
      y1 = (int)vertex[vert[i][1]][1];
      intImg->drawLine(x, y, x1, y1, col[i][0],col[i][1],col[i][2]);
   }
}


//----------------------------------------------------------------------------
/// Set new values for output image if necessary
void vvSoftVR::setOutputImageSize()
{
#ifndef VV_REMOTE_RENDERING
   GLint viewport[4];                             // OpenGL viewport information (position and size)

   vvDebugMsg::msg(3, "vvSoftVR::setOutputImageSize()");
   glGetIntegerv(GL_VIEWPORT, viewport);

   if (vWidth>0 && vHeight>0 &&
      vWidth==viewport[2] && vHeight==viewport[3])// already done?
      return;
   vWidth = viewport[2];
   vHeight = viewport[3];
   vvDebugMsg::msg(1, "Window dimensions: ", vWidth, vHeight);
   if (vWidth<1 || vHeight<1) vWidth = vHeight = 1;
   if (outImg != NULL) delete outImg;
   outImg = new vvSoftImg(vWidth, vHeight);

   findViewportMatrix(vWidth, vHeight);
#endif
}


//----------------------------------------------------------------------------
/// Gets the volume dimensions to standard object space
void vvSoftVR::findVolumeDimensions()
{
   vvDebugMsg::msg(3, "vvSoftVR::findVolumeDimensions()");

   switch (principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         len[0] = vd->vox[1];
         len[1] = vd->vox[2];
         len[2] = vd->vox[0];
         break;
      case virvo::cartesian_axis< 3 >::Y:
         len[0] = vd->vox[2];
         len[1] = vd->vox[0];
         len[2] = vd->vox[1];
         break;
      case virvo::cartesian_axis< 3 >::Z:
      default:
         len[0] = vd->vox[0];
         len[1] = vd->vox[1];
         len[2] = vd->vox[2];
         break;
   }

   vvDebugMsg::msg(3, "Permuted volume dimensions are: ", len[0], len[1], len[2]);
}


//----------------------------------------------------------------------------
/** Generate raw volume data for the principal axes.
  @param data uchar data array of scalar values (need to be copied)
*/
void vvSoftVR::findAxisRepresentations()
{
   size_t frameSize;                              // number of bytes per frame
   size_t sliceVoxels;                            // number of voxels per slice
   size_t offset;                                 // unit: voxels
   size_t srcIndex;                               // unit: bytes
   uint8_t* data;

   vvDebugMsg::msg(3, "vvSoftVR::findAxisRepresentations()");

   frameSize    = vd->getFrameBytes();
   sliceVoxels  = vd->getSliceVoxels();
   data = vd->getRaw();

   // Raw data for z axis view:
   delete[] raw[2];
   raw[2] = new uint8_t[frameSize];
   memcpy(raw[2], data, frameSize);

   // Raw data for x axis view:
   delete[] raw[0];
   raw[0] = new uint8_t[frameSize];
   size_t i=0;
   for (ptrdiff_t x=vd->vox[0]-1; x>=0; --x)                // counts slices in x axis view
      for (ssize_t z=0; z<vd->vox[2]; ++z)                  // counts height in x axis view
   {
      offset = z * sliceVoxels + x;
      for (ptrdiff_t y=vd->vox[1]-1; y>=0; --y)
      {
         srcIndex = (y * vd->vox[0] + offset) * vd->getBPV();
         for (size_t c=0; c<vd->getBPV(); ++c)
         {
            raw[0][i] = data[srcIndex + c];
            ++i;
         }
      }
   }

   // Raw data for y axis view:
   if (raw[1]!=NULL) delete[] raw[1];
   raw[1] = new uint8_t[frameSize];
   i=0;
   for (ssize_t y=0; y<vd->vox[1]; ++y)
   {
      for (ptrdiff_t x=vd->vox[0]-1; x>=0; --x)
   {
      offset = x + y * vd->vox[0];
      for (ptrdiff_t z=vd->vox[2]-1; z>=0; --z)
      {
         srcIndex = (offset + z * sliceVoxels) * vd->getBPV();
         for (size_t c=0; c<vd->getBPV(); ++c)
         {
            raw[1][i] = data[srcIndex + c];
            ++i;
         }
      }
   }
   }
}


//----------------------------------------------------------------------------
/** Run length encode the volume data.
  Encoding scheme: X is first byte.<UL>
  <LI>if X>0: copy next X voxels</LI>
  <LI>if X<0: repeat next voxel X times</LI>
  <LI>if X=0: done</LI></UL>
  Runs of same voxels must contain at least 3 voxels.
*/
void vvSoftVR::encodeRLE()
{
   size_t lineSize;                               // number of bytes per line
   uint8_t* src;                                  // pointer to unencoded array
   uint8_t* dst;                                  // pointer to encoded array
   int i,j;
   size_t len;                                    // length of encoded data [bytes]
   size_t rest;                                   // number of bytes remaining in buffer
   virvo::vector< 3, size_t > numvox;             // object dimensions (x,y,z) [voxels]

   vvDebugMsg::msg(1, "vvSoftVR::encodeRLE()");

   if (vd->getBPV() != 1) return;                      // TODO: enhance for other data types

   numvox = virvo::vector< 3, size_t >(vd->vox);

   vvDebugMsg::msg(1, "Original volume size: ", static_cast<int>(vd->getFrameBytes()));

   // Prepare 3 sets of compressed data, one for each principal axis:
   for (i=0; i<3; ++i)
   {
      delete[] rleStart[i];
      delete[] rle[i];
      rle[i] = new uchar[vd->getFrameBytes()];     // reserve as much RAM as in uncompressed case
      rleStart[i] = new uchar*[numvox[i] * numvox[(i+2)%3]];
   }

   // Now compress the data:
   for (i=0; i<3; ++i)
   {
      src = raw[i];
      dst = rle[i];
      rest = vd->getFrameBytes();
      lineSize = numvox[(i+1)%3];

      for (size_t a=0; a<numvox[i]; ++a)
         for (size_t b=0; b<numvox[(i+2)%3]; ++b)
      {
         rleStart[i][a * numvox[(i+2)%3] + b] = dst;
         vvToolshed::encodeRLE(dst, src, numvox[(i+1)%3], vd->getBPV(), rest, &len);
         dst  += len;
         rest -= len;
         src  += lineSize;
      }
      if (vvDebugMsg::isActive(1))
      {
         cerr << "Compressed size: " << dst-rle[i] << " = " << (dst-rle[i])*100.0/vd->getFrameBytes() << " %" << endl;
         cerr << "rest = " << rest << endl;
      }
      if (rest<=0)
      {
         for (j=0; j<3; ++j)
         {
            delete[] rleStart[j];
            rleStart[j] = NULL;
         }
         vvDebugMsg::msg(1, "RLE compression ineffective");
         return;
      }
   }
}


//----------------------------------------------------------------------------
// See parent for comments.
void vvSoftVR::updateTransferFunction()
{
   vvDebugMsg::msg(1, "vvSoftVR::updateTransferFunction()");

   int lutEntries = getLUTSize();

   vd->computeTFTexture(lutEntries, 1, 1, rgbaTF);

   updateLUT(1.f);
}

void vvSoftVR::updateLUT(float dist)
{
   vvDebugMsg::msg(1, "vvSoftVR::updateLUT()", dist);

   const int lutEntries = getLUTSize();
   // Copy RGBA values to internal array:
   for (int i=0; i<lutEntries; ++i)
      for (int c=0; c<4; ++c)
         rgbaConv[i][c] = (uchar)(rgbaTF[i*4+c] * 255.0f);

   // Make pre-integrated LUT:
   if (_preIntegration)
   {
      //makeLookupTextureOptimized(dist);           // use this line for fast pre-integration LUT
      makeLookupTextureCorrect(dist);   // use this line for slow but more correct pre-integration LUT
   }
}


//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs rather slow compared to
  makeLookupTextureOptimized because it does a correct applications of
  the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvSoftVR::makeLookupTextureCorrect(float thickness)
{
   vd->tf[0].makePreintLUTCorrect(PRE_INT_TABLE_SIZE, &preIntTable[0][0][0], thickness);
}


//----------------------------------------------------------------------------
/** Creates the look-up table for pre-integrated rendering.
  This version of the code runs much faster than makeLookupTextureCorrect
  due to some minor simplifications of the volume rendering integral.
  This method is
 * Copyright (C) 2001  Klaus Engel   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
@author Klaus Engel

I would like to thank Martin Kraus who helped in the adaptation
of the pre-integration method to Virvo.
@param thickness  distance of two volume slices in the direction
of the principal viewing axis (defaults to 1.0)
*/
void vvSoftVR::makeLookupTextureOptimized(float thickness)
{
   vd->tf[0].makePreintLUTOptimized(PRE_INT_TABLE_SIZE, &preIntTable[0][0][0], thickness);
}


//----------------------------------------------------------------------------
// See parent for comments.
void vvSoftVR::updateVolumeData()
{
   findAxisRepresentations();
}


//----------------------------------------------------------------------------
// See parent for comments
bool vvSoftVR::instantClassification() const
{
   vvDebugMsg::msg(3, "vvSoftVR::instantClassification()");
   return true;
}


//----------------------------------------------------------------------------
/** Compute number of entries in RGBA LUT.
  @return the number of entries in the RGBA lookup table.
*/
int vvSoftVR::getLUTSize()
{
   vvDebugMsg::msg(2, "vvSoftVR::getLUTSize()");
   return (vd->getBPV()==2) ? 4096 : 256;
}


//----------------------------------------------------------------------------
/** Compute view matrix.
  owView = glPM x glMV
*/
void vvSoftVR::findViewMatrix()
{
    mat4 mv = gl::getModelviewMatrix();
    mat4 pr = gl::getProjectionMatrix();

    owView = pr * mv;

    VV_LOG(3)  << "owView: " << owView;
}


//----------------------------------------------------------------------------
/** Generates the permutation matrix which transforms object space
  into standard object space.
  The permutation matrix is chosen among three choices, depending on the
  current principal viewing axis.
*/
void vvSoftVR::findPermutationMatrix()
{
   vvDebugMsg::msg(3, "vvSoftVR::findPermutationMatrix()");

   for (size_t i = 0; i < 4; ++i)
   {
      for (size_t j = 0; j < 4; ++j)
      {
         osPerm(i, j) = 0.0f;
      }
   }

   switch (principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         osPerm(0, 1) = 1.0f;
         osPerm(1, 2) = 1.0f;
         osPerm(2, 0) = 1.0f;
         osPerm(3, 3) = 1.0f;
         break;
      case virvo::cartesian_axis< 3 >::Y:
         osPerm(0, 2) = 1.0f;
         osPerm(1, 0) = 1.0f;
         osPerm(2, 1) = 1.0f;
         osPerm(3, 3) = 1.0f;
         break;
      case virvo::cartesian_axis< 3 >::Z:
      default:
         osPerm(0, 0) = 1.0f;
         osPerm(1, 1) = 1.0f;
         osPerm(2, 2) = 1.0f;
         osPerm(3, 3) = 1.0f;
         break;
   }
   VV_LOG(3) << "osPerm: " << osPerm;
}


//----------------------------------------------------------------------------
/** Compute conversion matrix from world space to OpenGL viewport space.
  This method only requires the size of the OpenGL viewport and
  can thus be called only when the user resizes the OpenGL window.
  Goal: invert y coordinate, scale, and translate origin to image center.<BR>
  Required 2D matrix:
  <PRE>
    w/2  0   0   w/2
    0   h/2  0   h/2
    0    0   1    0
    0    0   0    1
  </PRE>
*/
void vvSoftVR::findViewportMatrix(int w, int h)
{
   vvDebugMsg::msg(2, "vvSoftVR::findViewportMatrix()");

   wvConv = mat4::identity();
   wvConv(0, 0) = (float)(w / 2);
   wvConv(0, 3) = (float)(w / 2);
   wvConv(1, 1) = (float)(h / 2);
   wvConv(1, 3) = (float)(h / 2);

   VV_LOG(3) << "wvConv: " << wvConv;

}


//----------------------------------------------------------------------------
/** Determine intermediate image positions of bottom left and top right
  corner of a slice.
  @param slice  current slice index
  @param start  bottom left corner
  @param end    top right corner (pass NULL if not required)
*/
void vvSoftVR::findSlicePosition(int slice, vec3* start, vec3* end)
{
   vec4 start4, end4;
   findSlicePosition(slice, &start4, end?&end4:NULL);
   vec3 tmp = start4.xyz() / start4.w;
   for (int i = 0; i < 3; ++i)
   {
     (*start)[i] = tmp[i];
   }
   if (end)
   {
     tmp = end4.xyz() / end4.w;
     for (int i = 0; i < 3; ++i)
     {
       (*end)[i] = tmp[i];
     }
   }
}


//----------------------------------------------------------------------------
/** Determine intermediate image positions of bottom left and top right
  corner of a slice.
  @param slice  current slice index
  @param start  bottom left corner
  @param end    top right corner (pass NULL if not required)
*/
void vvSoftVR::findSlicePosition(int slice, vec4* start, vec4* end)
{
   // Determine voxel coordinates in object space:
   switch (principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         (*start)[0] =  0.5f * _size[0] - (float)slice / (float)len[2] * _size[0];
         (*start)[1] = -0.5f * _size[1];
         (*start)[2] = -0.5f * _size[2];
         (*start)[3] = 1.;
         if (end)
         {
            (*end)[0]   =  (*start)[0];
            (*end)[1]   = -(*start)[1];
            (*end)[2]   = -(*start)[2];
            (*end)[3] = 1.;
         }
         break;
      case virvo::cartesian_axis< 3 >::Y:
         (*start)[0] = -0.5f * _size[0];
         (*start)[1] =  0.5f * _size[1] - (float)slice / (float)len[2] * _size[1];
         (*start)[2] = -0.5f * _size[2];
         (*start)[3] = 1.;
         if (end)
         {
            (*end)[0]   = -(*start)[0];
            (*end)[1]   =  (*start)[1];
            (*end)[2]   = -(*start)[2];
            (*end)[3] = 1.;
         }
         break;
      case virvo::cartesian_axis< 3 >::Z:
         (*start)[0] = -0.5f * _size[0];
         (*start)[1] = -0.5f * _size[1];
         (*start)[2] =  0.5f * _size[2] - (float)slice / (float)len[2] * _size[2];
         (*start)[3] = 1.;
         if (end)
         {
            (*end)[0]   = -(*start)[0];
            (*end)[1]   = -(*start)[1];
            (*end)[2]   =  (*start)[2];
            (*end)[3] = 1.;
         }
         break;
   }

   // Project bottom left voxel of this slice to intermediate image:
   *start = oiShear * *start;
   if (end)
      *end = oiShear * *end;
}


//----------------------------------------------------------------------------
/** Compute the clipping plane equation in the permuted voxel coordinate system.
 */
void vvSoftVR::findClipPlaneEquation()
{
   mat4 oxConv;                                  // conversion matrix from object space to permuted voxel space
   mat4 xxPerm;                                  // coordinate permutation matrix
   vec3 planePoint;                              // point on clipping plane = starting point of normal
   vec3 normalPoint;                             // end point of normal

   vvDebugMsg::msg(3, "vvSoftVR::findClipPlaneEquation()");

   // Compute conversion matrix:
   oxConv = mat4::identity();
   oxConv(0, 0) =  (float)vd->vox[0]  / _size[0];
   oxConv(1, 1) = -(float)vd->vox[1] / _size[1]; // negate because y coodinate points down
   oxConv(2, 2) = -(float)vd->vox[2] / _size[2]; // negate because z coordinate points back
   oxConv(0, 3) =  (float)vd->vox[0]  / 2.0f;
   oxConv(1, 3) =  (float)vd->vox[1] / 2.0f;
   oxConv(2, 3) =  (float)vd->vox[2] / 2.0f;

   // Find coordinate permutation matrix:
   for (size_t i = 0; i < 4; ++i)
   {
      for (size_t j = 0; j < 4; ++j)
      {
         xxPerm(i, j) = 0.0f;
      }
   }

   switch (principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         xxPerm(0, 1) =-1.0f;
         xxPerm(0, 3) = (float)vd->vox[1];
         xxPerm(1, 2) = 1.0f;
         xxPerm(2, 0) =-1.0f;
         xxPerm(2, 3) = (float)vd->vox[0];
         xxPerm(3, 3) = 1.0f;
         break;
      case virvo::cartesian_axis< 3 >::Y:
         xxPerm(0, 2) =-1.0f;
         xxPerm(0, 3) = (float)vd->vox[2];
         xxPerm(1, 0) =-1.0f;
         xxPerm(1, 3) = (float)vd->vox[0];
         xxPerm(2, 1) = 1.0f;
         xxPerm(3, 3) = 1.0f;
         break;
      case virvo::cartesian_axis< 3 >::Z:
      default:
         xxPerm(0, 0) = 1.0f;
         xxPerm(1, 1) = 1.0f;
         xxPerm(2, 2) = 1.0f;
         xxPerm(3, 3) = 1.0f;
         break;
   }

   // Find two points determining the plane:
   planePoint = getParameter(VV_CLIP_PLANE_POINT);
   normalPoint = getParameter(VV_CLIP_PLANE_POINT);
   normalPoint += getParameter(VV_CLIP_PLANE_NORMAL).asVec3f();

   // Transfer points to voxel coordinate system:
   planePoint = ( oxConv * vec4(normalPoint, 1.0f) ).xyz();
   normalPoint = ( oxConv * vec4(normalPoint, 1.0f) ).xyz();

   // Permute the points:
   planePoint = ( xxPerm * vec4(planePoint, 1.0f) ).xyz();
   normalPoint = ( xxPerm * vec4(planePoint, 1.0f) ).xyz();

   // Compute plane equation:
   xClipNormal = normalPoint;
   xClipNormal -= planePoint;
   xClipNormal = normalize(xClipNormal);
   xClipDist = dot(xClipNormal, planePoint);
}


//----------------------------------------------------------------------------
/** Tests if a voxel [permuted voxel space] is clipped by the clipping plane.
  @param x,y,z  voxel coordinates
  @returns true if voxel is clipped, false if it is visible
*/
bool vvSoftVR::isVoxelClipped(int x, int y, int z)
{
   vvDebugMsg::msg(3, "vvSoftVR::isClipped()");

   if (!getParameter(VV_CLIP_MODE)) return false;

   if (xClipNormal[0] * (float)x + xClipNormal[1] *
      (float)y + xClipNormal[2] * (float)z > xClipDist)
      return true;
   else
      return false;
}


//----------------------------------------------------------------------------
/** Set warp mode.
  @param warpMode find valid warp modes in enum WarpType
*/
void vvSoftVR::setWarpMode(WarpType wm)
{
   vvDebugMsg::msg(3, "vvSoftVR::setWarpMode()");
   warpMode = wm;
}


//----------------------------------------------------------------------------
/** Get curernt warp mode.
  @return current warp mode
*/
vvSoftVR::WarpType vvSoftVR::getWarpMode()
{
   vvDebugMsg::msg(3, "vvSoftVR::getWarpMode()");
   return warpMode;
}


//----------------------------------------------------------------------------
/** Set new frame number.
  Additionally to the call to the superclass, the axis representations
  have to be re-computed.
  @see vvRenderer#setCurrentFrame(int)
*/
void vvSoftVR::setCurrentFrame(size_t index)
{
   vvDebugMsg::msg(3, "vvSoftVR::setCurrentFrame()");
   vvRenderer::setCurrentFrame(index);
   findAxisRepresentations();
}


//----------------------------------------------------------------------------
/** Return intermediate image in vvImage format.
  @param img returned intermediate image
*/
void vvSoftVR::getIntermediateImage(vvImage* image)
{
   vvDebugMsg::msg(3, "vvSoftVR::getIntermediateImage()", intImg->width, intImg->height);

   image->setNewImage(short(intImg->width), short(intImg->height), intImg->data);
}


//----------------------------------------------------------------------------
/** Return warp matrix iwWarp (intermediate image space to world space).
  @param iwWarp matrix which will be set to the warp matrix
*/
mat4 vvSoftVR::getWarpMatrix() const
{
    return iwWarp;
}


//----------------------------------------------------------------------------
/** Compute the bounding box of the slice data on the intermediate image.
  @param xmin,xmax returned minimum and maximum pixel index horizontally [0..image_width-1]
  @param ymin,ymax returned minimum and maximum pixel index vertically [0..image_height-1]
*/
void vvSoftVR::getIntermediateImageExtent(int* xmin, int* xmax, int* ymin, int* ymax)
{
   vec3 corner[4];                                // corners of first and last voxel slice on intermediate image
   int i;

   findSlicePosition(0, &corner[0], &corner[1]);
   findSlicePosition(len[2], &corner[2], &corner[3]);

   *xmin = (int)corner[0][0];
   *ymin = (int)corner[0][1];
   *xmax = int(corner[0][0]) + 1;
   *ymax = int(corner[0][1]) + 1;

   for (i=1; i<4; ++i)                            // loop thru rest of array
   {
      *xmin = ts_min(*xmin, (int)corner[i][0]);
      *ymin = ts_min(*ymin, (int)corner[i][1]);
      *xmax = ts_max(*xmax, int(corner[i][0]) + 1);
      *ymax = ts_max(*ymax, int(corner[i][1]) + 1);
   }
   *xmin = ts_clamp(*xmin, 0, intImg->width-1);
   *xmax = ts_clamp(*xmax, 0, intImg->width-1);
   *ymin = ts_clamp(*ymin, 0, intImg->height-1);
   *ymax = ts_clamp(*ymax, 0, intImg->height-1);
}


//----------------------------------------------------------------------------
/** Prepare the rendering of the intermediate image: check projection type,
    factor view matrix, etc.
    @return true if preparation was ok, false if an error occurred
*/
bool vvSoftVR::prepareRendering()
{
   vvDebugMsg::msg(3, "vvSoftVR::prepareRendering()");

   // Translate object by its position:
   mat4 trans = mat4::identity();                // translation matrix
   trans = translate( trans, vec3(vd->pos[0], vd->pos[1], vd->pos[2]) );
   mat4 mv = gl::getModelviewMatrix();
   mv = mv * trans;
   gl::setModelviewMatrix(mv);

   // Make sure a parallel projection matrix is used:
   vvMatrix pm = gl::getProjectionMatrix();
   if (rendererType==SOFTPAR && !pm.isProjOrtho())
   {
      vvDebugMsg::msg(1, "Parallel projection matrix expected! Rendering aborted.");
      return false;
   }
   else if (rendererType==SOFTPER && pm.isProjOrtho())
   {
      vvDebugMsg::msg(1, "Perspective projection matrix expected! Rendering aborted.");
      return false;
   }

   if (rendererType==SOFTPER)
   {
      // Cull object if behind viewer:
      if (getCullingStatus(pm.getNearPlaneZ()) == -1)
      {
         cerr << "culled" << endl;
         return false;
      }
   }

   setOutputImageSize();
   findViewMatrix();
   factorViewMatrix();                            // do the factorization
   findVolumeDimensions();                        // precompute the permuted volume dimensions
   if (getParameter(VV_CLIP_MODE)) findClipPlaneEquation();         // prepare clipping plane processing

   // Set interpolation types:
   intImg->setWarpInterpolation(warpInterpol);

   // Validate computed matrices:
   if (vvDebugMsg::isActive(3))
   {
      vvMatrix ovTest = vvMatrix(oiShear);
      vvMatrix ovView;
      ovTest.multiplyLeft(ivWarp);
      ovTest.print("ovTest = ivWarp x siShear x osPerm");
      ovView = vvMatrix(owView);
      ovView.multiplyLeft(wvConv);
      ovView.print("ovView = wvConv x owView");
   }
   return true;
}


//----------------------------------------------------------------------------
/** Get the volume object's culling status respective to near plane.
  A boundary sphere is used for the test, thus it is not 100% exact.
  @param nearPlaneZ  z coordinate of near plane
  @return <UL>
          <LI> 1  object is entirely in front of the near plane (=visible)</LI>
          <LI> 0  object is partly in front and partly behind the near plane</LI>
          <LI>-1  object is entirely behind the near plane (=invisible)</LI>
          </UL>
*/
int vvSoftVR::getCullingStatus(float nearPlaneZ)
{
   vvDebugMsg::msg(3, "vvSoftVR::getCullingStatus()");

   vec3 nearNormal;                               // normal vector of near plane
   vec3 volPos;                                   // volume position
   float     radius;                              // bounding sphere radius
   float     volDist;                             // distance of volume from near plane

   // Generate plane equation for near plane (distance value = nearPlaneZ):
   nearNormal = vec3(0.0f, 0.0f, 1.0f);

   // Find bounding sphere radius:
   radius = length(_size) / 2.0f;

   // Find volume midpoint location:
   mat4 mv = gl::getModelviewMatrix();
   volPos = ( mv * vec4(0.0f, 0.0f, 0.0f, 1.0f) ).xyz();

   // Apply plane equation to volume midpoint:
   volDist = dot(nearNormal, volPos) - nearPlaneZ;

   if (fabs(volDist) < radius) return 0;
   else if (volDist < 0) return 1;
   else return -1;
}


//----------------------------------------------------------------------------
// see parent
void vvSoftVR::setParameter(ParameterType param, const vvParam& value)
{
   vvDebugMsg::msg(3, "vvSoftVR::setParameter()");
   switch (param)
   {
      case vvRenderer::VV_SLICEINT:
         sliceInterpol = value;
         setQuality(_quality);
         break;
      case vvRenderer::VV_WARPINT:
         warpInterpol = value;
         break;
#if 0
      case vvRenderer::VV_COMPRESS:
         compression = value;
         break;
      case vvRenderer::VV_MULTIPROC:
         multiprocessing = value;
         break;
      case vvRenderer::VV_SLICEBUF:
         sliceBuffer = value;
         cerr << "sliceBuffer set to " << int(sliceBuffer) << endl;
         break;
      case vvRenderer::VV_LOOKUP:
         bilinLookup = value;
         cerr << "bilinLookup set to " << int(bilinLookup) << endl;
         break;
 #endif
      case vvRenderer::VV_PREINT:
         _preIntegration = value;
         if (_preIntegration) updateLUT(1.f);
         cerr << "preIntegration set to " << int(_preIntegration) << endl;
         break;
     case vvRenderer::VV_OPCORR:
         opCorr = value;
         cerr << "opCorr set to " << int(opCorr) << endl;
         break;
      default:
         vvRenderer::setParameter(param, value);
         break;
   }
}


//----------------------------------------------------------------------------
// see parent
vvParam vvSoftVR::getParameter(ParameterType param) const
{
   vvDebugMsg::msg(3, "vvSoftVR::getParameter()");

   switch (param)
   {
      case vvRenderer::VV_SLICEINT:
         return sliceInterpol;
      case vvRenderer::VV_WARPINT:
         return warpInterpol;
#if 0
      case vvRenderer::VV_COMPRESS:
         return compression;
      case vvRenderer::VV_MULTIPROC:
         return multiprocessing;
      case vvRenderer::VV_SLICEBUF:
         return sliceBuffer;
      case vvRenderer::VV_LOOKUP:
         return bilinLookup;
#endif
      case vvRenderer::VV_OPCORR:
         return opCorr;
      default:
         return vvRenderer::getParameter(param);
   }
}

void vvSoftVR::setQuality(float q)
{
  quality = 1.f;
  _quality = q;
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
