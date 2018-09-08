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

#include <math.h>
#include <assert.h>
#include "private/vvlog.h"
#include "vvdebugmsg.h"
#include "vvsoftper.h"
#include "vvsoftimg.h"
#include "vvclock.h"
#include "vvvoldesc.h"
#include "vvtoolshed.h"

using virvo::mat4;
using virvo::vec3;
using virvo::vec4;

//----------------------------------------------------------------------------
/// Constructor.
vvSoftPer::vvSoftPer(vvVolDesc* vd, vvRenderState rs) : vvSoftVR(vd, rs)
{
   vvDebugMsg::msg(1, "vvSoftPer::vvSoftPer()");

   rendererType = SOFTPER;

   intImg = new vvSoftImg(0, 0);

   setQuality(_quality);                           // not only sets quality but also resizes intermediate image
}


//----------------------------------------------------------------------------
/// Destructor.
vvSoftPer::~vvSoftPer()
{
   vvDebugMsg::msg(1, "vvSoftPer::~vvSoftPer()");
}


//----------------------------------------------------------------------------
/** Composite the volume slices to the intermediate image.
  The function prepareRendering() must be called before this method.
  The shear transformation matrices have to be computed before calling this method.
  The volume slices are processed from front to back.
  @param from,to optional arguments to define first and last intermediate image line to render.
                 if not passed, the entire intermediate image will be rendered
*/
void vvSoftPer::compositeVolume(int from, int to)
{
   int slice;                                     // currently processed slice
   int i;

   vvDebugMsg::msg(3, "vvSoftPer::compositeVolume(): ", from, to);

   intImg->clear();

   for (i=0; i<len[2]; ++i)                       // traverse volume slice by slice
   {
      // Determine slice index which depends on the stacking order:
      if (stacking) slice = i;
      else slice = len[2] - i - 1;

      // Composite slice according to current rendering mode:
      if (sliceInterpol) compositeSliceBilinear(slice, from, to);
      else compositeSliceNearest(slice, from, to);
   }
}


//----------------------------------------------------------------------------
/** Composite the voxels from one slice into the intermediate image.
  2D nearest neighbor interpolation is used.
  Naming convention for variables:<BR>
  i  = intermediate image space (coordinates are: u,v)<BR>
  v  = voxel data space<P>
  The compositing formula uses the UNDER operator
  (see Drebin et.al.: Volume Rendering):<BR>
  (a UNDER b) = (b OVER a)
  <PRE>
    vc = voxel color component [0..1]
    va = voxel alpha component (0=transparent) [0..1]
ic = intermediate image color component [0..1]
ia = invermediate image alpha component (0=transparent) [0..1]
UNDER operator:
ic := ia * ic + (1 - ia) * (vc * va)
ia := ia + va * (1 - ia)
</PRE>
@param slice   index of slice to composite [permuted value]
@param from    first intermediate image line to render (bottom-most line, -1 to render all lines)
@param to      last intermediate image line to render (top-most line)
*/
void vvSoftPer::compositeSliceNearest(int slice, int from, int to)
{
#define FLOAT_MATH                             // undefine for integer math
   vec3 vStart;                                   // bottom left voxel of the current slice
   vec3 vEnd;                                     // top right voxel of the current slice
   float  iStartX, iStartY;                       // coordinates of bottom left intermediate image pixel for this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates
   int    vPosX, vPosY;                           // current slice x and y coordinates [16.16 fixed point]
   int    vFracY;                                 // remainder of y step (always <0) [16.16 fixed point]
   int    vStepX, vStepY;                         // step size [16.16 fixed point]
   int    ix,iy;                                  // counters [intermediate image space]
   uchar* vScalar;                                // pointer to current scalar voxel data
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
#ifdef FLOAT_MATH
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   float  tmp;
   float  ia1;                                    // 1 - ia
#else
   uint   vr,vg,vb,va;                            // RGBA components of current voxel
   uint   ir,ig,ib,ia;                            // RGBA components of current image pixel
   uint   ia1;                                    // = 255-ia
#endif

   // Compute values which are constant in the compositing loop:
   findSlicePosition(slice, &vStart, &vEnd);
   iStartX     = vStart[0];
   iStartY     = vStart[1];
   iPosX       = vvToolshed::round(iStartX);
   iPosY       = vvToolshed::round(iStartY);
   iSlice[0]   = vvToolshed::round(vEnd[0] - vStart[0]);
   iSlice[1]   = vvToolshed::round(vEnd[1] - vStart[1]);

                                                  // return if slice doesn't fit in intermediate image
   if (iPosX<0 || iPosX+iSlice[0]>=intImg->width ||
      iPosY<0 || iPosY+iSlice[1]>=intImg->height ||
      iSlice[0]<=0 || iSlice[1]<=0)
      return;

   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0]);
   vScalar     = raw[principal] + vd->getBPV() * (slice * len[0] * len[1] + (len[1] - 1) * len[0]);
   vPosX       = 0;
   vPosY       = (len[1] - 1) << 16;
   vFracY      = 0;
   vStepX      = (len[0] << 16) / iSlice[0];      // 16.16 value
   vStepY      = (len[1] << 16) / iSlice[1];      // 16.16 value

   if (from != -1)                                // render only specific lines?
   {
      int iTopLine;                               // topmost intermediate image line

      iTopLine = iPosY + iSlice[1] - 1;
      if (to < iPosY || from > iTopLine) return;  // return if section to render is outside of slice area

      // Constrain section indices:
      if (from < iPosY) from = iPosY;
      if (to > iTopLine) to = iTopLine;

      // Modify first voxel to draw:
      for (int i=0; i<(from - iPosY); ++i)        // TODO: optimize this loop!
      {
         vFracY  += vStepY;
         vScalar -= (vFracY >> 16) * len[0] * vd->getBPV();
         vPosY   -= vStepY;
         vFracY  &= 0xffff;                       // delete integer part of 16.16 value
      }

      // Modify first pixel to draw:
      iPosY = from;

      // Modify number of lines to draw:
      iSlice[1] = to - from + 1;
   }

   // Compute starting values for values which are variable in the compositing loop:
   iPixel = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);

   // Traverse intermediate image pixels which correspond to the current slice:
   for (iy=0; iy<iSlice[1]; ++iy)
   {
      vPosX = 0;
      for (ix=0; ix<iSlice[0]; ++ix)
      {
         // Determine image color components and scale to [0..1]:
#ifdef FLOAT_MATH
         ir = (float)(*(iPixel++)) / 255.0f;
         ig = (float)(*(iPixel++)) / 255.0f;
         ib = (float)(*(iPixel++)) / 255.0f;
         ia = (float)(*(iPixel++)) / 255.0f;
#else
         ir = (int)(*(iPixel++));
         ig = (int)(*(iPixel++));
         ib = (int)(*(iPixel++));
         ia = (int)(*(iPixel++));
#endif
         if ((ia<255) &&                          // early ray termination for opaque image pixels
                                                  // skip clipped voxels
            (!getParameter(VV_CLIP_MODE) || (!isVoxelClipped(vPosX >> 16, vPosY >> 16, slice))))
         {
            // Determine voxel color components and scale to [0..1]:
#ifdef FLOAT_MATH
            vr = (float)rgbaConv[vScalar[vPosX >> 16]][0] / 255.0f;
            vg = (float)rgbaConv[vScalar[vPosX >> 16]][1] / 255.0f;
            vb = (float)rgbaConv[vScalar[vPosX >> 16]][2] / 255.0f;
            va = (float)rgbaConv[vScalar[vPosX >> 16]][3] / 255.0f;
#else
            vr = (int)rgbaConv[vScalar[vPosX >> 16]][0];
            vg = (int)rgbaConv[vScalar[vPosX >> 16]][1];
            vb = (int)rgbaConv[vScalar[vPosX >> 16]][2];
            va = (int)rgbaConv[vScalar[vPosX >> 16]][3];
#endif

            // Accumulate new intermediate image pixel values.
            iPixel -= 4;                          // start over with intermediate image components
#ifdef FLOAT_MATH
            ia1 = 1.0f - ia;
            *(iPixel++) = (uchar)((tmp = (255.0f * (ir + ia1 * vr * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ig + ia1 * vg * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ib + ia1 * vb * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ia + ia1 * va))) < 255.0f ? tmp : 255.0f);
#else
            ia1 = 0xFF - ia;
            // TODO: color model should be adapted to the one used in FLOAT_MATH!
            *(iPixel++) = (uchar)(((ia * ir) + (vr * ia1)) >> 8);
            *(iPixel++) = (uchar)(((ia * ig) + (vg * ia1)) >> 8);
            *(iPixel++) = (uchar)(((ia * ib) + (vb * ia1)) >> 8);
            *(iPixel++) = (uchar)(ia         + ((va * ia1) >> 8));
#endif
         }

         // Switch to next voxel:
         vPosX += vStepX;
      }
      iPixel  += iLineOffset;
      vFracY  += vStepY;
      vScalar -= (vFracY >> 16) * len[0] * vd->getBPV();
      vPosY   -= vStepY;
      vFracY  &= 0xffff;                          // delete integer part of 16.16 value
   }
}


//----------------------------------------------------------------------------
/** Composite the voxels from one slice into the intermediate image.
  Bilinear footprint interpolation is used.
  Two cases have to be distinguished:<UL>
  <LI>pixel distance < voxel distance (multiple voxels per pixel):
      use footprint resampling</LI>
  <LI>pixel distance > voxel distance (multiple pixels per voxel):
      use 4-voxel bilinear resampling</LI></UL>

  @param slice      index of slice to composite [permuted value]
  @param from    first intermediate image line to render (bottom-most line, -1 to render all lines)
  @param to      last intermediate image line to render (top-most line)
*/
void vvSoftPer::compositeSliceBilinear(int slice, int from, int to)
{
   vec3 vStart;                                   // bottom left voxel of the current slice
   vec3 vEnd;                                     // top right voxel of the current slice
   float  vPosYBase;                              // first slice y coordinate
   float  vStepX, vStepY;                         // step size: voxels traversed per image pixel
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   float  vPosX, vPosY;                           // current slice x and y coordinates
   float  ia1;                                    // = 1-ia
   uchar* vSliceBase;                             // pointer to top left (first) voxel in current slice
   uchar* iPixelBase;                             // pointer to first intermediate image pixel
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iPosX, iPosY;                           // current intermediate image coordinates
   int    iPixelLine;                             // number of bytes on a line in the intermediate image
   int    iSlice[2];                              // slice dimensions in intermediate image (width,height)
   int    ix;                                     // counter [intermediate image space]
   int    iy;                                     // counter [intermediate image space]
   bool   zoomMode;                               // true  = voxel slice smaller than image slice: accumulate voxels
   // false = voxel slice larger than image slice:  bilinearly interpolate
   float  tmp;

   // Compute slice position and size on intermediate image:
   findSlicePosition(slice, &vStart, &vEnd);
   iPosX       = vvToolshed::round(vStart[0]);
   iPosY       = vvToolshed::round(vStart[1]);
   iSlice[0]   = vvToolshed::round(vEnd[0] - vStart[0]);
   iSlice[1]   = vvToolshed::round(vEnd[1] - vStart[1]);

   // Check if slice fits to intermediate image. It should
   // always fit, because the intermediate image size was selected
   // for it to do. But for the viewer inside the volume
   // this is the only solution so far.
   if (iPosX<0 || iPosX+iSlice[0]>=intImg->width  ||
      iPosY<0 || iPosY+iSlice[1]>=intImg->height ||
      iSlice[0]<=0 || iSlice[1]<=0)
      return;

   // Initialize compositing parameters:
   iPixelLine = intImg->PIXEL_SIZE * intImg->width;
   vStepX      = (float)(len[0]-1) / (float)(iSlice[0]-1);
   vStepY      = (float)(len[1]-1) / (float)(iSlice[1]-1);
   vPosYBase   = (float)(len[1] - 1);
   if (vStepX<1.0f || vStepY<1.0f) zoomMode = false;
   else zoomMode = true;

   // Render only specific lines?
   if (from != -1)
   {
      int iTopLine;                               // topmost intermediate image line

      iTopLine = iPosY + iSlice[1] - 1;
      if (to < iPosY || from > iTopLine) return;  // return if section to render is outside of slice area

      // Constrain section indices:
      if (from < iPosY) from = iPosY;
      if (to > iTopLine) to = iTopLine;

      // Modify first voxel to draw:
      vPosYBase -= (from - iPosY) * vStepY;

      // Modify first pixel to draw:
      iPosY = from;

      // Modify number of lines to draw:
      iSlice[1] = to - from + 1;
   }

   // Compute starting values for values which are variable in the compositing loop:
   iPixelBase  = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);
   vSliceBase  = raw[principal] + vd->getBPV() * slice * len[0] * len[1];

   // Traverse intermediate image pixels which correspond to the current slice:
   for (iy=0; iy<iSlice[1]; ++iy)
   {
      vPosX  = 0.0f;
      vPosY  = vPosYBase  - iy * vStepY;
      iPixel = iPixelBase + iy * iPixelLine;
      for (ix=0; ix<iSlice[0]; ++ix)
      {
         // Determine image color components and scale to [0..1]:
         ir = (float)(*(iPixel++)) / 255.0f;
         ig = (float)(*(iPixel++)) / 255.0f;
         ib = (float)(*(iPixel++)) / 255.0f;
         ia = (float)(*(iPixel++)) / 255.0f;

         if ((ia<255) &&                          // early ray termination for opaque image pixels
                                                  // skip clipped voxels
            (!getParameter(VV_CLIP_MODE) || (!isVoxelClipped((int)vPosX, (int)vPosY, slice))))
         {
            // Determine voxel color components and scale to [0..1]:
            if (zoomMode)
               accumulateVoxels(vSliceBase, vPosX, vPosY, vStepX, vStepY, &vr, &vg, &vb, &va);
            else
               interpolateVoxels(vSliceBase, vPosX, vPosY, &vr, &vg, &vb, &va);

            // Accumulate new intermediate image pixel values.
            iPixel -= 4;                          // start over with intermediate image components
            // Color model suggested by Martin Kraus:
            ia1 = 1.0f - ia;
            *(iPixel++) = (uchar)((tmp = (255.0f * (ir + ia1 * vr * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ig + ia1 * vg * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ib + ia1 * vb * va))) < 255.0f ? tmp : 255.0f);
            *(iPixel++) = (uchar)((tmp = (255.0f * (ia + ia1 * va))) < 255.0f ? tmp : 255.0f);
            /*  Initial color model:
                    ia1 = 1.0f - ia;
                    *(iPixel++) = (uchar)(255.0f * (ia * ir + ia1 * vr));
                    *(iPixel++) = (uchar)(255.0f * (ia * ig + ia1 * vg));
                    *(iPixel++) = (uchar)(255.0f * (ia * ib + ia1 * vb));
                    *(iPixel++) = (uchar)(255.0f * (ia      + ia1 * va));
            */
         }

         // Switch to next voxel:
         vPosX += vStepX;
      }
   }
}


//----------------------------------------------------------------------------
/** Bilinearly interpolate voxels to gain an RGBA tuple.
  @param sliceBase  pointer to top left voxel in the current slice
  @param fx,fy    pixel location in voxel space
  @param r,g,b,a  accumulated RGBA return value
*/
void vvSoftPer::interpolateVoxels(uchar* sliceBase, float fx, float fy,
float* r, float* g, float* b, float* a)
{
   uchar* vScalar[4];                             // ptr to scalar data: 0=bot.left, 1=top left, 2=top right, 3=bot.right
   float  weight[4];                              // resampling weights, one for each of the four neighboring voxels (for indices see vScalar[])
   float  frac[2];                                // fractions for resampling (x,y)
   int    top;                                    // top voxel row to process
   int    left;                                   // left voxel column to process

   // Compute top left voxel location:
   left = (int)fx;
   top  = (int)fy;

   // No interpolation for border voxels:
   if (left<0 || left+1>=len[0] || top<0 || top+1>=len[1])
   {
      if (left<0) left=0;
      else if (left+1>=len[0]) left = len[0]-1;
      if (top<0) top=0;
      else if (top+1>=len[1]) top = len[1]-1;
      vScalar[0] = sliceBase + vd->getBPV() * (top * len[0] + left);

      *r = ((float)rgbaConv[*vScalar[0]][0]) / 255.0f;
      *g = ((float)rgbaConv[*vScalar[0]][1]) / 255.0f;
      *b = ((float)rgbaConv[*vScalar[0]][2]) / 255.0f;
      *a = ((float)rgbaConv[*vScalar[0]][3]) / 255.0f;
      return;
   }

   // Compute resampling fractions:
   frac[0] = fx - (float)left;
   frac[1] = fy - (float)top;

   // Compute bilinear resampling weights:
   weight[0] = (1.0f - frac[0]) * frac[1];
   weight[1] = (1.0f - frac[0]) * (1.0f - frac[1]);
   weight[2] = frac[0] * (1.0f - frac[1]);
   weight[3] = frac[0] * frac[1];

   // Compute pointers to scalar values:
   vScalar[1] = sliceBase + vd->getBPV() * (top * len[0] + left);
   vScalar[2] = vScalar[1] + vd->getBPV();
   vScalar[0] = vScalar[1] + vd->getBPV() * len[0];
   vScalar[3] = vScalar[0] + vd->getBPV();

   // Determine interpolated voxel color components and scale to [0..1]:
   *r = ((float)rgbaConv[*vScalar[0]][0] * weight[0] +
      (float)rgbaConv[*vScalar[1]][0] * weight[1] +
      (float)rgbaConv[*vScalar[2]][0] * weight[2] +
      (float)rgbaConv[*vScalar[3]][0] * weight[3]) / 255.0f;
   *g = ((float)rgbaConv[*vScalar[0]][1] * weight[0] +
      (float)rgbaConv[*vScalar[1]][1] * weight[1] +
      (float)rgbaConv[*vScalar[2]][1] * weight[2] +
      (float)rgbaConv[*vScalar[3]][1] * weight[3]) / 255.0f;
   *b = ((float)rgbaConv[*vScalar[0]][2] * weight[0] +
      (float)rgbaConv[*vScalar[1]][2] * weight[1] +
      (float)rgbaConv[*vScalar[2]][2] * weight[2] +
      (float)rgbaConv[*vScalar[3]][2] * weight[3]) / 255.0f;
   *a = ((float)rgbaConv[*vScalar[0]][3] * weight[0] +
      (float)rgbaConv[*vScalar[1]][3] * weight[1] +
      (float)rgbaConv[*vScalar[2]][3] * weight[2] +
      (float)rgbaConv[*vScalar[3]][3] * weight[3]) / 255.0f;
}


//----------------------------------------------------------------------------
/** Accumulate voxels to gain an RGBA tuple.
  @param sliceBase  pointer to top left voxel in the current slice
  @param fx,fy    footprint center location within slice
  @param fw,fh    footprint size (width,height)
  @param r,g,b,a  accumulated RGBA return value
*/
void vvSoftPer::accumulateVoxels(uchar* sliceBase, float fx, float fy, float fw, float fh,
float* r, float* g, float* b, float* a)
{
   uchar* scalar;                                 // pointer to current scalar value
   int    top;                                    // first voxel row to process
   int    left;                                   // first voxel column to process
   int    w, h;                                   // number of voxels to process in x and y direction
   int    x,y;                                    // currently processed voxel coordinates
   int    lineOffset;                             // voxel space line offset
   int    numVoxels;                              // total number of voxels accumulated

   // Determine top left voxel of accumulation area:
   left = int(ceilf(fx - fw/2.0f));
   top  = int(ceilf(fy - fh/2.0f));
   w    = int(fx + fw/2.0f) - left + 1;
   h    = int(fy + fh/2.0f) - top  + 1;

   // Constrain the values to be within the slice:
   if (left < 0)
   {
      w += left;
      left = 0;
   }
   if (top < 0)
   {
      h += top;
      top = 0;
   }
   if (left+w > len[0]) w = len[0] - left;
   if (top +h > len[1]) h = len[1] - top;
   assert(w>=0 && h>=0);

   // Accumulate voxels (compute average value):
   scalar = sliceBase + vd->getBPV() * (top * len[0] + left);
   lineOffset = (int)vd->getBPV() * len[0];
   *r = *g = *b = *a = 0.0f;
   numVoxels = w * h;
   for (y=0; y<h; ++y)
   {
      for (x=0; x<w; ++x)
      {
         *r += (float)rgbaConv[scalar[x]][0] / 255.0f;
         *g += (float)rgbaConv[scalar[x]][1] / 255.0f;
         *b += (float)rgbaConv[scalar[x]][2] / 255.0f;
         *a += (float)rgbaConv[scalar[x]][3] / 255.0f;
      }
      scalar += lineOffset;
   }
   *r /= (float)numVoxels;
   *g /= (float)numVoxels;
   *b /= (float)numVoxels;
   *a /= (float)numVoxels;
}


//----------------------------------------------------------------------------
/** Set rendering quality.
  When quality changes, the intermediate image must be resized and the diConv
  matrix has to be recomputed.
  @see vvRenderer#setQuality
*/
void vvSoftPer::setQuality(float q)
{
   const float SIZE_FACTOR = 2.0f;                // size factor for intermediate image size
   int intImgSize;                                // edge size of intermediate image [pixels]

   vvDebugMsg::msg(3, "vvSoftPer::setQuality()", q);

#ifdef VV_XVID
   q = 1.0f;
#endif
   _quality = q;

   intImgSize = (int)(SIZE_FACTOR * (2.0f * _quality) * ts_max(vd->vox[0], vd->vox[1], vd->vox[2]));
   if (intImgSize<1)
   {
      intImgSize = 1;
      _quality = 1.0f / (SIZE_FACTOR * 2.0f * ts_max(vd->vox[0], vd->vox[1], vd->vox[2]));
   }

   intImgSize = ts_clamp(intImgSize, 16, 4096);
   intImgSize = (int)vvToolshed::getTextureSize(intImgSize);

   intImg->setSize(intImgSize, intImgSize);
   findDIConvMatrix();
   vvDebugMsg::msg(3, "Intermediate image edge length: ", intImgSize);
}


//----------------------------------------------------------------------------
/** Create matrix for conversion from deformed (sheared) space to
    intermediate image coordinates.
  This function only requires the size of the intermediate image and
  can thus be called in the class constructor.
  Goal: shift right and down.<BR>
  Required 3D matrix:
  <PRE>
    1   0   0  w/2
    0   1   0  h/2
    0   0   1   0
    0   0   0   1
</PRE>
*/
void vvSoftPer::findDIConvMatrix()
{
   vvDebugMsg::msg(3, "vvSoftPer::findDIConvMatrix()");

   diConv = mat4::identity();
   diConv(0, 3) = (float)(intImg->width / 2);
   diConv(1, 3) = (float)(intImg->height / 2);

   VV_LOG(3) << "diConv: " << diConv;
}


//----------------------------------------------------------------------------
/// Find the eye position in object space.
void vvSoftPer::findOEyePosition()
{
   mat4 woView;                              // inverse of viewing transformation matrix

   vvDebugMsg::msg(3, "vvSoftPer::findOEyePosition()");

   // Find eye position in object space:
   oEye = vec4(0.0f, 0.0f, -1.0f, 0.0f);
   woView = inverse(owView);
   oEye = woView * oEye;
   VV_LOG(3) << "Eye position in object space: " << oEye;
}


//----------------------------------------------------------------------------
/** Computes the principal viewing axis.
  Strategy: Compute vectors from eye to volume vertices and find a
  principal axis for each vertex. Then use the most frequently occurring
  axis and the corresponding stacking order as output.<BR>
  Requires: eye position in object space
*/
void vvSoftPer::findPrincipalAxis()
{
    typedef virvo::cartesian_axis< 3 > axis_type;

   axis_type pa[8] =
   {
        axis_type::X, axis_type::X, axis_type::X, axis_type::X,
        axis_type::X, axis_type::X, axis_type::X, axis_type::X
   };                                             // principal axis for each volume vertex
   vec3 vertex;                                   // volume vertex
   vec3 dist;                                     // distance from eye to volume corner
   vec3 oEye3;                                    // eye position in object space
   float ax, ay, az;                              // absolute coordinate components
   float maxDist;                                 // maximum distance
   int   count[3];                                // occurrences of coordinate axes as principal viewing axis (0=x-axis etc)
   int   maxCount;                                // maximum counter value
   int   i;                                       // counter
   bool  stack[3] = {true, true, true };          // stacking order for coordinate axes
   vec3 size = vd->getSize();

   vvDebugMsg::msg(3, "vvSoftPer::findPrincipalAxis()");

   oEye3 = oEye.xyz() / oEye.w;                   // convert eye coordinates from vector4 to vector3

   // Find principal axes:
   for (i=0; i<8; ++i)
   {
      // Generate volume corners:
      vertex[0] = (float)(i % 2);
      vertex[1] = (float)((i/2) % 2);
      vertex[2] = (float)((i/4) % 2);
      vertex -= 0.5f;                           // vertices become -0.5 or +0.5
      vertex *= size;                         // vertices are scaled to correct object space coordinates

      // Compute distance between eye and corner:
      dist = vertex - oEye3;

      // Determine the principal viewing axis and the stacking order:
      ax = (float)fabs(dist[0]);
      ay = (float)fabs(dist[1]);
      az = (float)fabs(dist[2]);
      maxDist = ts_max(ax, ay, az);
      if      (maxDist==ax) { pa[i] = axis_type::X; stack[0] = (dist[0] < 0.0f); }
      else if (maxDist==ay) { pa[i] = axis_type::Y; stack[1] = (dist[1] < 0.0f); }
      else                  { pa[i] = axis_type::Z; stack[2] = (dist[2] < 0.0f); }
      vvDebugMsg::msg(3, "Found principal viewing axis (0=x, 1=y, 2=z): ", pa[i]);
   }

   // Find the dominating principal axis:
   for (i=0; i<3; ++i)
      count[i] = 0;
   for (i=0; i<8; ++i)
   {
      switch (pa[i])
      {
         case axis_type::X: ++count[0]; break;
         case axis_type::Y: ++count[1]; break;
         case axis_type::Z: ++count[2]; break;
         default: break;
      }
   }

   // Assign the dominant axis for the principal axis (favor the Z axis for ties):
   maxCount = ts_max(count[0], count[1], count[2]);
   if (maxCount==count[2])      { principal = axis_type::Z; stacking = stack[2]; }
   else if (maxCount==count[1]) { principal = axis_type::Y; stacking = stack[1]; }
   else                         { principal = axis_type::X; stacking = stack[0]; }

   if (vvDebugMsg::isActive(3)) cerr << "Principal axis: " << principal << endl;
   if (vvDebugMsg::isActive(3))
   {
      cerr << "Stacking order: ";
      if (stacking) cerr << "true" << endl;
      else cerr << "false" << endl;
   }
}


//----------------------------------------------------------------------------
/** Compute the shift matrix.
  The shift matrix translates the origin of the object coordinate
  system along the z axis if the eye is in the z=0 plane.
*/
void vvSoftPer::findShiftMatrix()
{
   vvDebugMsg::msg(3, "vvSoftPer::findShiftMatrix()");

   vec4 permEye = oEye;                           // object space eye position after permutation
   permEye = osPerm * permEye;

   shift = mat4::identity();
   if (permEye[2] == 0.0f)                        // is eye in z=0 plane?
   {
      // TODO: find algorithm when eye inside of volume
      // Shift eye position to other edge of volume to prevent
      // division by zero in shear:
      vec3 size = vd->getSize();
      shift = translate(shift, vec3(0.0f, 0.0f, -size[2]));
      cerr << "Eye inside of volume!" << endl;
   }
   VV_LOG(3) << "shift: " << shift;
}


//----------------------------------------------------------------------------
/// Find the eye position in standard object space.
void vvSoftPer::findSEyePosition()
{
   vvDebugMsg::msg(3, "vvSoftPer::findSEyePosition()");

   sEye = osPerm * oEye;
   sEye = shift * sEye;
   VV_LOG(3) << "Eye position in standard object space: " << sEye;
}


//----------------------------------------------------------------------------
/// Scale front voxel slice to 1:1 pixel:voxel ratio.
void vvSoftPer::findScaleMatrix()
{
   float sf;                                      // scale factor to make front slice 1:1 pixel:voxel

   vvDebugMsg::msg(3, "vvSoftPer::findScaleMatrix()");

   scale = mat4::identity();

   vec3 size = vd->getSize();
   switch(principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         scale = virvo::scale( scale, vec3(vd->vox[1] / size[1], vd->vox[2] / size[2], vd->vox[0] / size[0]) );
         break;
      case virvo::cartesian_axis< 3 >::Y:
         scale = virvo::scale( scale, vec3(vd->vox[2] / size[2], vd->vox[0] / size[0], vd->vox[1] / size[1]) );
         break;
      case virvo::cartesian_axis< 3 >::Z:
      default:
         scale = virvo::scale( scale, vec3(vd->vox[0] / size[0], vd->vox[1] / size[1], vd->vox[2] / size[2]) );
         break;
   }

   if (stacking==true)
      sf = 1.0f - sdShear(3, 2);
   else
      sf = sdShear(3, 2) + 1;

   sf = 1.0f / sf;                                // invert scale factor
   scale = virvo::scale( scale, vec3(sf, sf, 1.0f) );

   // Adjust intermediate image size to desired image quality:
   scale = virvo::scale( scale, vec3(_quality, _quality, 1.0f) );

   VV_LOG(3) << "scale: " << scale;
}


//----------------------------------------------------------------------------
/** Compute the pure shear matrix.
  The pure shear matrix consists only of the shear, it transforms standard
  object space to deformed object space. No coodinate
  adjustment is done yet. This calculation depends on the eye coordinates
  in standard object space.
*/
void vvSoftPer::findSDShearMatrix()
{
   vvDebugMsg::msg(3, "vvSoftPer::findShearMatrix()");

   sdShear = mat4::identity();
   assert(sEye[2] != 0.0f);                       // this should be asserted by the shift matrix
   sdShear(0, 2) = -sEye[0] / sEye[2];          // shear in x direction
   sdShear(1, 2) = -sEye[1] / sEye[2];          // shear in y direction
   sdShear(3, 2) = -sEye[3] / sEye[2];          // perspective scale component

   VV_LOG(3) << "sdShear: " << sdShear;
}


//----------------------------------------------------------------------------
/** Compute the complete shear matrix.
  The complete shear matrix transforms object space to intermediate image
  space.<BR>
  oiShear = diConv x scale x sdShear x shift x osPerm
*/
void vvSoftPer::findOIShearMatrix()
{
    oiShear = diConv * scale * sdShear * shift * osPerm;

    VV_LOG(3) << "oiShear: " << oiShear;
}


//----------------------------------------------------------------------------
/** Compute the warp matrix.
  ivWarp = wvConv x owView x soPerm x dsShear x invScale x idConv
*/
void vvSoftPer::findWarpMatrix()
{
   mat4 ioShear;                             // inverse of oiShear

   vvDebugMsg::msg(3, "vvSoftPer::findWarpMatrix()");

   // Compute inverse of oiShear:
   ioShear = inverse(oiShear);

   // Assemble warp matrices:
   iwWarp = owView * ioShear;
   VV_LOG(3) << "iwWarp: " << iwWarp;
   ivWarp = wvConv * iwWarp;
   VV_LOG(3) << "ivWarp: " << ivWarp;
}


//----------------------------------------------------------------------------
/// Factor the view matrix to obtain separate shear and warp matrices.
void vvSoftPer::factorViewMatrix()
{
   vvDebugMsg::msg(3, "vvSoftPer::factorViewMatrix()");

   findOEyePosition();
   findPrincipalAxis();
   findPermutationMatrix();
   findShiftMatrix();
   findSEyePosition();
   findSDShearMatrix();
   findScaleMatrix();
   findOIShearMatrix();
   findWarpMatrix();
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
