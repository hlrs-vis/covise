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
#include "vvtoolshed.h"

#include <math.h>
#include <assert.h>
#include "private/vvlog.h"
#include "vvdebugmsg.h"
#include "vvclock.h"
#include "vvsoftimg.h"
#include "vvvoldesc.h"
#include "vvsoftpar.h"

using virvo::mat4;
using virvo::vec3;
using virvo::vec4;

//----------------------------------------------------------------------------
/** Constructor.
  @param vd volume description of volume to display
  @see vvRenderer
*/
vvSoftPar::vvSoftPar(vvVolDesc* vd, vvRenderState rs) : vvSoftVR(vd, rs)
{
   int imgSize;                                   // edge size of intermediate image [pixels]

   vvDebugMsg::msg(1, "vvSoftPar::vvSoftPar()");

   rendererType = SOFTPAR;
   bufSlice[0] = bufSlice[1] = NULL;
   bufSliceLen[0] = bufSliceLen[1] = 0;
   readSlice = 0;
   wViewDir = vec3(0.0f, 0.0f, 1.0f);

   // Provide enough space for all possible shear matrices:
   imgSize = (int)(2 * ts_max(vd->vox[0], vd->vox[1], vd->vox[2]));
   intImg = new vvSoftImg(imgSize, imgSize);
}


//----------------------------------------------------------------------------
/// Destructor.
vvSoftPar::~vvSoftPar()
{
   vvDebugMsg::msg(1, "vvSoftPar::~vvSoftPar()");

   delete[] bufSlice[0];
   delete[] bufSlice[1];
}


//----------------------------------------------------------------------------
/** Composite the volume slices to the intermediate image.
  The function prepareRendering() must be called before this method.
  The shear transformation matrices have to be computed before calling this method.
  The volume slices are processed from front to back.
  @param from,to optional arguments to define first and last intermediate image line to render.
                 if not passed, the entire intermediate image will be rendered
*/
void vvSoftPar::compositeVolume(int from, int to)
{
   int slice;                                     // currently processed slice
   int firstSlice;                                // first slice to process
   int lastSlice;                                 // last slice to process
   int sliceStep;                                 // step size to get to next slice

   vvDebugMsg::msg(3, "vvSoftPar::compositeVolume(): ", from, to);

   intImg->clear();

   // If stacking==true then draw front to back, else draw back to front:
   firstSlice = (stacking) ? 0 : (len[2]-1);
   lastSlice  = (stacking) ? (len[2]-1) : 0;
   sliceStep  = (stacking) ? 1 : -1;

   earlyRayTermination = 0;

   if (_preIntegration)
   {
      if (!sliceBuffer)
      {
         // Manage two buffer slices:
         for (size_t i=0; i<2; ++i)
         {
            if (bufSlice[i]==NULL || bufSliceLen[0]!=len[0]-1 || bufSliceLen[1]!=len[1]-1)
            {
               delete[] bufSlice[i];
               bufSlice[i] = new float[(len[0]-1) * (len[1]-1) * vd->getBPV()];
            }
         }
         bufSliceLen[0] = len[0] - 1;
         bufSliceLen[1] = len[1] - 1;
      }
      else
      {
         // Manage one buffer slice:
         if (bufSlice[0]==NULL || bufSliceLen[0]!=intImg->width || bufSliceLen[1]!=intImg->height)
         {
            delete[] bufSlice[0];
            bufSlice[0] = new float[intImg->width * intImg->height * vd->getBPV()];
         }
         bufSliceLen[0] = intImg->width;
         bufSliceLen[1] = intImg->height;
      }
   }

   if (opCorr)
   {
      // Compute opacity correction table:
      int i;
      float dx;
      vec3 pos1, pos2, dummy;
      findSlicePosition(0, &pos1, &dummy);
      findSlicePosition(1, &pos2, &dummy);
      pos1 -= pos2;
      dx = sqrtf(1.0f + pos1[0] * pos1[0] + pos1[1] * pos1[1]);
      for (i=0; i<VV_OP_CORR_TABLE_SIZE; ++i)
      {
         opacityCorr[i] = (1.0f - powf(1.0f - i / float(VV_OP_CORR_TABLE_SIZE), dx)) * 256.0f;
         if (i==0) colorCorr[i] = 1.0f;
         else colorCorr[i] = opacityCorr[i] / (i / float(VV_OP_CORR_TABLE_SIZE) * 256.0f);
      }
   }

   for (slice=firstSlice; slice!=lastSlice; slice += sliceStep)
   {
      if (_preIntegration)  compositeSlicePreIntegrated(slice, sliceStep);
      else if (compression && rleStart[0]!=NULL)
      {
         if (sliceInterpol) compositeSliceCompressedBilinear(slice);
         else               compositeSliceCompressedNearest(slice);
      }
      else
      {
         if (sliceInterpol) compositeSliceBilinear(slice);
         else               compositeSliceNearest(slice, from, to);
      }
   }

   //  cerr << "Early ray termination: " << earlyRayTermination << " of " << vd->getFrameVoxels() <<
   //    " (" << (100.0f * earlyRayTermination / vd->getFrameVoxels()) << "%)" << endl;
}


//----------------------------------------------------------------------------
/** Composite the voxels from one slice into the intermediate image using
  a nearest neighbor resampling algorithm.
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
void vvSoftPar::compositeSliceNearest(int slice, int from, int to)
{
   vec3 vStart;                                   // bottom left voxel of this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates (Y=0 is bottom)
   int    ix,iy;                                  // counters [intermediate image space]
   uchar* vScalar;                                // pointer to scalar voxel data corresponding to current image pixel
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
   float  tmp;

   findSlicePosition(slice, &vStart, NULL);
   iPosX     = vvToolshed::round(vStart[0]);      // use nearest intermediate image column
   iPosY     = vvToolshed::round(vStart[1]);      // use nearest intermediate image line
   iSlice[0] = len[0];
   iSlice[1] = len[1];
   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0]);
   vScalar = raw[principal] + vd->getBPV() * (slice * len[0] * len[1] + (len[1] - 1) * len[0]);

   if (from != -1)                                // render only specific lines?
   {
      int iTopLine;                               // topmost intermediate image line

      iTopLine = iPosY + iSlice[1] - 1;
      if (to < iPosY || from > iTopLine) return;  // return if section to render is outside of slice area

      // Constrain section indices:
      if (from < iPosY) from = iPosY;
      if (to > iTopLine) to = iTopLine;

      // Modify first voxel to draw:
      vScalar -= (from - iPosY) * len[0] * vd->getBPV();

      // Modify first pixel to draw:
      iPosY = from;

      // Modify number of lines to draw:
      iSlice[1] = to - from + 1;
   }

   iPixel  = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);

   // Traverse intermediate image pixels which correspond to the current slice.
   // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
   for (iy=0; iy<iSlice[1]; ++iy)
   {
      for (ix=0; ix<iSlice[0]; ++ix)
      {
         if (rgbaConv[*vScalar][3]==0) ++earlyRayTermination;
         if (rgbaConv[*vScalar][3]>0 &&           // skip transparent voxels
                                                  // skip clipped voxels
            (!getParameter(VV_CLIP_MODE) || !isVoxelClipped(ix, iSlice[1]-iy-1, slice)))
         {
            // Determine image color components and scale to [0..1]:
            ir = (float)(*(iPixel++)) / 255.0f;
            ig = (float)(*(iPixel++)) / 255.0f;
            ib = (float)(*(iPixel++)) / 255.0f;
            ia = (float)(*(iPixel++)) / 255.0f;

            if (ia < 1.0f)                        // skip opaque intermediate image pixels
            {
               // Determine voxel color components and scale to [0..1]:
               vr = (float)rgbaConv[*vScalar][0] / 255.0f;
               vg = (float)rgbaConv[*vScalar][1] / 255.0f;
               vb = (float)rgbaConv[*vScalar][2] / 255.0f;
               va = (float)rgbaConv[*vScalar][3] / 255.0f;
               iPixel -= vvSoftImg::PIXEL_SIZE;   // start over with intermediate image components

               // Accumulate new intermediate image pixel values.
               *(iPixel++) = (uchar)((tmp = (255.0f * (ir + (1.0f - ia) * vr * va))) < 255.0f ? tmp : 255.0f);
               *(iPixel++) = (uchar)((tmp = (255.0f * (ig + (1.0f - ia) * vg * va))) < 255.0f ? tmp : 255.0f);
               *(iPixel++) = (uchar)((tmp = (255.0f * (ib + (1.0f - ia) * vb * va))) < 255.0f ? tmp : 255.0f);
               *(iPixel++) = (uchar)((tmp = (255.0f * (ia + (1.0f - ia) * va))) < 255.0f ? tmp : 255.0f);
            }
         }
         else iPixel += vvSoftImg::PIXEL_SIZE;

         // Switch to next voxel:
         vScalar += vd->getBPV();
      }
      vScalar -= (2 * len[0]) * vd->getBPV();
      iPixel += iLineOffset;
   }
}


//----------------------------------------------------------------------------
/** Composite a slice to the intermediate image using bilinear interpolation.
  @param slice slice number to composite
*/
void vvSoftPar::compositeSliceBilinear(int slice)
{
   vec3 vStart;                                   // bottom left voxel of this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates (Y=0 is bottom)
   int    ix,iy;                                  // counters [intermediate image space]
   uchar* vScalar[4];                             // ptr to scalar data: 0=bot.left, 1=top left, 2=top right, 3=bot.right
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   int    vLineOffset;                            // offset to next line in volume
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
   float  frac[2];                                // fractions for resampling (x,y)
   float  weight[4];                              // resampling weights, one for each of the four neighboring voxels (for indices see vScalar[])
   float  tmp;
   int    i;
   const bool postClassification = false;

   findSlicePosition(slice, &vStart, NULL);
   iPosX     = int(vStart[0]) + 1;                // use intermediate image column right of bottom left voxel location
   iPosY     = int(vStart[1]) + 1;                // use intermediate image line top of bottom left voxel location
   vScalar[0]= raw[principal] + vd->getBPV() * (slice * len[0] * len[1] + (len[1] - 1) * len[0]);
   vScalar[1]= vScalar[0] - vd->getBPV() * len[0];
   vScalar[2]= vScalar[1] + vd->getBPV();
   vScalar[3]= vScalar[0] + vd->getBPV();
   iPixel    = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);
   iSlice[0] = len[0];
   iSlice[1] = len[1];
   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0] + 1);
   vLineOffset = (2 * len[0] - 1) * (int)vd->getBPV();
   frac[0]   = (float)iPosX - vStart[0];
   frac[1]   = (float)iPosY - vStart[1];

   // Compute bilinear resampling weights:
   weight[0] = (1.0f - frac[0]) * (1.0f - frac[1]);
   weight[1] = (1.0f - frac[0]) * frac[1];
   weight[2] = frac[0] * frac[1];
   weight[3] = frac[0] * (1.0f - frac[1]);

   // Traverse intermediate image pixels which correspond to the current slice.
   // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
   for (iy=0; iy<iSlice[1]-1; ++iy)
   {
      for (ix=0; ix<iSlice[0]-1; ++ix)
      {
                                                  // skip clipped voxels
         if (!getParameter(VV_CLIP_MODE) || !isVoxelClipped(ix, iSlice[1]-iy-1, slice))
         {
                                                  // skip transparent voxels
            if (rgbaConv[*vScalar[0]][3]>0 || rgbaConv[*vScalar[1]][3]>0 ||
               rgbaConv[*vScalar[2]][3]>0 || rgbaConv[*vScalar[3]][3]>0)
            {
               // Determine image color components and scale to [0..1]:
               ir = (float)(*(iPixel++)) / 255.0f;
               ig = (float)(*(iPixel++)) / 255.0f;
               ib = (float)(*(iPixel++)) / 255.0f;
               ia = (float)(*(iPixel++)) / 255.0f;

               if (ia>=1.0f) ++earlyRayTermination;
               if (ia < 1.0f)                     // skip opaque intermediate image pixels
               {
                  if(postClassification)
                  {
                     // Determine interpolated voxel value:
                     uchar v = static_cast<uchar>(*vScalar[0] * weight[0]
                        + *vScalar[1] * weight[1]
                        + *vScalar[2] * weight[2]
                        + *vScalar[3] * weight[3]);
                     va = rgbaConv[v][3] / 255.0f;
                     if (va>0.0f)                    // skip transparent voxels (yes, do it again!)
                     {
                        vr = rgbaConv[v][0] / 255.0f;
                        vg = rgbaConv[v][1] / 255.0f;
                        vb = rgbaConv[v][2] / 255.0f;
                        // start over with intermediate image components
                        iPixel -= vvSoftImg::PIXEL_SIZE;

                        // Accumulate new intermediate image pixel values.
                        *(iPixel++) = (uchar)((tmp = (255.0f * (ir + (1.0f - ia) * vr * va))) < 255.0f ? tmp : 255.0f);
                        *(iPixel++) = (uchar)((tmp = (255.0f * (ig + (1.0f - ia) * vg * va))) < 255.0f ? tmp : 255.0f);
                        *(iPixel++) = (uchar)((tmp = (255.0f * (ib + (1.0f - ia) * vb * va))) < 255.0f ? tmp : 255.0f);
                        *(iPixel++) = (uchar)((tmp = (255.0f * (ia + (1.0f - ia) * va))) < 255.0f ? tmp : 255.0f);
                     }
                  }
                  else
                  {
                  // Determine interpolated voxel color components and scale to [0..1]:
                  va = ((float)rgbaConv[*vScalar[0]][3] * weight[0] +
                     (float)rgbaConv[*vScalar[1]][3] * weight[1] +
                     (float)rgbaConv[*vScalar[2]][3] * weight[2] +
                     (float)rgbaConv[*vScalar[3]][3] * weight[3]) / 255.0f;
                  if (va>0.0f)                    // skip transparent voxels (yes, do it again!)
                  {
                     vr = ((float)rgbaConv[*vScalar[0]][0] * weight[0] +
                        (float)rgbaConv[*vScalar[1]][0] * weight[1] +
                        (float)rgbaConv[*vScalar[2]][0] * weight[2] +
                        (float)rgbaConv[*vScalar[3]][0] * weight[3]) / 255.0f;
                     vg = ((float)rgbaConv[*vScalar[0]][1] * weight[0] +
                        (float)rgbaConv[*vScalar[1]][1] * weight[1] +
                        (float)rgbaConv[*vScalar[2]][1] * weight[2] +
                        (float)rgbaConv[*vScalar[3]][1] * weight[3]) / 255.0f;
                     vb = ((float)rgbaConv[*vScalar[0]][2] * weight[0] +
                        (float)rgbaConv[*vScalar[1]][2] * weight[1] +
                        (float)rgbaConv[*vScalar[2]][2] * weight[2] +
                        (float)rgbaConv[*vScalar[3]][2] * weight[3]) / 255.0f;

                                                  // start over with intermediate image components
                     iPixel -= vvSoftImg::PIXEL_SIZE;

                     // Accumulate new intermediate image pixel values.
                     /*
                      *(iPixel++) = (uchar)(255.0f * (ia * ir + (1.0f - ia) * vr));
                      *(iPixel++) = (uchar)(255.0f * (ia * ig + (1.0f - ia) * vg));
                      *(iPixel++) = (uchar)(255.0f * (ia * ib + (1.0f - ia) * vb));
                      *(iPixel++) = (uchar)(255.0f * (ia + va * (1.0f - ia)));
                      */
                     *(iPixel++) = (uchar)((tmp = (255.0f * (ir + (1.0f - ia) * vr * va))) < 255.0f ? tmp : 255.0f);
                     *(iPixel++) = (uchar)((tmp = (255.0f * (ig + (1.0f - ia) * vg * va))) < 255.0f ? tmp : 255.0f);
                     *(iPixel++) = (uchar)((tmp = (255.0f * (ib + (1.0f - ia) * vb * va))) < 255.0f ? tmp : 255.0f);
                     *(iPixel++) = (uchar)((tmp = (255.0f * (ia + (1.0f - ia) * va))) < 255.0f ? tmp : 255.0f);
                  }
                  }
               }
            }
            else iPixel += vvSoftImg::PIXEL_SIZE;
         }
         else iPixel += vvSoftImg::PIXEL_SIZE;

         // Switch to next voxel:
         vScalar[0] = vScalar[3];
         vScalar[1] = vScalar[2];
         vScalar[2] += vd->getBPV();
         vScalar[3] += vd->getBPV();
      }
      for (i=0; i<4; ++i)
         vScalar[i] -= vLineOffset;
      iPixel += iLineOffset;
   }
}


//----------------------------------------------------------------------------
/** Composite the voxels from one slice into the intermediate image using
  compressed volume data and nearest neighbor resampling.<BR>
  Note: This rendering algorithm does not support clipping planes!<BR>
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
*/
void vvSoftPar::compositeSliceCompressedNearest(int slice)
{
   vec3 vStart;                                   // bottom left voxel of this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates (Y=0 is bottom)
   int    ix,iy,i;                                // counters [intermediate image space]
   uchar* vScalar;                                // pointer to scalar voxel data corresponding to current image pixel
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
   int    count;

   findSlicePosition(slice, &vStart, NULL);
   iPosX     = vvToolshed::round(vStart[0]);      // use nearest intermediate image column
   iPosY     = vvToolshed::round(vStart[1]);      // use nearest intermediate image line
   iPixel    = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);
   iSlice[0] = len[0];
   iSlice[1] = len[1];
   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0]);

   // Traverse intermediate image pixels which correspond to the current slice.
   // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
   for (iy=0; iy<iSlice[1]; ++iy)
   {
      ix = 0;                                     // start with first pixel in row
      vScalar = rleStart[principal][slice * len[1] + (len[1] - 1 - iy)];

      while (ix<iSlice[0])
      {
         count = (int)*vScalar++;
         if (count > 127)                         // replicate run?
         {
            count -= 127;                         // remove bias
            if (rgbaConv[*vScalar][3] > 0)        // skip transparent voxels
            {
               // Determine voxel color components and scale to [0..1]:
               vr = (float)rgbaConv[*vScalar][0] / 255.0f;
               vg = (float)rgbaConv[*vScalar][1] / 255.0f;
               vb = (float)rgbaConv[*vScalar][2] / 255.0f;
               va = (float)rgbaConv[*vScalar][3] / 255.0f;

               for (i=0; i<count; ++i)
               {
                  // Determine image color components and scale to [0..1]:
                  ir = (float)(*(iPixel++)) / 255.0f;
                  ig = (float)(*(iPixel++)) / 255.0f;
                  ib = (float)(*(iPixel++)) / 255.0f;
                  ia = (float)(*(iPixel++)) / 255.0f;

                  if (ia < 1.0f)                  // skip opaque intermediate image pixels
                  {
                                                  // start over with intermediate image components
                     iPixel -= vvSoftImg::PIXEL_SIZE;

                     // Accumulate new intermediate image pixel values.
                     *(iPixel++) = (uchar)(255.0f * (ia * ir + (1.0f - ia) * vr));
                     *(iPixel++) = (uchar)(255.0f * (ia * ig + (1.0f - ia) * vg));
                     *(iPixel++) = (uchar)(255.0f * (ia * ib + (1.0f - ia) * vb));
                     *(iPixel++) = (uchar)(255.0f * (ia + va * (1.0f - ia)));
                  }
               }
            }
            else iPixel += count * vvSoftImg::PIXEL_SIZE;
            ++vScalar;
         }
         else                                     // literal run
         {
            ++count;                              // remove bias
            for (i=0; i<count; ++i)
            {
               if (rgbaConv[*vScalar][3]>0)       // skip transparent voxels
               {
                  // Determine image color components and scale to [0..1]:
                  ir = (float)(*(iPixel++)) / 255.0f;
                  ig = (float)(*(iPixel++)) / 255.0f;
                  ib = (float)(*(iPixel++)) / 255.0f;
                  ia = (float)(*(iPixel++)) / 255.0f;

                  if (ia < 1.0f)                  // skip opaque intermediate image pixels
                  {
                     // Determine voxel color components and scale to [0..1]:
                     vr = (float)rgbaConv[*vScalar][0] / 255.0f;
                     vg = (float)rgbaConv[*vScalar][1] / 255.0f;
                     vb = (float)rgbaConv[*vScalar][2] / 255.0f;
                     va = (float)rgbaConv[*vScalar][3] / 255.0f;

                                                  // start over with intermediate image components
                     iPixel -= vvSoftImg::PIXEL_SIZE;

                     // Accumulate new intermediate image pixel values.
                     *(iPixel++) = (uchar)(255.0f * (ia * ir + (1.0f - ia) * vr));
                     *(iPixel++) = (uchar)(255.0f * (ia * ig + (1.0f - ia) * vg));
                     *(iPixel++) = (uchar)(255.0f * (ia * ib + (1.0f - ia) * vb));
                     *(iPixel++) = (uchar)(255.0f * (ia + va * (1.0f - ia)));
                  }
               }
               else iPixel += vvSoftImg::PIXEL_SIZE;

               // Switch to next voxel:
               ++vScalar;
            }
         }
         ix += count;
      }
      iPixel += iLineOffset;
   }
}


//----------------------------------------------------------------------------
/** Composite a slice to the intermediate image using bilinear interpolation
    and RLE compression.
  @param slice slice number to composite
*/
void vvSoftPar::compositeSliceCompressedBilinear(int slice)
{
   vec3 vStart;                                   // bottom left voxel of this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates (Y=0 is bottom)
   int    ix,iy;                                  // counters [intermediate image space]
   uchar* vScalar[4];                             // ptr to scalar data: 0=bot.left, 1=top left, 2=top right, 3=bot.right
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   int    vLineOffset;                            // offset to next line in volume
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
   float  frac[2];                                // fractions for resampling (x,y)
   float  weight[4];                              // resampling weights, one for each of the four neighboring voxels (for indices see vScalar[])
   int    i;

   findSlicePosition(slice, &vStart, NULL);
   iPosX     = int(vStart[0]) + 1;                // use intermediate image column right of bottom left voxel location
   iPosY     = int(vStart[1]) + 1;                // use intermediate image line top of bottom left voxel location
   iPixel    = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);
   iSlice[0] = len[0];
   iSlice[1] = len[1];
   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0] + 1);
   vLineOffset = (2 * len[0] - 1) * (int)vd->getBPV();
   frac[0]   = (float)iPosX - vStart[0];
   frac[1]   = (float)iPosY - vStart[1];

   // Compute bilinear resampling weights:
   weight[0] = (1.0f - frac[0]) * (1.0f - frac[1]);
   weight[1] = (1.0f - frac[0]) * frac[1];
   weight[2] = frac[0] * frac[1];
   weight[3] = frac[0] * (1.0f - frac[1]);

   // Traverse intermediate image pixels which correspond to the current slice.
   // 1 is subtracted from each loop counter to remain inside of the volume boundaries:
   for (iy=0; iy<iSlice[1]-1; ++iy)
   {
      vScalar[0]= rleStart[principal][slice * len[1] + (len[1] - 1 - iy)];
      vScalar[1]= vScalar[0] - len[0];
      vScalar[2]= vScalar[1] + 1;
      vScalar[3]= vScalar[0] + 1;

      for (ix=0; ix<iSlice[0]-1; ++ix)
      {
                                                  // skip clipped voxels
         if (!getParameter(VV_CLIP_MODE) || !isVoxelClipped(ix, iSlice[1]-iy-1, slice))
         {
                                                  // skip transparent voxels
            if (rgbaConv[*vScalar[0]][3]>0 || rgbaConv[*vScalar[1]][3]>0 ||
               rgbaConv[*vScalar[2]][3]>0 || rgbaConv[*vScalar[3]][3]>0)
            {
               // Determine image color components and scale to [0..1]:
               ir = (float)(*(iPixel++)) / 255.0f;
               ig = (float)(*(iPixel++)) / 255.0f;
               ib = (float)(*(iPixel++)) / 255.0f;
               ia = (float)(*(iPixel++)) / 255.0f;

               if (ia < 1.0f)                     // skip opaque intermediate image pixels
               {
                  // Determine interpolated voxel color components and scale to [0..1]:
                  vr = ((float)rgbaConv[*vScalar[0]][0] * weight[0] +
                     (float)rgbaConv[*vScalar[1]][0] * weight[1] +
                     (float)rgbaConv[*vScalar[2]][0] * weight[2] +
                     (float)rgbaConv[*vScalar[3]][0] * weight[3]) / 255.0f;
                  vg = ((float)rgbaConv[*vScalar[0]][1] * weight[0] +
                     (float)rgbaConv[*vScalar[1]][1] * weight[1] +
                     (float)rgbaConv[*vScalar[2]][1] * weight[2] +
                     (float)rgbaConv[*vScalar[3]][1] * weight[3]) / 255.0f;
                  vb = ((float)rgbaConv[*vScalar[0]][2] * weight[0] +
                     (float)rgbaConv[*vScalar[1]][2] * weight[1] +
                     (float)rgbaConv[*vScalar[2]][2] * weight[2] +
                     (float)rgbaConv[*vScalar[3]][2] * weight[3]) / 255.0f;
                  va = ((float)rgbaConv[*vScalar[0]][3] * weight[0] +
                     (float)rgbaConv[*vScalar[1]][3] * weight[1] +
                     (float)rgbaConv[*vScalar[2]][3] * weight[2] +
                     (float)rgbaConv[*vScalar[3]][3] * weight[3]) / 255.0f;

                  iPixel -= vvSoftImg::PIXEL_SIZE;// start over with intermediate image components

                  // Accumulate new intermediate image pixel values.
                  *(iPixel++) = (uchar)(255.0f * (ia * ir + (1.0f - ia) * vr));
                  *(iPixel++) = (uchar)(255.0f * (ia * ig + (1.0f - ia) * vg));
                  *(iPixel++) = (uchar)(255.0f * (ia * ib + (1.0f - ia) * vb));
                  *(iPixel++) = (uchar)(255.0f * (ia + va * (1.0f - ia)));
               }
            }
            else iPixel += vvSoftImg::PIXEL_SIZE;
         }
         else iPixel += vvSoftImg::PIXEL_SIZE;

         // Switch to next voxel:
         vScalar[0] = vScalar[3];
         vScalar[1] = vScalar[2];
         vScalar[2] += vd->getBPV();
         vScalar[3] += vd->getBPV();
      }
      for (i=0; i<4; ++i)
         vScalar[i] -= vLineOffset;
      iPixel += iLineOffset;
   }
}


//----------------------------------------------------------------------------
/** Composite a slice to the intermediate image using the LUT with
  pre-integrated values for the compositing. The pre-integration LUT needs
  to be recomputed whenever the transfer function changes.
  @see makeLookupTextureOptimized
  @see makeLookupTextureCorrect
  @param slice slice number to composite
  @param sliceStep 1 if counting up, -1 if counting down
*/
void vvSoftPar::compositeSlicePreIntegrated(int slice, int sliceStep)
{
   vec3 vStart;                                   // bottom left voxel of this slice
   int    iPosX, iPosY;                           // current intermediate image coordinates (Y=0 is bottom)
   int    ix,iy;                                  // counters [intermediate image space]
   uchar* vScalarB[4];                            // ptr to scalar data (back): 0=bot.left, 1=top left, 2=top right, 3=bot.right
   uchar* iPixel;                                 // pointer to current intermediate image pixel
   int    iLineOffset;                            // offset to next line on intermediate image
   float  vr,vg,vb,va;                            // RGBA components of current voxel
   float  ir,ig,ib,ia;                            // RGBA components of current image pixel
   int    iSlice[2];                              // slice dimensions in intermediate image (x,y)
   float  frac[2];                                // fractions for resampling (x,y)
   float  weight[4];                              // resampling weights, one for each of the four neighboring voxels (for indices see vScalar[])
   float  preWght[4];                             // weights for lookup into pre-interpolation table
   float  sf, sb;                                 // indices into pre-integrated table (f=front, b=back)
   int    sfi, sbi;                               // integer indices
   float  sf1, sb1;                               // sx - int(sx) = fraction after decimal point
   float  sizeFactor;                             // precomputed factor depending on pre-integration table size
   int    bufX, bufY;                             // coordinates in buffer slice
   int    iSliceOffset[2];                        // difference of locations of current and previous slice
   float  tmp;
   static int   lastIPos[2];                      // last slice position on the intermediate image

   findSlicePosition(slice, &vStart, NULL);
   iPosX     = int(vStart[0]) + 1;                // use intermediate image column right of bottom left voxel location
   iPosY     = int(vStart[1]) + 1;                // use intermediate image line top of bottom left voxel location
   iPixel    = intImg->data + intImg->PIXEL_SIZE * (iPosX + iPosY * intImg->width);
   iSlice[0] = len[0];
   iSlice[1] = len[1];
   iLineOffset = intImg->PIXEL_SIZE * (intImg->width - iSlice[0] + 1);
   frac[0]   = (float)iPosX - vStart[0];
   frac[1]   = (float)iPosY - vStart[1];
   sizeFactor = float(PRE_INT_TABLE_SIZE) / 256.0f;
   vr = vg = vb = 0.0f;                           // prevent warning

   // Compute bilinear resampling weights for current slice:
   weight[0] = (1.0f - frac[0]) * (1.0f - frac[1]);
   weight[1] = (1.0f - frac[0]) * frac[1];
   weight[2] = frac[0] * frac[1];
   weight[3] = frac[0] * (1.0f - frac[1]);

   // Traverse intermediate image pixels which correspond to the current slice.
   // 1 is subtracted from each loop counter to remain inside the volume boundaries,
   // because the back voxel is always taken from one slice beyond the current one:
                                                  // only compute backup slice for slice 0
   if ((slice==0 && sliceStep==1) || (slice==len[2]-1 && sliceStep==-1))
   {
      for (iy=0; iy<iSlice[1]-1; ++iy)
      {
         vScalarB[0]= raw[principal] + vd->getBPV() * slice * len[0] * len[1] +
            (len[1] - 1 - iy) * len[0];
         vScalarB[1]= vScalarB[0] - len[0];
         vScalarB[2]= vScalarB[1] + 1;
         vScalarB[3]= vScalarB[0] + 1;

         for (ix=0; ix<iSlice[0]-1; ++ix)
         {
            if (sliceInterpol)
            {
               sb = ((float)*vScalarB[0] * weight[0] +
                  (float)*vScalarB[1] * weight[1] +
                  (float)*vScalarB[2] * weight[2] +
                  (float)*vScalarB[3] * weight[3]) * sizeFactor;
            }
            else
            {
               sb = float(*vScalarB[0]) * sizeFactor;
            }
            if (sliceBuffer)
            {
               bufSlice[0][(ix + iPosX) + (iy + iPosY) * bufSliceLen[0]] = sb;
            }
            else
            {
               bufSlice[1-readSlice][ix + iy * bufSliceLen[0]] = sb;
            }
            vScalarB[0] = vScalarB[3];
            vScalarB[1] = vScalarB[2];
            vScalarB[2] += vd->getBPV();
            vScalarB[3] += vd->getBPV();
         }
      }
   }
   else
   {
      iSliceOffset[0] = iPosX - lastIPos[0];
      iSliceOffset[1] = iPosY - lastIPos[1];

      for (iy=0; iy<iSlice[1]-1; ++iy)
      {
         // Compute the voxels on the current slice contributing to the current intImg-Pixel:
         vScalarB[0]= raw[principal] + vd->getBPV() * slice * len[0] * len[1] +
            (len[1] - 1 - iy) * len[0];
         vScalarB[1]= vScalarB[0] - len[0];
         vScalarB[2]= vScalarB[1] + 1;
         vScalarB[3]= vScalarB[0] + 1;

         for (ix=0; ix<iSlice[0]-1; ++ix)
         {
            // Determine bilinearly interpolated scalar voxel value on current slice:
            if (sliceInterpol)
            {
               sb = ((float)*vScalarB[0] * weight[0] +
                  (float)*vScalarB[1] * weight[1] +
                  (float)*vScalarB[2] * weight[2] +
                  (float)*vScalarB[3] * weight[3]) * sizeFactor;
            }
            else
            {
               sb = float(*vScalarB[0]) * sizeFactor;
            }

            // Determine intermediate image color components and scale to [0..1]:
            ir = (float)(*(iPixel++));            // / 255.0f;
            ig = (float)(*(iPixel++));            // / 255.0f;
            ib = (float)(*(iPixel++));            // / 255.0f;
            ia = (float)(*(iPixel++)) / 255.0f;

            bufX = iSliceOffset[0] + ix;
            bufY = iSliceOffset[1] + iy;
            if (ia >= 1.0f) ++earlyRayTermination;
            if (ia < 1.0f &&                      // skip opaque intermediate image pixels
                                                  // and make sure that there is a value in the buffer slice corresponding with the current voxel
               bufX >= 0 && bufX < bufSliceLen[0] && bufY >= 0 && bufY < bufSliceLen[1])
            {
               // Determine interpolated voxel color components and scale to [0..1]:
               if (sliceBuffer)
               {
                  sf = bufSlice[0][(ix + lastIPos[0]) + (iy + lastIPos[1]) * bufSliceLen[0]];
               }
               else
               {
                  sf = bufSlice[readSlice][bufX + bufY * bufSliceLen[0]];
               }

               sb1 = sb - float(int(sb));
               sf1 = sf - float(int(sf));
               preWght[0] = (1.0f - sf1) * (1.0f - sb1);
               preWght[1] = (1.0f - sf1) * sb1;
               preWght[2] = sf1 * sb1;
               preWght[3] = sf1 * (1.0f - sb1);

               if (bilinLookup)
               {
                  sfi = int(sf);
                  sbi = int(sb);

                  // Perform pre-integration table look-up with bilinear interpolation:
                  va = (preIntTable[sfi][sbi][3]     * preWght[0] +
                     preIntTable[sfi][sbi+1][3]   * preWght[1] +
                     preIntTable[sfi+1][sbi+1][3] * preWght[2] +
                     preIntTable[sfi+1][sbi][3]   * preWght[3]);

                  if (va>0.0f)                    // skip transparent voxels
                  {
                     vr = (preIntTable[sfi][sbi][0]     * preWght[0] +
                        preIntTable[sfi][sbi+1][0]   * preWght[1] +
                        preIntTable[sfi+1][sbi+1][0] * preWght[2] +
                        preIntTable[sfi+1][sbi][0]   * preWght[3]);

                     vg = (preIntTable[sfi][sbi][1]     * preWght[0] +
                        preIntTable[sfi][sbi+1][1]   * preWght[1] +
                        preIntTable[sfi+1][sbi+1][1] * preWght[2] +
                        preIntTable[sfi+1][sbi][1]   * preWght[3]);

                     vb = (preIntTable[sfi][sbi][2]     * preWght[0] +
                        preIntTable[sfi][sbi+1][2]   * preWght[1] +
                        preIntTable[sfi+1][sbi+1][2] * preWght[2] +
                        preIntTable[sfi+1][sbi][2]   * preWght[3]);
                  }
               }
               else
               {
                  sfi = int(sf + 0.5f);
                  sbi = int(sb + 0.5f);

                  // Perform pre-integration table look-up with nearest neighbor interpolation:
                  va = preIntTable[sfi][sbi][3];
                  if (va>0.0f)                    // skip transparent voxels
                  {
                     vr = preIntTable[sfi][sbi][0];
                     vg = preIntTable[sfi][sbi][1];
                     vb = preIntTable[sfi][sbi][2];
                  }
               }

               if (va>0.0f)                       // skip transparent voxels
               {
                  if (opCorr)
                  {
                     int index = int(va * VV_OP_CORR_TABLE_SIZE / 256.0f);
                     va = opacityCorr[index];
                     vr *= colorCorr[index];
                     vg *= colorCorr[index];
                     vb *= colorCorr[index];
                  }

                  iPixel -= vvSoftImg::PIXEL_SIZE;// start over with intermediate image components

                  // Accumulate new intermediate image pixel values.
                  *(iPixel++) = (uchar)((tmp = ((ir + (1.0f - ia) * vr))) < 255.0f ? tmp : 255.0f);
                  *(iPixel++) = (uchar)((tmp = ((ig + (1.0f - ia) * vg))) < 255.0f ? tmp : 255.0f);
                  *(iPixel++) = (uchar)((tmp = ((ib + (1.0f - ia) * vb))) < 255.0f ? tmp : 255.0f);
                  *(iPixel++) = (uchar)((tmp = ((255.0f * ia + (1.0f - ia) * va))) < 255.0f ? tmp : 255.0f);
               }
            }
            if (sliceBuffer)
            {
               bufSlice[0][(ix + iPosX) + (iy + iPosY) * bufSliceLen[0]] = sb;
            }
            else
            {
               bufSlice[1-readSlice][ix + iy * bufSliceLen[0]] = sb;
            }

            // Switch to next voxel:
            vScalarB[0] = vScalarB[3];
            vScalarB[1] = vScalarB[2];
            vScalarB[2] += vd->getBPV();
            vScalarB[3] += vd->getBPV();
         }
         iPixel += iLineOffset;
      }
   }
   lastIPos[0] = iPosX;
   lastIPos[1] = iPosY;
   readSlice = 1 - readSlice;
}


//----------------------------------------------------------------------------
/** Compute viewing direction in object space.
  oViewDir = woView x wViewDir
*/
void vvSoftPar::findOViewingDirection()
{

   vvDebugMsg::msg(3, "vvSoftPar::findOViewingDirection()");

   // Compute inverse of view matrix:
   mat4 woView = inverse(owView);

   // Compute viewing direction:
   oViewDir = ( woView * vec4(wViewDir, 1.0f) ).xyz();
   VV_LOG(3) << "oViewDir: " << oViewDir;
}


//----------------------------------------------------------------------------
/// Compute the principal viewing axis.
void vvSoftPar::findPrincipalAxis()
{
   float maximum;                                 // maximum coordinate value

   vvDebugMsg::msg(3, "vvSoftPar::findPrincipalAxis()");

   maximum = (float)ts_max(fabs(oViewDir[0]), fabs(oViewDir[1]), fabs(oViewDir[2]));

   if      (fabs(oViewDir[0]) == maximum) principal = virvo::cartesian_axis< 3 >::X;
   else if (fabs(oViewDir[1]) == maximum) principal = virvo::cartesian_axis< 3 >::Y;
   else principal = virvo::cartesian_axis< 3 >::Z;

   if (oViewDir[principal] > 0) stacking = false;
   else stacking = true;

   if (vvDebugMsg::isActive(3))
      cerr << "Principal axis: " << principal << endl;
   if (vvDebugMsg::isActive(3))
   {
      cerr << "Stacking order: ";
      if (stacking) cerr << "true" << endl;
      else cerr << "false" << endl;
   }
}


//----------------------------------------------------------------------------
/** Compute the viewing direction in standard object space.
  sViewDir = osPerm x oViewDir
*/
void vvSoftPar::findSViewingDirection()
{
   vvDebugMsg::msg(3, "vvSoftPar::findSViewingDirection()");

   sViewDir = ( osPerm * vec4(oViewDir, 1.0f) ).xyz();
   VV_LOG(3) << "sViewDir: " << sViewDir;
}


//----------------------------------------------------------------------------
/** Find the shear matrix.
  The shear matrix converts standard object space into intermediate
  image space. It consists of two shear factors, a mirroring on the
  y axis, and two translation factors.
*/
void vvSoftPar::findShearMatrix()
{
   float si, sj;                                  // shear factors
   mat4 siShear;                                 // shear standard object space to intermediate image space

   vvDebugMsg::msg(3, "vvSoftPar::findShearMatrix()");

   // Compute shear factors:
   si = - sViewDir[0] / sViewDir[2];
   sj = - sViewDir[1] / sViewDir[2];

   // Assemble standard object space shear matrix from shear factors:
   siShear = mat4::identity();
   siShear(0, 2) = si;
   siShear(1, 2) = sj;

   // Add scale factor depending on object size:
   vec3 size = vd->getSize();
   mat4 scaleMat = mat4::identity();
   switch(principal)
   {
      case virvo::cartesian_axis< 3 >::X:
         scaleMat = scale( scaleMat, vec3(vd->vox[1] / size[1], vd->vox[2] / size[2], vd->vox[0] / size[0]) );
         break;
      case virvo::cartesian_axis< 3 >::Y:
         scaleMat = scale( scaleMat, vec3(vd->vox[2] / size[2], vd->vox[0] / size[0], vd->vox[1] / size[1]) );
         break;
      case virvo::cartesian_axis< 3 >::Z:
         scaleMat = scale( scaleMat, vec3(vd->vox[0] / size[0], vd->vox[1] / size[1], vd->vox[2] / size[2]) );
         break;
   }
   siShear = scaleMat * siShear;

   // Create conversion matrix for intermediate image coordinates:
   // Shift right and down.
   // Required 3D matrix:
   //  1   0    0   w/2
   //  0   1    0   h/2
   //  0   0    1    0
   //  0   0    0    1
   mat4 imgConv = mat4::identity();
   imgConv = scale( imgConv, vec3(quality, quality, 1.0f) );
   imgConv(0, 3) = (float)(intImg->width / 2);
   imgConv(1, 3) = (float)(intImg->height / 2);
   siShear = imgConv * siShear;
   if (vvDebugMsg::isActive(3))
   {
      intImg->print("intImg:");
   }
   VV_LOG(3) << "imgConv: " << imgConv;
   VV_LOG(3) << "siShear: " << siShear;

   // Assemble final shear matrix:
   oiShear = siShear * osPerm;

   VV_LOG(3) << "oiShear: " << oiShear;
}


//----------------------------------------------------------------------------
/** Compute the 2D warp matrix.
  ivWarp = wvConv x owView x soPerm x isShear<BR>
  The resulting warp matrix must be a 2D matrix.
*/
void vvSoftPar::findWarpMatrix()
{

   vvDebugMsg::msg(3, "vvSoftPar::findWarpMatrix()");

   // Compute inverse of shear matrix:
   mat4 ioShear = inverse(oiShear);

   // Compute warp matrices:
   iwWarp = owView * ioShear;
   VV_LOG(3) << "iwWarp: " << iwWarp;
   ivWarp = wvConv * iwWarp;
   VV_LOG(3) << "ivWarp: " << ivWarp;
}


//----------------------------------------------------------------------------
/// Factor the view matrix to obtain separate shear and warp matrices.
void vvSoftPar::factorViewMatrix()
{
   vvDebugMsg::msg(3, "vvSoftPar::factorViewMatrix()");

   findOViewingDirection();
   findPrincipalAxis();
   findPermutationMatrix();
   findSViewingDirection();
   findShearMatrix();
   findWarpMatrix();
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
