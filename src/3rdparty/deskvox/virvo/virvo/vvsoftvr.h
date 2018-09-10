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

#ifndef VV_SOFTVR_H
#define VV_SOFTVR_H

#include "math/math.h"
#include "vvexport.h"
#include "vvrenderer.h"

class vvImage;
class vvSoftImg;

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

/** A new rendering algorithm based on shear-warp factorization.
  This is the dispatcher class which chooses among the
  parallel (vvSoftPar) and perspective (vvSoftPer) projection variants,
  and it contains routines which are common to the above subclasses.

  The volume data are kept in memory three times, once for each principal
  axis. The permutations of coordinate axes are as follows:

  <PRE>
  Principal Axis    Coordinate System    Permutation Matrix
  ----------------------------------------------------------
z                / 0 1 0 0 \
|               |  0 0 1 0  |
X Axis              |___y           |  1 0 0 0  |
/                 \ 0 0 0 1 /
x
x                / 0 0 1 0 \
|               |  1 0 0 0  |
Y Axis              |___z           |  0 1 0 0  |
/                 \ 0 0 0 1 /
y
y                / 1 0 0 0 \
|               |  0 1 0 0  |
Z Axis              |___x           |  0 0 1 0  |
/                 \ 0 0 0 1 /
z

</PRE>

Compositing strategy: Bilinear resampling is performed within
the volume boundaries, so that there are always 4 voxel values
used for the resampling.

@author Juergen Schulze-Doebold (schulze@hlrs.de)
@see vvRenderer
@see vvSoftPar
@see vvSoftPer
*/
class VIRVOEXPORT vvSoftVR : public vvRenderer
{
   protected:
      enum WarpType                               /// possible warp techniques
      {
         SOFTWARE,                                ///< perform warp in software
         TEXTURE,                                 ///< use 2D texturing hardware for warp
         CUDATEXTURE                              ///< use direct copy from CUDA and 2D texturing hardware
      };
      enum
      {
         PRE_INT_TABLE_SIZE = 256
      };
      vvSoftImg* outImg;                          ///< output image
      uchar* raw[3];                              ///< scalar voxel field for principle viewing axes (x, y, z)
      virvo::mat4 owView;                         ///< viewing transformation matrix from object space to world space
      virvo::mat4 osPerm;                         ///< permutation matrix
      virvo::mat4 wvConv;                         ///< conversion from world space to OpenGL viewport space
      virvo::mat4 oiShear;                        ///< shear matrix from object space to intermediate image space
      virvo::mat4 ivWarp;                         ///< warp object from intermediate image space to OpenGL viewport space
      virvo::mat4 iwWarp;                         ///< warp object from intermediate image space to world space
      int vWidth;                                 ///< OpenGL viewport width [pixels]
      int vHeight;                                ///< OpenGL viewport height [pixels]
      int len[3];                                 ///< volume dimensions in standard object space (x,y,z)
      virvo::cartesian_axis< 3 > principal;       ///< principal viewing axis
      bool stacking;                              ///< slice stacking order; true=front to back
      WarpType warpMode;                          ///< current warp mode
      float rgbaTF[4096*4];                       ///< transfer function lookup table
      uchar rgbaConv[4096][4];                    ///< density to RGBA conversion table (max. 8 bit density supported) [scalar values][RGBA]
      virvo::vec3 xClipNormal;                    ///< clipping plane normal in permuted voxel coordinate system
      float xClipDist;                            ///< clipping plane distance in permuted voxel coordinate system
      uchar** rleStart[3];                        ///< pointer lists to line beginnings, for each principal viewing axis (x,y,z). If first entry is NULL, there is no RLE compressed volume data
      uchar* rle[3];                              ///< RLE encoded volume data for each principal viewing axis (x,y,z)
      int numProc;                                ///< number of processors in system
      bool compression;                           ///< true = use compressed volume data for rendering
      bool multiprocessing;                       ///< true = use multiprocessing where possible
      bool sliceInterpol;                         ///< inter-slice interpolation mode: true=bilinear interpolation (default), false=nearest neighbor
      bool warpInterpol;                          ///< warp interpolation: true=bilinear, false=nearest neighbor
      bool sliceBuffer;                           ///< slice buffer: true=intermediate image aligned, false=slice aligned
      bool bilinLookup;                           ///< true=bilinear lookup in pre-integration table, false=nearest neighbor lookup
      bool opCorr;                                ///< true=opacity correction on
      float quality;                              ///< quality actually used
      float oldQuality;                           ///< previous image quality
                                                  ///< size of pre-integrated LUT ([sf][sb][RGBA])
      uchar preIntTable[PRE_INT_TABLE_SIZE][PRE_INT_TABLE_SIZE][4];
      int earlyRayTermination;                    ///< counter for number of voxels which are skipped due to early ray termination
      bool _timing;
      virvo::vec3 _size;

      void setOutputImageSize();
      void findVolumeDimensions();
      virtual void findAxisRepresentations();
      void encodeRLE();
      int  getLUTSize();
      void findViewMatrix();
      void findPermutationMatrix();
      void findViewportMatrix(int, int);
      void findSlicePosition(int, virvo::vec4*, virvo::vec4*);
      void findSlicePosition(int, virvo::vec3*, virvo::vec3*);
      void findClipPlaneEquation();
      bool isVoxelClipped(int, int, int);
      void compositeOutline();
      virtual int  getCullingStatus(float);
      virtual void factorViewMatrix() = 0;
      virtual void setQuality(float q);
      virtual void updateLUT(float dist);

   public:
      vvSoftImg* intImg;                          ///< intermediate image

      vvSoftVR(vvVolDesc*, vvRenderState);
      virtual ~vvSoftVR();
      void     updateTransferFunction();
      void     makeLookupTextureCorrect(float = 1.0f);
      void     makeLookupTextureOptimized(float = 1.0f);
      void     updateVolumeData();
      bool     instantClassification() const;
      void     setWarpMode(WarpType);
      WarpType getWarpMode();
      void     setCurrentFrame(size_t);
      void     renderVolumeGL();
      void     getIntermediateImage(vvImage*);
      virvo::mat4 getWarpMatrix() const;
      bool     prepareRendering();
      virtual void setParameter(ParameterType param, const vvParam& value);
      virtual vvParam getParameter(ParameterType param) const;
      virtual void compositeVolume(int = -1, int = -1) = 0;
      virtual void getIntermediateImageExtent(int*, int*, int*, int*);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
