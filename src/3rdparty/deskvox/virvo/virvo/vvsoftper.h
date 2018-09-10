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

#ifndef VV_SOFTPER_H
#define VV_SOFTPER_H

#include "vvexport.h"
#include "vvrenderer.h"
#include "vvsoftvr.h"

/** This class is the implementation of a new volume rendering algorithm
  based on the perspective shear-warp factorization.
  Coordinate systems:
  <UL>
    <LI>o = object space</LI>
    <LI>s = standard object space</LI>
    <LI>d = deformed space</LI>
    <LI>i = intermediate image coordinates</LI>
    <LI>w = world coordinates</LI>
    <LI>v = OpenGL viewport space</LI>
  </UL>

Terminology:<PRE>
glMV    = OpenGL modelview matrix
glPM    = OpenGL projection matrix (perspective transformation matrix)
owView  = combined view matrix from object space to world space (Lacroute: M_view)
wvConv  = conversion from world to OpenGL viewport
osPerm  = permutation from object space to standard object space (Lacroute: P)
sdShear = shear matrix from standard object space to deformed space
oiShear = shear matrix from object space to intermediate image space
diConv  = conversion from deformed space to intermediate image coordinates
iwWarp  = 2D warp from intermediate image to world (Lacroute: M_warp)
ivWarp  = 2D warp from intermediate image to OpenGL viewport
scale   = scaling matrix (Lacroute: S)
shift   = shift matrix to translate volume if eye is in slice k=0 (Lacroute: T_shift)
oEye    = eye position in object space (Lacroute: e_o)
sEye    = eye position in standard object space (Lacroute: e_s)
x       = matrix multiplication, order for multiple multiplications: from right to left
-1      = inverse of matrix (e.g. owView-1 = woView)
</PRE>

Important equations:<PRE>
owView  = glPM x glMV
ovView  = wvConv x owView
osPerm  = depending on principal viewing axis
wvConv  = depending on OpenGL viewport size
ovView  = ivWarp x diConv x scale x sdShear x shift x osPerm
oiConv  = diConv x scale x sdShear x shift x osPerm
diConv  = depending on intermediate image size
oEye    = woView x {0,0,-1,0}
shift   = depending on oEye and osPerm
sEye    = shift x osPerm x oEye
scale   = depending on sdShear and volume size
sdShear = depending on sEye
oiShear = diConv x scale x sdShear x shift x osPerm
iwWarp  = owView x soPerm x shift-1 x dsShear x scale-1 x idConv
ivWarp  = wvConv x owView x soPerm x shift-1 x dsShear x scale-1 x idConv
</PRE>

@author Juergen P. Schulze (schulze@hlrs.de)
@see vvRenderer
@see vvSoftVR
@see vvSoftImg
*/
class VIRVOEXPORT vvSoftPer : public vvSoftVR
{
   protected:
      virvo::vec4 oEye;                          ///< eye position [object space]
      virvo::vec4 sEye;                          ///< eye position [standard object space]
      virvo::mat4 shift;                         ///< shift volume if eye in slice 0 = T_shift
      virvo::mat4 scale;                         ///< scale front voxel slice to 1:1 pixel:voxel ratio
      virvo::mat4 diConv;                        ///< convert deformed space to intermediate image space
      virvo::mat4 sdShear;                       ///< shear matrix from standard object space to deformed (sheared) space

      void compositeSliceNearest(int, int = -1, int = -1);
      void compositeSliceBilinear(int, int = -1, int = -1);
      void interpolateVoxels(uchar*, float, float, float*, float*, float*, float*);
      void accumulateVoxels(uchar*, float, float, float, float, float*, float*, float*, float*);
      void setQuality(float);
      void findDIConvMatrix();
      void findOEyePosition();
      void findPrincipalAxis();
      void findShiftMatrix();
      void findSEyePosition();
      void findScaleMatrix();
      void findSDShearMatrix();
      void findOIShearMatrix();
      void findWarpMatrix();
      void factorViewMatrix();

   public:
      vvSoftPer(vvVolDesc*, vvRenderState);
      virtual ~vvSoftPer();
      void compositeVolume(int = -1, int = -1);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
