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

#ifndef VV_SOFTPAR_H
#define VV_SOFTPAR_H

#include "vvrenderer.h"
#include "vvsoftvr.h"
#include "vvexport.h"

/** Parallel projection shear-warp algorithm.
  The algorithm was implemented according to the description in
  P. Lacroute's Ph.D. thesis (Stanford University).<BR>
  Coordinate systems (used in first two letters of matrix names (from-to) and
  first letter of vector names):
  <UL>
    <LI>o = object space</LI>
    <LI>s = standard object space</LI>
    <LI>i = intermediate image coordinates</LI>
    <LI>w = world coordinates</LI>
    <LI>v = OpenGL viewport space</LI>
</UL>

Terminology:<PRE>
glPM     = OpenGL projection matrix (perspective transformation matrix)
glMV     = OpenGL modelview matrix
owView   = combined view matrix from object space to world space (Lacroute: M_view)
wvConv   = conversion from world to OpenGL viewport
osPerm   = permutation from object space to standard object space (Lacroute: P)
siShear  = shear matrix from standard object space to intermediate image (Lacroute: M_shear)
oiShear  = shear matrix from object space to intermediate image space
iwWarp   = 2D warp from intermediate image to world (Lacroute: M_warp)
ivWarp   = 2D warp from intermediate image to OpenGL viewport
oViewDir = user's viewing direction [object space]
sViewDir = user's viewing direction [standard object space]
x        = matrix multiplication, order for multiple multiplications: from right to left
-1       = inverse of a matrix (e.g. owView-1 = woView)
</PRE>

Important equations:<PRE>
owView  = glPM x glMV
isShear = isShear(sViewDir, imgConv, objectSize)
osPerm  = depending on principal viewing axis
wvConv  = depending on OpenGL viewport size
oiShear = siShear x osPerm
iwWarp  = owView x soPerm x isShear
ivWarp  = wvConv x owView x soPerm x isShear
ovView  = wvConv x glMV x glPM
ovView  = wvConv x iwWarp x siShear x osPerm
</PRE>

@author Juergen Schulze-Doebold (schulze@hlrs.de)
@see vvRenderer
@see vvSoftVR
@see vvSoftImg
*/
class VIRVOEXPORT vvSoftPar : public vvSoftVR
{
   protected:
      virvo::vec3 wViewDir;                       ///< viewing direction [world space]
      virvo::vec3 oViewDir;                       ///< viewing direction [object space]
      virvo::vec3 sViewDir;                       ///< viewing direction [standard object space]
      float* bufSlice[2];                         ///< buffer slices for preintegrated rendering
      int bufSliceLen[2];                         ///< size of buffer slices
      int readSlice;                              ///< index of buffer slice currently used for reading [0..1]
      enum
      {
         VV_OP_CORR_TABLE_SIZE = 1024
      };
      float opacityCorr[VV_OP_CORR_TABLE_SIZE];
      float colorCorr[VV_OP_CORR_TABLE_SIZE];

      void compositeSliceNearest(int, int = -1, int = -1);
      void compositeSliceBilinear(int);
      void compositeSliceCompressedNearest(int);
      void compositeSliceCompressedBilinear(int);
      void compositeSlicePreIntegrated(int, int);
      void findOViewingDirection();
      void findPrincipalAxis();
      void findSViewingDirection();
      void findShearMatrix();
      void findWarpMatrix();
      void factorViewMatrix();

   public:
      vvSoftPar(vvVolDesc*, vvRenderState);
      virtual ~vvSoftPar();
      void compositeVolume(int = -1, int = -1);
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
