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

#ifndef VV_SOFTIMG_H
#define VV_SOFTIMG_H

#include "math/forward.h"
#include "vvexport.h"
#include "vvopengl.h"
#include "vvinttypes.h"

#include <boost/shared_ptr.hpp>


/** Description of pixel image.
  This class was written for the software implementation of the
  shear-warp algorithm, but it can also be used for different purposes.
  Pixel (0|0) is bottom left, pixel (max_x|0) is bottom right,
  pixel (0|max_Y) is top left, and pixel (max_x|max_y) is top right.

  @author Juergen Schulze-Doebold (schulze@hlrs.de)
  @see vvSoftPar
  @see vvSoftPer
  @see vvSoftVR
*/

class VIRVOEXPORT vvSoftImg
{
   private:
      struct Impl;
      boost::shared_ptr<Impl> impl;

      enum AlignType                              /// alignment of images
      {
         BOTTOM_LEFT
      };
      bool      warpInterpolation;                ///< true = linear interpolation, false = nearest neighbor interpolation
      bool      reinitTex;                        ///< true if texture parameters have to be (re-)set
      bool      canUsePbo;                        ///< true if GL functions for PBOs are available
   protected:
      bool      usePbo;                           ///< default: false

   public:
      static const int PIXEL_SIZE;                ///< byte per pixel (3 for RGB, 4 for RGBA)
      int     width;                              ///< width in pixels
      int     height;                             ///< height in pixels
      uchar*  data;                               ///< Pointer to scalar image data. Data arrangement: origin is bottom left,
      ///< pixels to the right are first, then up. All RGB
      ///< components are saved in a row for each pixel.
      bool    deleteData;                         ///< true if image data pointer is to be deleted when not used anymore

      vvSoftImg(int=0, int=0);
      virtual ~vvSoftImg();
      void setBuffer(uchar* buf);
      virtual void setSize(int, int);
      void initTexture(GLuint format);
      void zoom(vvSoftImg*);
      void copy(AlignType, vvSoftImg*);
      void overlay(vvSoftImg*);
      void draw();
      void drawTex();
      void clear();
      void fill(int, int, int, int);
      void drawBorder(int, int, int);
      void warp(virvo::mat4 const& w, vvSoftImg*);
      void warpTex(virvo::mat4 const& w);
      void putPixel(int, int, uint);
      void drawLine(int, int, int, int, int, int, int);
      void saveGLState();
      void restoreGLState();
      void setWarpInterpolation(bool);
      void setImageData(int, int, uchar*);
      void print(const char*);
      uint getPboName() const;
};
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
