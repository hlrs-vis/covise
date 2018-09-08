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

#ifndef VV_TEXTUREUTIL_H
#define VV_TEXTUREUTIL_H

#include <boost/scoped_ptr.hpp>

#include "math/math.h"
#include "vvexport.h"
#include "vvinttypes.h"
#include "vvpixelformat.h"

class vvVolDesc;

namespace virvo
{

  class VIRVOEXPORT TextureUtil
  {
    public:

      //--- Types -------------------------------------------------------------

      enum ErrorType
      {
        Ok = 0,
        NumChannelsMismatch,
        Unknown
      };

      /// Channels bitfield
      typedef uint64_t Channels;

      /// Use with Channels bitfield
      enum { All=-1,
          R=1, G=2, B=4, A=8,
          RG=3, RB=5, GB=6, RA=9, GA=10, BA=12,
          RGB=7, RGA=11, RBA=13, GBA=14,
          RGBA=15 };

      /// Return type for getTexture() functions
      typedef const uint8_t* Pointer;


      //--- Functions ---------------------------------------------------------

      /**
       * @brief Constructor
       *
       * @param vd volume description, must be valid throughout object lifetime
       */
      TextureUtil(const vvVolDesc* vd);

      /**
       * @brief Destructor, for pimpl
       */
     ~TextureUtil();
  
      /**
       * @brief Obtain a memory pointer that can be used to
       *        setup a 3D texture with a GPU API like OpenGL
       *
       * @return output
       * @param tf texel format of the output texture
       * @param chans bitfield with channels to copy: default=all
       * @param frame animation frame to prepare texture for
       */
      Pointer getTexture(PixelFormat tf,
          Channels chans = All,
          int frame = 0);

      /**
       * @brief @see getTexture(), overload to obtain only a section
       *        of the texture specified by the *right-open* interval
       *          [first.x..last.x)
       *          [first.y..last.y)
       *          [first.z..last.z)
       *
       * @return output
       * @param first 3-D index of first voxel
       * @param last 3-D index of last voxel
       * @param tf texel format of the output texture
       * @param chans bitfield with channels to copy: default=all
       * @param frame animation frame to prepare texture for
       */
      Pointer getTexture(vec3i first,
          vec3i last,
          PixelFormat tf,
          Channels chans = All,
          int frame = 0);

      /**
       * @brief @see getTexture(), overload to obtain only a section
       *        of the texture specified by the *right-open* interval
       *          [first.x..last.x)
       *          [first.y..last.y)
       *          [first.z..last.z),
       *        This overload creates an RGBA texture by using
       *        individual voxels as an index into a 1-D RGBA lut
       *
       * @return output
       * @param vd volume description
       * @param first 3-D index of first voxel
       * @param last 3-D index of last voxel
       * @param rgba the RGBA lookup texture
       * @param bpcDst bytes (1|2|4) per RGBA channel in output texture
       *        (must match bpc in RGBA lut!)
       * @param frame animation frame to prepare texture for
       */
      Pointer getTexture(vec3i first,
          vec3i last,
          const uint8_t* rgba,
          int bpcDst,
          int frame = 0);

    private:

      struct Impl;
      boost::scoped_ptr<Impl> impl_;
  };

}

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
