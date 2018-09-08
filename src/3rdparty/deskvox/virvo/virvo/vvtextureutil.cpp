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

#include <bitset>
#include <cstring> // memcpy
#include <vector>

#include "vvtextureutil.h"
#include "vvvoldesc.h"

namespace virvo
{

  //--- Helpers ---------------------------------------------------------------

  // Determine native texel format of voldesc
  PixelFormat nativeFormat(const vvVolDesc* vd)
  {
    if (vd->bpc == 1 && vd->getChan() == 1)
    {
      return PF_R8;
    }
    else if (vd->bpc == 1 && vd->getChan() == 2)
    {
      return PF_RG8;
    }
    else if (vd->bpc == 1 && vd->getChan() == 3)
    {
      return PF_RGB8;
    }
    else if (vd->bpc == 1 && vd->getChan() == 4)
    {
      return PF_RGBA8;
    }
    else if (vd->bpc == 2 && vd->getChan() == 1)
    {
      return PF_R16UI;
    }
    else if (vd->bpc == 2 && vd->getChan() == 2)
    {
      return PF_RG16UI;
    }
    else if (vd->bpc == 2 && vd->getChan() == 3)
    {
      return PF_RGB16UI;
    }
    else if (vd->bpc == 2 && vd->getChan() == 4)
    {
      return PF_RGBA16UI;
    }
    else if (vd->bpc == 4 && vd->getChan() == 1)
    {
      return PF_R32F;
    }
    else if (vd->bpc == 4 && vd->getChan() == 2)
    {
      return PF_RG32F;
    }
    else if (vd->bpc == 4 && vd->getChan() == 3)
    {
      return PF_RGB32F;
    }
    else if (vd->bpc == 4 && vd->getChan() == 4)
    {
      return PF_RGBA32F;
    }
    else
    {
      // Unsupported!
      return PixelFormat(-1);
    }
  }

  size_t computeTextureSize(vec3i first, vec3i last, PixelFormat tf)
  {
    PixelFormatInfo info = mapPixelFormat(tf);

    // Employ some sanity checks
    // TODO...

    vec3i size = last - first;
    return size.x * size.y * size.z * info.size;
  }


  //--- Private impl ----------------------------------------------------------

  struct TextureUtil::Impl
  {
    Impl(const vvVolDesc* vd) : vd(vd) {}

    // The volume description
    const vvVolDesc* vd;

    // Memory to hold the texture, in case we need it
    std::vector<uint8_t> mem;
  };


  //--- Interface -------------------------------------------------------------

  TextureUtil::TextureUtil(const vvVolDesc* vd)
    : impl_(new Impl(vd))
  {
  }

  TextureUtil::~TextureUtil()
  {
  }

  TextureUtil::Pointer TextureUtil::getTexture(PixelFormat tf,
      TextureUtil::Channels chans,
      int frame)
  {
    return getTexture(vec3i(0),
        vec3i(impl_->vd->vox),
        tf,
        chans,
        frame);
  }

  TextureUtil::Pointer TextureUtil::getTexture(vec3i first,
      vec3i last,
      PixelFormat tf,
      TextureUtil::Channels chans,
      int frame)
  {
    PixelFormatInfo info = mapPixelFormat(tf);

    const vvVolDesc* vd = impl_->vd;


    //--- Sanity checks ---------------

    // More than 4 channels: user needs to explicitly state
    // which channels (s)he's interested in
    if (vd->getChan() > 4 && chans == All)
      return NULL;

    // TODO: requires C++11 std::bitset(ull) ctor!
    if (chans != All && info.components != std::bitset<64>(chans).count())
      return NULL;


    //--- Make texture ----------------

    // Maybe we can just return a pointer from the voldesc
    if (nativeFormat(vd) == tf && first.xy() == vec2i(0) && last.xy() == vec2i(vd->vox.xy()))
    {
      return vd->getRaw(frame) + first.z * vd->getSliceBytes();
    }

    // Maybe the conversion operation is trivial and we can
    // copy over sections of the volume data
    if (nativeFormat(vd) == tf)
    {
      // Reserve memory
      impl_->mem.resize(computeTextureSize(first, last, tf));

      const uint8_t* raw = vd->getRaw(frame);
      uint8_t* dst = &impl_->mem[0];

      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            memcpy(dst, raw, vd->getBPV());
            raw += vd->getBPV();
            dst += vd->getBPV();
          }
        }
      }

      return &impl_->mem[0];
    }

    // No use, have to iterate over all voxels
    // TODO: support N-byte
    if (info.size / info.components == 1/*byte*/)
    {
      // Reserve memory
      impl_->mem.resize(computeTextureSize(first, last, tf));

      const uint8_t* raw = vd->getRaw(frame);
      uint8_t* dst = &impl_->mem[0];

      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            for (int c = 0; c < vd->getChan(); ++c)
            {
              if ((chans >> c) & 1)
              {
                *dst++ = static_cast<uint8_t>(vd->rescaleVoxel(raw, 1/*byte*/, c));
              }

              raw += vd->bpc;
            }
          }
        }
      }

      return &impl_->mem[0];
    }

    // Unsupported, error unknown
    return NULL;
  }

  TextureUtil::Pointer TextureUtil::getTexture(vec3i first,
      vec3i last,
      const uint8_t* rgba,
      int bpcDst,
      int frame)
  {
    const vvVolDesc* vd = impl_->vd;

    // This is only supported if volume has <= 4 channels
    if (vd->getChan() > 4)
      return NULL;

    // Single channel: rescale voxel to 8-bit, use as index into RGBA lut
    if (vd->getChan() == 1)
    {
      // Reserve memory
      impl_->mem.resize(computeTextureSize(first, last, PF_RGBA8));

      const uint8_t* raw = vd->getRaw(frame);
      uint8_t* dst = &impl_->mem[0];

      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            int index = vd->rescaleVoxel(raw, 1/*byte*/, 0) * bpcDst;
            dst[0] = rgba[index * 4];
            dst[1] = rgba[index * 4 + 1];
            dst[2] = rgba[index * 4 + 2];
            dst[3] = rgba[index * 4 + 3];
            raw += vd->bpc;
            dst += bpcDst * 4;
          }
        }
      }

      return &impl_->mem[0];
    }

    // Two or three channels: RG(B) values come from 3-D texture,
    // calculate alpha as mean of sum of RG(B) conversion table results
    if (vd->getChan() == 2 || vd->getChan() == 3)
    {
      // TODO: only implemented for RGBA8 lut!
      if (bpcDst != 1)
        return NULL;

      // Reserve memory
      impl_->mem.resize(computeTextureSize(first, last, PF_RGBA8));

      const uint8_t* raw = vd->getRaw(frame);
      uint8_t* dst = &impl_->mem[0];

      for (int z = first.z; z != last.z; ++z)
      {
        for (int y = first.y; y != last.y; ++y)
        {
          for (int x = first.x; x != last.x; ++x)
          {
            int alpha = 0;
            for (int c = 0; c < vd->getChan(); ++c)
            {
              uint8_t index = vd->rescaleVoxel(raw, 1/*byte*/, c);
              alpha += static_cast<int>(rgba[index * 4 + c]);
              dst[c] = index;
              raw += vd->bpc;
            }

            dst[3] = static_cast<uint8_t>(alpha / vd->getChan());
            dst += 4;
          }
        }
      }

      return &impl_->mem[0];
    }

    // Four channels: just skip the RGBA lut.
    // TODO: this is legacy behavior, but is it actually desired??
    if (vd->getChan() == 4)
    {
      return getTexture(first,
          last,
          nativeFormat(vd),
          RGBA,
          frame);
    }

    // Unsupported, error unknown
    return NULL;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
