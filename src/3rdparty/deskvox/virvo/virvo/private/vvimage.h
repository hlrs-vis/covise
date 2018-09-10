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


#ifndef VV_PRIVATE_IMAGE_H
#define VV_PRIVATE_IMAGE_H


#include "vvexport.h"
#include "vvpixelformat.h"
#include "vvcompressedvector.h"

#include <assert.h>
#include <stddef.h>
#include <vector>


class vvSocketIO;


namespace virvo
{


class Image
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:
  // Construct an empty image
  Image() : data_(0), width_(0), height_(0), format_(PF_UNSPECIFIED), stride_(0)
  {
  }

  // Construct a new image
  VVAPI Image(int w, int h, PixelFormat format = PF_RGBA8, int stride = 0);

  // Copy-construct a new image
  VVAPI Image(unsigned char* data, int w, int h, PixelFormat format = PF_RGBA8, int stride = 0);

  // Destructor
  VVAPI virtual ~Image();

  // Returns the image buffer
  CompressedVector& data() { return data_; }

  // Returns the image buffer
  CompressedVector const& data() const { return data_; }

  // Returns the width of the image
  int width() const { return width_; }

  // Returns the height of the image
  int height() const { return height_; }

  // Returns the size in bytes of a single pixel
  PixelFormat format() const { return format_; }

  // Returns the size in bytes of a single scanline
  int stride() const { return stride_; }

  // Returns the size in bytes of the image
  // TODO: Check for overflow
  size_t size() const { return stride_ * height_; }

  // Resets the image
  VVAPI void resize(int w, int h, PixelFormat format = PF_RGBA8, int stride = 0);

  // Resets the image
  VVAPI void assign(unsigned char* data, int w, int h, PixelFormat format = PF_RGBA8, int stride = 0);

  // Compress the image
  VVAPI bool compress(CompressionType ct = Compress_JPEG);

  // Decompress the image
  VVAPI bool decompress();

private:
  void init(int w, int h, PixelFormat format = PF_RGBA8, int stride = 0);

private:
  // The image data
  CompressedVector data_;
  // Width in pixels
  int width_;
  // Height in pixels
  int height_;
  // Format of the pixels
  PixelFormat format_;
  // Size in bytes of a single scanline
  int stride_;

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & data_;
    a & width_;
    a & height_;
    a & format_;
    a & stride_;
  }
};


} // namespace virvo


#endif // VV_PRIVATE_IMAGE_H
