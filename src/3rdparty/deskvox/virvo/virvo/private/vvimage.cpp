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


#include "vvimage.h"
#include "vvcompress.h"


using virvo::Image;
using virvo::PixelFormat;


Image::Image(int w, int h, PixelFormat format, int stride)
{
  resize(w, h, format, stride);
}


Image::Image(unsigned char* data, int w, int h, PixelFormat format, int stride)
{
  assign(data, w, h, format, stride);
}


Image::~Image()
{
}


void Image::resize(int w, int h, PixelFormat format, int stride)
{
  init(w, h, format, stride);
  data_.resize(size());
}


void Image::assign(unsigned char* data, int w, int h, PixelFormat format, int stride)
{
  assert( data );

  init(w, h, format, stride);
  data_.assign(data, data + size());
}


bool Image::compress(virvo::CompressionType ct)
{
  assert( data_.getCompressionType() == Compress_None );

  switch (ct)
  {
  case Compress_None:
    return true;

  case Compress_JPEG:
    {
      virvo::JPEGOptions options;

      options.format  = this->format();
      options.w       = this->width();
      options.h       = this->height();
      options.pitch   = this->stride();
      options.quality = 75; // TODO: Make this an option?!?!

      if (virvo::encodeJPEG(data_, options))
        return true;

      // JPEG compression failed. Try Snappy.
      return compress(Compress_Snappy);
    }

  case Compress_PNG:
    {
      virvo::PNGOptions options;

      options.format  = this->format();
      options.w       = this->width();
      options.h       = this->height();
      options.pitch   = this->stride();
      options.compression_level = 5;

      if (virvo::encodePNG(data_, options))
        return true;

      // PNG compression failed. Try Snappy.
      return compress(Compress_Snappy);
    }

  case Compress_Snappy:
    return virvo::encodeSnappy(data_);
  }

  return false;
}


bool Image::decompress()
{
  switch (data_.getCompressionType())
  {
  case Compress_None:
    return true;

  case Compress_JPEG:
    {
      virvo::JPEGOptions options;

      options.format  = this->format();
      options.w       = this->width();
      options.h       = this->height();
      options.pitch   = this->stride();

      return virvo::decodeJPEG(data_, options);
    }

  case Compress_PNG:
    {
      virvo::PNGOptions options;

      options.format  = this->format();
      options.w       = this->width();
      options.h       = this->height();
      options.pitch   = this->stride();

      return virvo::decodePNG(data_, options);
    }

  case Compress_Snappy:
    return virvo::decodeSnappy(data_);
  }

  return false;
}


void Image::init(int w, int h, PixelFormat format, int stride)
{
  assert( w > 0 );
  assert( h > 0 );
  assert( format != PF_UNSPECIFIED );

  width_ = w;
  height_ = h;
  format_ = format;
  stride_ = stride <= 0 ? w * getPixelSize(format) : stride;
}
