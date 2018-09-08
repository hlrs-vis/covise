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


#include "vvibrimage.h"
#include "vvcompress.h"


using virvo::IbrImage;
using virvo::PixelFormat;


IbrImage::IbrImage()
  : color_()
  , depth_()
  , depthMin_(0.0f)
  , depthMax_(0.0f)
  , viewMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , projMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , viewport_(0, 0, 0, 0)
{
}


IbrImage::IbrImage(int w, int h, PixelFormat colorFormat, PixelFormat depthFormat)
  : color_(w, h, colorFormat)
  , depth_(w, h, depthFormat)
  , depthMin_(0.0f)
  , depthMax_(1.0f)
  , viewMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , projMatrix_(1.0f, 1.0f, 1.0f, 1.0f)
  , viewport_(0, 0, w, h)
{
}


IbrImage::~IbrImage()
{
}


bool IbrImage::compress(virvo::CompressionType ctColor, virvo::CompressionType ctDepth)
{
  // Compress the color buffer
  if (!color_.compress(ctColor))
    return false;

  // Compress the depth buffer
  if (!depth_.compress(ctDepth))
    return false;

  return true;
}


bool IbrImage::decompress()
{
  // Decompress the color buffer
  if (!color_.decompress())
    return false;

  // Decompress the depth buffer
  if (!depth_.decompress())
    return false;

  return true;
}
