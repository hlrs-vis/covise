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


#ifndef VV_PRIVATE_IBR_IMAGE_H
#define VV_PRIVATE_IBR_IMAGE_H


#include "math/math.h"

#include "vvimage.h"


class vvSocketIO;


namespace virvo
{


class IbrImage
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:
  // Construct an empty (invalid) IBR image
  VVAPI IbrImage();

  // Construct a new IBR image
  VVAPI IbrImage(int w, int h, PixelFormat colorFormat, PixelFormat depthFormat);

  // Destructor
  VVAPI virtual ~IbrImage();

  // Returns the color buffer
  Image& colorBuffer() { return color_; }

  // Returns the color buffer
  Image const& colorBuffer() const { return color_; }

  // Returns the depth buffer
  Image& depthBuffer() { return depth_; }

  // Returns the depth buffer
  Image const& depthBuffer() const { return depth_; }

  // Returns the width of the IBR frame
  int width() const
  {
    assert( colorBuffer().width() == depthBuffer().width() );
    return colorBuffer().width();
  }

  // Returns the height of the IBR frame
  int height() const
  {
    assert( colorBuffer().height() == depthBuffer().height() );
    return colorBuffer().height();
  }

  // Returns the minimum depth value
  float depthMin() const { return depthMin_; }

  // Sets the minimum depth value
  void setDepthMin(float value) { depthMin_ = value; }

  // Returns the maximum depth value
  float depthMax() const { return depthMax_; }

  // Sets the maximum depth value
  void setDepthMax(float value) { depthMax_ = value; }

  // Returns the model-view matrix
  virvo::mat4 const& viewMatrix() const { return viewMatrix_; }

  // Sets the model-view matrix
  void setViewMatrix(virvo::mat4 const& value) { viewMatrix_ = value; }

  // Returns the projection matrix
  virvo::mat4 const& projMatrix() const { return projMatrix_; }

  // Sets the projection matrix
  void setProjMatrix(virvo::mat4 const& value) { projMatrix_ = value; }

  // Returns the viewport
  virvo::recti const& viewport() const { return viewport_; }

  // Sets the viewport
  void setViewport(virvo::recti const& value) { viewport_ = value; }

  // Compress the image
  VVAPI bool compress(CompressionType ctColor = Compress_Snappy, CompressionType ctDepth = Compress_Snappy);

  // Decompress the image
  VVAPI bool decompress();

private:
  // The color buffer
  Image color_;
  // The depth buffer
  Image depth_;
  // Depth range
  float depthMin_;
  float depthMax_;
  // View matrix
  virvo::mat4 viewMatrix_;
  // Projection matrix
  virvo::mat4 projMatrix_;
  // The viewport
  virvo::recti viewport_;

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & color_;
    a & depth_;
    a & depthMin_;
    a & depthMax_;
    a & viewMatrix_;
    a & projMatrix_;
    a & viewport_;
  }
};


} // namespace virvo


#endif // VV_PRIVATE_IBR_IMAGE_H
