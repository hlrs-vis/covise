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

#include <boost/make_shared.hpp>

#include "vvclipobj.h"


//============================================================================
// Provide access to private default constructor
//============================================================================

template <typename Base>
struct access_private : Base {};


//============================================================================
// Clip plane
//============================================================================

boost::shared_ptr<vvClipPlane> vvClipPlane::create()
{
  return boost::make_shared<access_private<vvClipPlane> >();
}


//============================================================================
// Clip sphere
//============================================================================

boost::shared_ptr<vvClipSphere> vvClipSphere::create()
{
  return boost::make_shared<access_private<vvClipSphere> >();
}


//============================================================================
// Clip cone
//============================================================================

boost::shared_ptr<vvClipCone> vvClipCone::create()
{
  return boost::make_shared<access_private<vvClipCone> >();
}


//============================================================================
// Clip triangle list
//============================================================================

boost::shared_ptr<vvClipTriangleList> vvClipTriangleList::create()
{
  return boost::make_shared<access_private<vvClipTriangleList> >();
}

void vvClipTriangleList::resize(size_t size)
{
  triangles_.resize(size);
}

size_t vvClipTriangleList::size() const
{
  return triangles_.size();
}

const vvClipTriangleList::Triangle* vvClipTriangleList::data() const
{
  return triangles_.data();
}

vvClipTriangleList::Triangle* vvClipTriangleList::data()
{
  return triangles_.data();
}

const vvClipTriangleList::Triangle& vvClipTriangleList::operator[](size_t i) const
{
    return triangles_[i];
}

vvClipTriangleList::Triangle& vvClipTriangleList::operator[](size_t i)
{
  return triangles_[i];
}


vvClipTriangleList::const_iterator vvClipTriangleList::begin() const
{
  return triangles_.begin();
}

vvClipTriangleList::iterator vvClipTriangleList::begin()
{
  return triangles_.begin();
}

vvClipTriangleList::const_iterator vvClipTriangleList::end() const
{
  return triangles_.end();
}

vvClipTriangleList::iterator vvClipTriangleList::end()
{
  return triangles_.end();
}

const vvClipTriangleList::Matrix& vvClipTriangleList::transform() const
{
  return transform_;
}

vvClipTriangleList::Matrix& vvClipTriangleList::transform()
{
  return transform_;
}


//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
