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

#ifndef VV_CLIP_OBJ_H
#define VV_CLIP_OBJ_H

#include <cstddef>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "math/math.h"
#include "mem/allocator.h"
#include "vvexport.h"

class vvSocketIO;

//============================================================================
// Clip object base
//============================================================================

class VIRVOEXPORT vvClipObj
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:

  virtual ~vvClipObj() {}

  vvClipObj() {}
  vvClipObj(vvClipObj const&);
  vvClipObj& operator=(vvClipObj const&);

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
  }
};


//============================================================================
// Clip plane
//============================================================================

class VIRVOEXPORT vvClipPlane
    : public vvClipObj
    , public virvo::basic_plane<3, float>
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:

  static boost::shared_ptr<vvClipPlane> create();

  vvClipPlane() {}
  vvClipPlane(vvClipPlane const&);
  vvClipPlane& operator=(vvClipPlane const&);

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & boost::serialization::base_object<vvClipObj>(*this);
    a & boost::serialization::base_object<virvo::basic_plane<3, float> >(*this);
  }
};


//============================================================================
// Clip sphere
//============================================================================

class VIRVOEXPORT vvClipSphere : public vvClipObj
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:

  static boost::shared_ptr<vvClipSphere> create();

public:

  virvo::vec3 center;
  float radius;

  vvClipSphere() {}
  vvClipSphere(vvClipSphere const&);
  vvClipSphere& operator=(vvClipSphere const&);

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & boost::serialization::base_object<vvClipObj>(*this);
    a & center;
    a & radius;
  }
};


//============================================================================
// Clip cone
//============================================================================

class VIRVOEXPORT vvClipCone : public vvClipObj
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:

  static boost::shared_ptr<vvClipCone> create();

public:

  virvo::vec3 tip;  // position of the cone's tip
  virvo::vec3 axis; // unit vector pointing from tip into the cone
  float theta;      // *half* angle between axis and cone surface

public:

  vvClipCone() {}
  vvClipCone(vvClipCone const&);
  vvClipCone& operator=(vvClipCone const&);

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & boost::serialization::base_object<vvClipObj>(*this);
    a & tip;
    a & axis;
    a & theta;
  }
};


//============================================================================
// Clip triangle list
//============================================================================

class VIRVOEXPORT vvClipTriangleList : public vvClipObj
{
  friend class ::vvSocketIO; // Serialize/Deserialize

public:

  static boost::shared_ptr<vvClipTriangleList> create();

public:

  typedef struct { virvo::vec3 v1, v2, v3; } Triangle;
  typedef std::vector<Triangle, virvo::mem::aligned_allocator<Triangle, 32> > Triangles;
  typedef virvo::mat4 Matrix;

  typedef Triangles::iterator       iterator;
  typedef Triangles::const_iterator const_iterator;

  void            resize(size_t size);
  size_t          size() const;

  const Triangle* data() const;
        Triangle* data();

  const Triangle& operator[](size_t i) const;
        Triangle& operator[](size_t i);

  const_iterator  begin() const;
        iterator  begin();

  const_iterator  end() const;
        iterator  end();

  const Matrix&   transform() const;
        Matrix&   transform();

public:

  vvClipTriangleList() {}
  vvClipTriangleList(vvClipTriangleList const&);
  vvClipTriangleList& operator=(vvClipTriangleList const&);

private:

  Triangles triangles_;
  virvo::mat4 transform_;

public:
  template<class A>
  void serialize(A& a, unsigned /*version*/)
  {
    a & boost::serialization::base_object<vvClipObj>(*this);
    a & triangles_;
    a & transform_;
  }
};

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
