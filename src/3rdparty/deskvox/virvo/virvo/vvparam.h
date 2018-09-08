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

#ifndef VV_PARAM_H
#define VV_PARAM_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "math/math.h"

#include "vvclipobj.h"
#include "vvexport.h"
#include "vvinttypes.h"
#include "vvcolor.h"

#include <cassert>
#include <stdexcept>

#include <boost/serialization/split_member.hpp>
#include <boost/any.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>

class vvParam
{
public:
  enum Type {
    VV_EMPTY,
    VV_BOOL,
    VV_CHAR,
    VV_UCHAR,
    VV_SHORT,
    VV_USHORT,
    VV_INT,
    VV_UINT,
    VV_LONG,
    VV_ULONG,
    VV_LLONG,
    VV_ULLONG,
    VV_FLOAT,
    VV_VEC2F,
    VV_VEC2I,
    VV_VEC3F,
    VV_VEC3D,
    VV_VEC3S,
    VV_VEC3US,
    VV_VEC3I,
    VV_VEC3UI,
    VV_VEC3L,
    VV_VEC3UL,
    VV_VEC3LL,
    VV_VEC3ULL,
    VV_VEC4F,
    VV_COLOR,
    VV_AABBF,
    VV_AABBD,
    VV_AABBI,
    VV_AABBUI,
    VV_AABBL,
    VV_AABBUL,
    VV_AABBLL,
    VV_AABBULL,
    VV_CLIP_OBJ
  };

private:
  // The type of this parameter
  Type type;
  // The value of this parameter
  boost::any value;

private:
  template<class T, class A>
  static void save_value(A& a, T const& x)
  {
    a & x;
  }

  template<class T, class A>
  void save_as(A& a) const
  {
    T x = boost::any_cast<T>(value);
    a & x;
  }

  template<class T, class A>
  static T load_value(A& a)
  {
    T x; a & x; return x;
  }

  template<class T, class A>
  void load_as(A& a)
  {
    T x; a & x; value = x;
  }

public:
  //--- serialization ------------------------------------------------------------------------------

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  template<class A>
  void save(A& a, unsigned /*version*/) const
  {
    save_value(a, static_cast<unsigned>(type));

    switch (type)
    {
    case VV_EMPTY:    /* DO NOTHING */                                        return;
    case VV_BOOL:     save_as< bool                                     >(a); return;
    case VV_CHAR:     save_as< char                                     >(a); return;
    case VV_UCHAR:    save_as< unsigned char                            >(a); return;
    case VV_SHORT:    save_as< short                                    >(a); return;
    case VV_USHORT:   save_as< unsigned short                           >(a); return;
    case VV_INT:      save_as< int                                      >(a); return;
    case VV_UINT:     save_as< unsigned                                 >(a); return;
    case VV_LONG:     save_as< long                                     >(a); return;
    case VV_ULONG:    save_as< unsigned long                            >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    save_as< long long                                >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   save_as< unsigned long long                       >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    save_as< float                                    >(a); return;
    case VV_VEC2F:    save_as< virvo::vector< 2, float >                >(a); return;
    case VV_VEC2I:    save_as< virvo::vector< 2, int >                  >(a); return;
    case VV_VEC3F:    save_as< virvo::vector< 3, float >                >(a); return;
    case VV_VEC3D:    save_as< virvo::vector< 3, double >               >(a); return;
    case VV_VEC3S:    save_as< virvo::vector< 3, short >                >(a); return;
    case VV_VEC3US:   save_as< virvo::vector< 3, unsigned short >       >(a); return;
    case VV_VEC3I:    save_as< virvo::vector< 3, int >                  >(a); return;
    case VV_VEC3UI:   save_as< virvo::vector< 3, unsigned int >         >(a); return;
    case VV_VEC3L:    save_as< virvo::vector< 3, long >                 >(a); return;
    case VV_VEC3UL:   save_as< virvo::vector< 3, unsigned long >        >(a); return;
    case VV_VEC3LL:   save_as< virvo::vector< 3, long long >            >(a); return;
    case VV_VEC3ULL:  save_as< virvo::vector< 3, unsigned long long >   >(a); return;
    case VV_VEC4F:    save_as< virvo::vector< 4, float >                >(a); return;
    case VV_COLOR:    save_as< vvColor                                  >(a); return;
    case VV_AABBF:    save_as< virvo::basic_aabb< float >               >(a); return;
    case VV_AABBD:    save_as< virvo::basic_aabb< double >              >(a); return;
    case VV_AABBI:    save_as< virvo::basic_aabb< int >                 >(a); return;
    case VV_AABBUI:   save_as< virvo::basic_aabb< unsigned int >        >(a); return;
    case VV_AABBL:    save_as< virvo::basic_aabb< long >                >(a); return;
    case VV_AABBUL:   save_as< virvo::basic_aabb< unsigned long >       >(a); return;
    case VV_AABBLL:   save_as< virvo::basic_aabb< long long >           >(a); return;
    case VV_AABBULL:  save_as< virvo::basic_aabb< unsigned long long >  >(a); return;
    case VV_CLIP_OBJ: save_as< boost::shared_ptr< vvClipObj >           >(a); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to serialize parameter");
  }

  template<class A>
  void load(A& a, unsigned /*version*/)
  {
    type = static_cast<Type>(load_value<unsigned>(a));

    switch (type)
    {
    case VV_EMPTY:    value = boost::any();                                   return;
    case VV_BOOL:     load_as< bool                                     >(a); return;
    case VV_CHAR:     load_as< char                                     >(a); return;
    case VV_UCHAR:    load_as< unsigned char                            >(a); return;
    case VV_SHORT:    load_as< short                                    >(a); return;
    case VV_USHORT:   load_as< unsigned short                           >(a); return;
    case VV_INT:      load_as< int                                      >(a); return;
    case VV_UINT:     load_as< unsigned                                 >(a); return;
    case VV_LONG:     load_as< long                                     >(a); return;
    case VV_ULONG:    load_as< unsigned long                            >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    load_as< long long                                >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   load_as< unsigned long long                       >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    load_as< float                                    >(a); return;
    case VV_VEC2F:    load_as< virvo::vector< 2, float >                >(a); return;
    case VV_VEC2I:    load_as< virvo::vector< 2, int >                  >(a); return;
    case VV_VEC3F:    load_as< virvo::vector< 3, float >                >(a); return;
    case VV_VEC3D:    load_as< virvo::vector< 3, double >               >(a); return;
    case VV_VEC3S:    load_as< virvo::vector< 3, short >                >(a); return;
    case VV_VEC3US:   load_as< virvo::vector< 3, unsigned short >       >(a); return;
    case VV_VEC3I:    load_as< virvo::vector< 3, int >                  >(a); return;
    case VV_VEC3UI:   load_as< virvo::vector< 3, unsigned int >         >(a); return;
    case VV_VEC3L:    load_as< virvo::vector< 3, long  >                >(a); return;
    case VV_VEC3UL:   load_as< virvo::vector< 3, unsigned long >        >(a); return;
    case VV_VEC3LL:   load_as< virvo::vector< 3, long long >            >(a); return;
    case VV_VEC3ULL:  load_as< virvo::vector< 3, unsigned long long >   >(a); return;
    case VV_VEC4F:    load_as< virvo::vector< 3, float >                >(a); return;
    case VV_COLOR:    load_as< vvColor                                  >(a); return;
    case VV_AABBF:    load_as< virvo::basic_aabb< float >               >(a); return;
    case VV_AABBD:    load_as< virvo::basic_aabb< double >              >(a); return;
    case VV_AABBI:    load_as< virvo::basic_aabb< int >                 >(a); return;
    case VV_AABBUI:   load_as< virvo::basic_aabb< unsigned int >        >(a); return;
    case VV_AABBL:    load_as< virvo::basic_aabb< long >                >(a); return;
    case VV_AABBUL:   load_as< virvo::basic_aabb< unsigned long >       >(a); return;
    case VV_AABBLL:   load_as< virvo::basic_aabb< long long >           >(a); return;
    case VV_AABBULL:  load_as< virvo::basic_aabb< unsigned long long >  >(a); return;
    case VV_CLIP_OBJ: load_as< boost::shared_ptr< vvClipObj >           >(a); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to deserialize parameter");
  }

  //------------------------------------------------------------------------------------------------

public:
  VVAPI vvParam();
  VVAPI vvParam(const bool& val);
  VVAPI vvParam(const char& val);
  VVAPI vvParam(const unsigned char& val);
  VVAPI vvParam(const short& val);
  VVAPI vvParam(const unsigned short& val);
  VVAPI vvParam(const int& val);
  VVAPI vvParam(const unsigned& val);
  VVAPI vvParam(const long& val);
  VVAPI vvParam(const unsigned long& val);
#if VV_HAVE_LLONG
  VVAPI vvParam(const long long& val);
#endif
#if VV_HAVE_ULLONG
  VVAPI vvParam(const unsigned long long& val);
#endif
  VVAPI vvParam(const float& val);
  VVAPI vvParam(virvo::vector< 2, float > const& val);
  VVAPI vvParam(virvo::vector< 2, int > const& val);
  VVAPI vvParam(virvo::vector< 3, float > const& val);
  VVAPI vvParam(virvo::vector< 3, double > const& val);
  VVAPI vvParam(virvo::vector< 3, short > const& val);
  VVAPI vvParam(virvo::vector< 3, unsigned short > const& val);
  VVAPI vvParam(virvo::vector< 3, int > const& val);
  VVAPI vvParam(virvo::vector< 3, unsigned int > const& val);
  VVAPI vvParam(virvo::vector< 3, long > const& val);
  VVAPI vvParam(virvo::vector< 3, unsigned long > const& val);
  VVAPI vvParam(virvo::vector< 3, long long > const& val);
  VVAPI vvParam(virvo::vector< 3, unsigned long long > const& val);
  VVAPI vvParam(virvo::vector< 4, float > const& val);
  VVAPI vvParam(const vvColor& val);
  VVAPI vvParam(virvo::basic_aabb< float > const& val);
  VVAPI vvParam(virvo::basic_aabb< double > const& val);
  VVAPI vvParam(virvo::basic_aabb< int > const& val);
  VVAPI vvParam(virvo::basic_aabb< unsigned int > const& val);
  VVAPI vvParam(virvo::basic_aabb< long > const& val);
  VVAPI vvParam(virvo::basic_aabb< unsigned long > const& val);
  VVAPI vvParam(virvo::basic_aabb< long long > const& val);
  VVAPI vvParam(virvo::basic_aabb< unsigned long long > const& val);
  VVAPI vvParam(boost::shared_ptr< vvClipObj > const& val);
  VVAPI vvParam(boost::shared_ptr< vvClipPlane > const& val);
  VVAPI vvParam(boost::shared_ptr< vvClipSphere > const& val);
  VVAPI vvParam(boost::shared_ptr< vvClipCone > const& val);
  VVAPI vvParam(boost::shared_ptr< vvClipTriangleList > const& val);

  VVAPI bool asBool() const;
  VVAPI char asChar() const;
  VVAPI unsigned char asUchar() const;
  VVAPI short asShort() const;
  VVAPI unsigned short asUshort() const;
  VVAPI int asInt() const;
  VVAPI unsigned int asUint() const;
  VVAPI long asLong() const;
  VVAPI unsigned long asUlong() const;
#if VV_HAVE_LLONG
  VVAPI long long asLlong() const;
#endif
#if VV_HAVE_ULLONG
  VVAPI unsigned long long asUllong() const;
#endif
  VVAPI float asFloat() const;
  VVAPI virvo::vector< 2, float > asVec2f() const;
  VVAPI virvo::vector< 2, int > asVec2i() const;
  VVAPI virvo::vector< 3, float > asVec3f() const;
  VVAPI virvo::vector< 3, double > asVec3d() const;
  VVAPI virvo::vector< 3, short > asVec3s() const;
  VVAPI virvo::vector< 3, unsigned short > asVec3us() const;
  VVAPI virvo::vector< 3, int > asVec3i() const;
  VVAPI virvo::vector< 3, unsigned int > asVec3ui() const;
  VVAPI virvo::vector< 3, long > asVec3l() const;
  VVAPI virvo::vector< 3, unsigned long > asVec3ul() const;
  VVAPI virvo::vector< 3, long long > asVec3ll() const;
  VVAPI virvo::vector< 3, unsigned long long > asVec3ull() const;
  VVAPI virvo::vector< 4, float > asVec4f() const;
  VVAPI vvColor asColor() const;
  VVAPI virvo::basic_aabb< float > asAABBf() const;
  VVAPI virvo::basic_aabb< double > asAABBd() const;
  VVAPI virvo::basic_aabb< int > asAABBi() const;
  VVAPI virvo::basic_aabb< unsigned int > asAABBui() const;
  VVAPI virvo::basic_aabb< long > asAABBl() const;
  VVAPI virvo::basic_aabb< unsigned long > asAABBul() const;
  VVAPI virvo::basic_aabb< long long > asAABBll() const;
  VVAPI virvo::basic_aabb< unsigned long long >  asAABBull() const;
  VVAPI boost::shared_ptr< vvClipObj > asClipObj() const;

  VVAPI operator bool() const;
  VVAPI operator char() const;
  VVAPI operator unsigned char() const;
  VVAPI operator short() const;
  VVAPI operator unsigned short() const;
  VVAPI operator int() const;
  VVAPI operator unsigned int() const;
  VVAPI operator long() const;
  VVAPI operator unsigned long() const;
#if VV_HAVE_LLONG
  VVAPI operator long long() const;
#endif
#if VV_HAVE_ULLONG
  VVAPI operator unsigned long long() const;
#endif
  VVAPI operator float() const;
  VVAPI operator virvo::vector< 2, float >() const;
  VVAPI operator virvo::vector< 2, int >() const;
  VVAPI operator virvo::vector< 3, float >() const;
  VVAPI operator virvo::vector< 3, double >() const;
  VVAPI operator virvo::vector< 3, short >() const;
  VVAPI operator virvo::vector< 3, unsigned short >() const;
  VVAPI operator virvo::vector< 3, int >() const;
  VVAPI operator virvo::vector< 3, unsigned int >() const;
  VVAPI operator virvo::vector< 3, long >() const;
  VVAPI operator virvo::vector< 3, unsigned long >() const;
  VVAPI operator virvo::vector< 3, long long >() const;
  VVAPI operator virvo::vector< 3, unsigned long long >() const;
  VVAPI operator virvo::vector< 4, float >() const;
  VVAPI operator vvColor() const;
  VVAPI operator virvo::basic_aabb< float >() const;
  VVAPI operator virvo::basic_aabb< double >() const;
  VVAPI operator virvo::basic_aabb< int >() const;
  VVAPI operator virvo::basic_aabb< unsigned int >() const;
  VVAPI operator virvo::basic_aabb< long >() const;
  VVAPI operator virvo::basic_aabb< unsigned long >() const;
  VVAPI operator virvo::basic_aabb< long long >() const;
  VVAPI operator virvo::basic_aabb< unsigned long long >() const;
  VVAPI operator boost::shared_ptr< vvClipObj >() const;

  // Returns the type of this parameter
  Type getType() const {
    return type;
  }

  // Returns whether this parameter is of type t
  bool isa(Type t) const {
    return type == t;
  }
};

#endif
