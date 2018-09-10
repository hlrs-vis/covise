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

#include "vvparam.h"

vvParam::vvParam() : type(VV_EMPTY)
{
}

vvParam::vvParam(const bool& val)
  : type(VV_BOOL)
  , value(val)
{
}

vvParam::vvParam(const char& val)
  : type(VV_CHAR)
  , value(val)
{
}

vvParam::vvParam(const unsigned char& val)
  : type(VV_UCHAR)
  , value(val)
{
}

vvParam::vvParam(const short& val)
  : type(VV_SHORT)
  , value(val)
{
}

vvParam::vvParam(const unsigned short& val)
  : type(VV_USHORT)
  , value(val)
{
}

vvParam::vvParam(const int& val)
  : type(VV_INT)
  , value(val)
{
}

vvParam::vvParam(const unsigned& val)
  : type(VV_UINT)
  , value(val)
{
}

vvParam::vvParam(const long& val)
  : type(VV_LONG)
  , value(val)
{
}

vvParam::vvParam(const unsigned long& val)
  : type(VV_ULONG)
  , value(val)
{
}

#if VV_HAVE_LLONG
vvParam::vvParam(const long long& val)
  : type(VV_LLONG)
  , value(val)
{
}
#endif

#if VV_HAVE_ULLONG
vvParam::vvParam(const unsigned long long& val)
  : type(VV_ULLONG)
  , value(val)
{
}
#endif

vvParam::vvParam(const float& val)
  : type(VV_FLOAT)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 2, float > const& val)
  : type(VV_VEC2F)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 2, int > const& val)
  : type(VV_VEC2I)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, float > const& val)
  : type(VV_VEC3F)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, double > const& val)
  : type(VV_VEC3D)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, short > const& val)
  : type(VV_VEC3S)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, unsigned short > const& val)
  : type(VV_VEC3US)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, int > const& val)
  : type(VV_VEC3I)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, unsigned int > const& val)
  : type(VV_VEC3UI)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, long > const& val)
  : type(VV_VEC3L)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, unsigned long > const& val)
  : type(VV_VEC3UL)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, long long > const& val)
  : type(VV_VEC3LL)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 3, unsigned long long > const& val)
  : type(VV_VEC3ULL)
  , value(val)
{
}

vvParam::vvParam(virvo::vector< 4, float > const& val)
  : type(VV_VEC4F)
  , value(val)
{
}

vvParam::vvParam(const vvColor& val)
  : type(VV_COLOR)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< float > const& val)
  : type(VV_AABBF)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< double > const& val)
  : type(VV_AABBD)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< int > const& val)
  : type(VV_AABBI)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< unsigned int > const& val)
  : type(VV_AABBUI)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< long > const& val)
  : type(VV_AABBL)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< unsigned long > const& val)
  : type(VV_AABBUL)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< long long > const& val)
  : type(VV_AABBLL)
  , value(val)
{
}

vvParam::vvParam(virvo::basic_aabb< unsigned long long > const& val)
  : type(VV_AABBULL)
  , value(val)
{
}

vvParam::vvParam(boost::shared_ptr< vvClipObj > const& val)
  : type(VV_CLIP_OBJ)
  , value(val)
{
}

vvParam::vvParam(boost::shared_ptr< vvClipPlane > const& val)
  : type(VV_CLIP_OBJ)
  , value(boost::static_pointer_cast<vvClipObj>(val))
{
}

vvParam::vvParam(boost::shared_ptr< vvClipSphere > const& val)
  : type(VV_CLIP_OBJ)
  , value(boost::static_pointer_cast<vvClipObj>(val))
{
}

vvParam::vvParam(boost::shared_ptr< vvClipCone > const& val)
  : type(VV_CLIP_OBJ)
  , value(boost::static_pointer_cast<vvClipObj>(val))
{
}

vvParam::vvParam(boost::shared_ptr< vvClipTriangleList > const& val)
  : type(VV_CLIP_OBJ)
  , value(boost::static_pointer_cast<vvClipObj>(val))
{
}

bool vvParam::asBool() const {
  return boost::any_cast<bool>(value);
}

char vvParam::asChar() const {
  return boost::any_cast<char>(value);
}

unsigned char vvParam::asUchar() const {
  return boost::any_cast<unsigned char>(value);
}

short vvParam::asShort() const {
  return boost::any_cast<short>(value);
}

unsigned short vvParam::asUshort() const {
  return boost::any_cast<unsigned short>(value);
}

int vvParam::asInt() const {
  return boost::any_cast<int>(value);
}

unsigned int vvParam::asUint() const {
  return boost::any_cast<unsigned int>(value);
}

long vvParam::asLong() const {
  return boost::any_cast<long>(value);
}

unsigned long vvParam::asUlong() const {
  return boost::any_cast<unsigned long>(value);
}

#if VV_HAVE_LLONG
long long vvParam::asLlong() const {
  return boost::any_cast<long long>(value);
}
#endif

#if VV_HAVE_ULLONG
unsigned long long vvParam::asUllong() const {
  return boost::any_cast<unsigned long long>(value);
}
#endif

float vvParam::asFloat() const {
  return boost::any_cast<float>(value);
}

virvo::vector< 2, float > vvParam::asVec2f() const {
  return boost::any_cast< virvo::vector< 2, float > >(value);
}

virvo::vector< 2, int > vvParam::asVec2i() const {
  return boost::any_cast< virvo::vector< 2, int > >(value);
}

virvo::vector< 3, float > vvParam::asVec3f() const {
  return boost::any_cast< virvo::vector< 3, float > >(value);
}

virvo::vector< 3, double > vvParam::asVec3d() const {
  return boost::any_cast< virvo::vector< 3, double > >(value);
}

virvo::vector< 3, short > vvParam::asVec3s() const {
  return boost::any_cast< virvo::vector< 3, short > >(value);
}

virvo::vector< 3, unsigned short > vvParam::asVec3us() const {
  return boost::any_cast< virvo::vector< 3, unsigned short > >(value);
}

virvo::vector< 3, int > vvParam::asVec3i() const {
  return boost::any_cast< virvo::vector< 3, int > >(value);
}

virvo::vector< 3, unsigned int > vvParam::asVec3ui() const {
  return boost::any_cast< virvo::vector< 3, unsigned int > >(value);
}

virvo::vector< 3, long > vvParam::asVec3l() const {
  return boost::any_cast< virvo::vector< 3, long > >(value);
}

virvo::vector< 3, unsigned long > vvParam::asVec3ul() const {
  return boost::any_cast< virvo::vector< 3, unsigned long > >(value);
}

virvo::vector< 3, long long > vvParam::asVec3ll() const {
  return boost::any_cast< virvo::vector< 3, long long > >(value);
}

virvo::vector< 3, unsigned long long > vvParam::asVec3ull() const {
  return boost::any_cast< virvo::vector< 3, unsigned long long > >(value);
}

virvo::vector< 4, float > vvParam::asVec4f() const {
  return boost::any_cast< virvo::vector< 4, float > >(value);
}

vvColor vvParam::asColor() const {
  return boost::any_cast<vvColor>(value);
}

virvo::basic_aabb< float > vvParam::asAABBf() const {
  return boost::any_cast< virvo::basic_aabb< float > >(value);
}

virvo::basic_aabb< double > vvParam::asAABBd() const {
  return boost::any_cast< virvo::basic_aabb< double > >(value);
}

virvo::basic_aabb< int > vvParam::asAABBi() const {
  return boost::any_cast< virvo::basic_aabb< int > >(value);
}

virvo::basic_aabb< unsigned int > vvParam::asAABBui() const {
  return boost::any_cast< virvo::basic_aabb< unsigned int > >(value);
}

virvo::basic_aabb< long > vvParam::asAABBl() const {
  return boost::any_cast< virvo::basic_aabb< long > >(value);
}

virvo::basic_aabb< unsigned long > vvParam::asAABBul() const {
  return boost::any_cast< virvo::basic_aabb< unsigned long > >(value);
}

virvo::basic_aabb< long long > vvParam::asAABBll() const {
  return boost::any_cast< virvo::basic_aabb< long long > >(value);
}

virvo::basic_aabb< unsigned long long >  vvParam::asAABBull() const {
  return boost::any_cast< virvo::basic_aabb< unsigned long long > >(value);
}

boost::shared_ptr< vvClipObj > vvParam::asClipObj() const {
  return boost::any_cast< boost::shared_ptr< vvClipObj > >(value);
}

vvParam::operator bool() const {
  return asBool();
}

vvParam::operator char() const {
  return asChar();
}

vvParam::operator unsigned char() const {
  return asUchar();
}

vvParam::operator short() const {
  return asShort();
}

vvParam::operator unsigned short() const {
  return asUshort();
}

vvParam::operator int() const {
  return asInt();
}

vvParam::operator unsigned int() const {
  return asUint();
}

vvParam::operator long() const {
  return asLong();
}

vvParam::operator unsigned long() const {
  return asUlong();
}

#if VV_HAVE_LLONG
vvParam::operator long long() const {
  return asLlong();
}
#endif

#if VV_HAVE_ULLONG
vvParam::operator unsigned long long() const {
  return asUllong();
}
#endif
  
vvParam::operator float() const {
  return asFloat();
}

vvParam::operator virvo::vector< 2, float >() const {
  return asVec2f();
}

vvParam::operator virvo::vector< 2, int >() const {
  return asVec2i();
}

vvParam::operator virvo::vector< 3, float >() const {
  return asVec3f();
}

vvParam::operator virvo::vector< 3, double >() const {
  return asVec3d();
}

vvParam::operator virvo::vector< 3, short >() const {
  return asVec3s();
}

vvParam::operator virvo::vector< 3, unsigned short >() const {
  return asVec3us();
}

vvParam::operator virvo::vector< 3, int >() const {
  return asVec3i();
}

vvParam::operator virvo::vector< 3, unsigned int >() const {
  return asVec3ui();
}

vvParam::operator virvo::vector< 3, long >() const {
  return asVec3l();
}

vvParam::operator virvo::vector< 3, unsigned long >() const {
  return asVec3ul();
}

vvParam::operator virvo::vector< 3, long long >() const {
  return asVec3ll();
}

vvParam::operator virvo::vector< 3, unsigned long long >() const {
  return asVec3ull();
}

vvParam::operator virvo::vector< 4, float >() const {
  return asVec4f();
}

vvParam::operator vvColor() const {
  return asColor();
}

vvParam::operator virvo::basic_aabb< float >() const {
  return asAABBf();
}

vvParam::operator virvo::basic_aabb< double >() const {
  return asAABBd();
}

vvParam::operator virvo::basic_aabb< int >() const {
  return asAABBi();
}

vvParam::operator virvo::basic_aabb< unsigned int >() const {
  return asAABBui();
}

vvParam::operator virvo::basic_aabb< long >() const {
  return asAABBl();
}

vvParam::operator virvo::basic_aabb< unsigned long >() const {
  return asAABBul();
}

vvParam::operator virvo::basic_aabb< long long >() const {
  return asAABBll();
}

vvParam::operator virvo::basic_aabb< unsigned long long >() const {
  return asAABBull();
}

vvParam::operator boost::shared_ptr< vvClipObj >() const {
  return asClipObj();
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
