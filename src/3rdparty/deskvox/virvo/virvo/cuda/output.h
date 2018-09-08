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

#ifndef _VV_CUDAOUTPUT_H_
#define _VV_CUDAOUTPUT_H_


#include <cuda_runtime.h>
 
#include "../vvexport.h"

#include <ostream>

inline std::ostream& operator<<(std::ostream& out, const uchar2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const uchar4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const int4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float2& v)
{
  out << v.x << " " << v.y;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float3& v)
{
  out << v.x << " " << v.y << " " << v.z;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const float4& v)
{
  out << v.x << " " << v.y << " " << v.z << " " << v.w;
  return out;
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
