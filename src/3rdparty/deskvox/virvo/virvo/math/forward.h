#ifndef VV_MATH_FORWARD_H
#define VV_MATH_FORWARD_H

#include "config.h"

#include <cstddef>


namespace MATH_NAMESPACE
{


//--------------------------------------------------------------------------------------------------
// Declarations
//

template < size_t Dim >
class cartesian_axis;

template < size_t Dim, typename T >
class vector;

template < size_t N /* rows */, size_t M /* columns */, typename T >
class matrix;

template < size_t Dim, typename T >
class basic_plane;

template < typename T >
class basic_aabb;

template < typename T >
class basic_ray;

template < size_t Dim, typename T, typename P = unsigned >
class basic_triangle;

template
<
    template < typename > class L,
    typename T
>
class rectangle;

template< typename T >
class xywh_layout;

namespace simd
{
union mask4;
class int4;
class float4;
}


//--------------------------------------------------------------------------------------------------
// Most common typedefs
//

typedef vector< 2, int >                    vec2i;
typedef vector< 2, unsigned int >           vec2ui;
typedef vector< 2, float >                  vec2f;
typedef vector< 2, double >                 vec2d;
typedef vector< 2, float >                  vec2;


typedef vector< 3, int >                    vec3i;
typedef vector< 3, unsigned int >           vec3ui;
typedef vector< 3, float >                  vec3f;
typedef vector< 3, double >                 vec3d;
typedef vector< 3, float >                  vec3;


typedef vector< 4, int >                    vec4i;
typedef vector< 4, unsigned int >           vec4ui;
typedef vector< 4, float >                  vec4f;
typedef vector< 4, double >                 vec4d;
typedef vector< 4, float >                  vec4;


typedef matrix< 4, 4, float >               mat4f;
typedef matrix< 4, 4, double >              mat4d;
typedef matrix< 4, 4, float >               mat4;


typedef basic_plane< 3, int >               plane3i;
typedef basic_plane< 3, float >             plane3f;
typedef basic_plane< 3, double >            plane3d;
typedef basic_plane< 3, float >             plane3;


typedef basic_aabb< int >                   aabbi;
typedef basic_aabb< float >                 aabbf;
typedef basic_aabb< double >                aabbd;
typedef basic_aabb< float >                 aabb;


typedef basic_ray< float >                  ray;
namespace simd
{
typedef basic_ray< float4 >                 ray4;
}


typedef rectangle< xywh_layout, int >       recti;
typedef rectangle< xywh_layout, float >     rectf;
typedef rectangle< xywh_layout, double >    rectd;
typedef rectangle< xywh_layout, float >     rect;


} // MATH_NAMESPACE


#endif // VV_MATH_FORWARD_H


