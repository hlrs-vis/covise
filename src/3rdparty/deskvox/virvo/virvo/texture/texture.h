#ifndef VV_TEXTURE_H
#define VV_TEXTURE_H


#include "detail/prefilter.h"
#include "detail/sampler1d.h"
#include "detail/sampler3d.h"
#include "detail/texture1d.h"
#include "detail/texture3d.h"



namespace virvo
{

//-------------------------------------------------------------------------------------------------
// tex1D - general case and specializations
//

template < typename VoxelT, typename FloatT >
VV_FORCE_INLINE VoxelT tex1D(texture< VoxelT, ElementType, 1 > const& tex, FloatT coord)
{

    // general case: return type equals voxel type
    typedef VoxelT return_type;

    return detail::tex1D< return_type >( tex, coord );

}


template < typename VoxelT >
VV_FORCE_INLINE vector< 4, simd::float4 > tex1D(texture< VoxelT, ElementType, 1 > const& tex, const simd::float4 &coord)
{

    // special case for AoS rgba colors
    typedef vector< 4, simd::float4 > return_type;

    return detail::tex1D< return_type >( tex, coord );

}


//-------------------------------------------------------------------------------------------------
// tex3D - general case and specializations
//

template < typename VoxelT, typename FloatT >
VV_FORCE_INLINE VoxelT tex3D(texture< VoxelT, ElementType, 3 > const& tex, const vector< 3, FloatT > &coord)
{

    // general case: return type equals voxel type
    typedef VoxelT return_type;

    return detail::tex3D< return_type >( tex, coord );

}


template < typename VoxelT >
VV_FORCE_INLINE simd::float4 tex3D(texture< VoxelT, ElementType, 3 > const& tex, const vector< 3, simd::float4 > &coord)
{

    // special case: lookup four voxels at once and return as 32-bit float vector
    typedef simd::float4 return_type;

    return detail::tex3D< return_type >( tex, coord );

}


} // virvo


#endif // VV_TEXTURE_H


