#ifndef VV_TEXTURE_FORWARD_H
#define VV_TEXTURE_FORWARD_H

#include <cstddef>

namespace virvo
{

//--------------------------------------------------------------------------------------------------
// Declarations
//

enum tex_address_mode
{
    Wrap = 0,
    Mirror,
    Clamp,
    Border
};


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol,
    CardinalSpline
};

enum tex_read_mode
{
    ElementType,
    NormalizedFloat
};


template
<
    typename VoxelT,
    tex_read_mode ReadMode,
    size_t Dim
>
class texture;

} // virvo


#endif // VV_TEXTURE_FORWARD_H


