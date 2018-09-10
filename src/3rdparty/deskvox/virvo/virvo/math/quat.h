#ifndef VV_MATH_QUAT_H
#define VV_MATH_QUAT_H

#include <stddef.h>

namespace MATH_NAMESPACE
{

class quat
{
public:

    float w;
    float x;
    float y;
    float z;

    quat();
    quat(float w, float x, float y, float z);
    quat(float w, vec3 const& v);

    static quat identity();

};

} // MATH_NAMESPACE

#include "detail/quat.inl"

#endif // VV_MATH_QUAT_H


