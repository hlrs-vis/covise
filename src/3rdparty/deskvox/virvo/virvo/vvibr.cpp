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

#include "math/math.h"
#include "vvibr.h"


namespace virvo
{
namespace ibr
{

void calcDepthRange(mat4 const& pr, mat4 const& mv,
                    aabb const& bbox, float& minval, float& maxval)
{
    vec3 center = vec4( mv * vec4(bbox.center(), 1.0f) ).xyz();
    vec3 min    = vec4( mv * vec4(bbox.min, 1.0f) ).xyz();
    vec3 max    = vec4( mv * vec4(bbox.max, 1.0f) ).xyz();

    float radius = length(max - min) * 0.5f;

    // Depth buffer of ibrPlanes
    vec3 scal(center);
    scal = normalize(scal) * radius;
    min = center - scal;
    max = center + scal;

    vec4 min4 = pr * vec4(min, 1.f);
    vec4 max4 = pr * vec4(max, 1.f);

    min = min4.xyz() / min4.w;
    max = max4.xyz() / max4.w;

    minval = (min.z+1.f) * 0.5f;
    maxval = (max.z+1.f) * 0.5f;
}

mat4 calcImgMatrix(mat4 const& pr, mat4 const& mv, recti const& vp,
    float depthRangeMin, float depthRangeMax)
{
    return inverse(pr * mv)
        * calcViewportMatrix(vp)
        * calcDepthScaleMatrix(depthRangeMin, depthRangeMax);
}

mat4 calcViewportMatrix(recti const& vp)
{
    mat4 result = mat4::identity();
    result = translate( result,
        vec3((vp[0] / (0.5f * vp[2])) - 1.0f, (vp[1] / (0.5f * vp[3])) - 1.0f, -1.0f) );
    result = scale( result, vec3(1.0f / (0.5f * vp[2]), 1.0f / (0.5f * vp[3]), 2.0f) );
    return result;
}

mat4 calcDepthScaleMatrix(const float depthRangeMin, const float depthRangeMax)
{
    mat4 result = mat4::identity();
    result = translate( result, vec3(0.0f, 0.0f, depthRangeMin) );
    result = scale( result, vec3(1.0f, 1.0f, (depthRangeMax - depthRangeMin)) );
    return result;
}


} // ibr
} // virvo


