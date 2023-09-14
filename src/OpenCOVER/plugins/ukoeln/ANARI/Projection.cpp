/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <math.h>
#include "Projection.h"

namespace glm {

void transformDepthFromWorldToGL(const float *world, float *gl,
                                 vec3 eye, vec3 dir, vec3 up,
                                 float fovy, float aspect, box2 imageRegion,
                                 mat4 view, mat4 proj,
                                 int width, int height)
{
  vec2 imgPlaneSize;
  imgPlaneSize.y = 2.f * tanf(0.5f * fovy);
  imgPlaneSize.x = imgPlaneSize.y * aspect;

  vec3 dir_du = normalize(cross(dir, up)) * imgPlaneSize.x;
  vec3 dir_dv = normalize(cross(dir_du, dir)) * imgPlaneSize.y;
  vec3 dir_00 = dir - .5f * dir_du - .5f * dir_dv;

  for (int y=0; y<height; ++y) {
    for (int x=0; x<width; ++x) {
      vec2 screen(x/(float)width, y/(float)height);
      screen.x = mix(imageRegion.min.x, imageRegion.max.x, screen.x);
      screen.y = mix(imageRegion.min.y, imageRegion.max.y, screen.y);

      vec3 ray_org = eye;
      vec3 ray_dir = normalize(dir_00 + screen.x * dir_du + screen.y * dir_dv);

      size_t index = x+size_t(width)*y;

      float t = world[index];
      if (!isfinite(t)) {
        gl[index] = 1.f;
      } else {
        vec3 worldPos = ray_org + ray_dir * t;

        vec4 P4 = proj * (view * vec4(worldPos,1.f));
        vec3 glPos = vec3(P4)/P4.w;

        gl[index] = clamp((glPos.z + 1.f) * .5f, 0.f, 1.f);
      }
    }
  }
}

} // namespace glm
