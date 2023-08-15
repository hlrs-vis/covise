/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <glm/glm.hpp>

namespace glm {

struct box2 {
  vec2 min, max;
};

static
void offaxisStereoCamera(vec3 LL, vec3 LR, vec3 UR, vec3 eye,
                         vec3 &dirOUT, vec3 &upOUT,
                         float &fovyOUT, float &aspectOUT,
                         box2 &imageRegionOUT)
{
  vec3 X = (LR-LL)/length(LR-LL);
  vec3 Y = (UR-LR)/length(UR-LR);
  vec3 Z = cross(X,Y);

  dirOUT = -Z;
  upOUT = Y;

  // eye position relative to screen/wall
  vec3 eyeP = eye-LL;

  // distance from eye to screen/wall
  float dist = dot(eyeP,Z);

  float left   = dot(eyeP,X);
  float right  = length(LR-LL)-left;
  float bottom = dot(eyeP,Y);
  float top    = length(UR-LR)-bottom;

  float newWidth = left<right ? 2*right : 2*left;
  float newHeight = bottom<top ? 2*top : 2*bottom;

  fovyOUT = 2*atan(newHeight/(2*dist));

  aspectOUT = newWidth/newHeight;

  imageRegionOUT.min.x = left<right ? (right-left)/newWidth : 0.f;
  imageRegionOUT.max.x = right<left ? (left+right)/newWidth : 1.f;
  imageRegionOUT.min.y = bottom<top ? (top-bottom)/newHeight: 0.f;
  imageRegionOUT.max.y = top<bottom ? (bottom+top)/newHeight : 1.f;
}

static
vec3 unprojectNDC(mat4 projInv, mat4 viewInv, vec3 ndc)
{
  vec4 v = viewInv * (projInv * vec4(ndc, 1.f));
  return vec3(v) / v.w;
}

static
bool intersectPlanePlane(vec3 na, vec3 pa, vec3 nb, vec3 pb,
                         vec3 &nl, vec3 &pl)
{
  vec3 nc = cross(na, nb);
  float det = length(nc)*length(nc);

  if (det == 0.f)
    return false;

  float da = -dot(na, pa);
  float db = -dot(nb, pb);

  pl = ((cross(nc, nb) * da) + (cross(na, nc) * db)) / det;
  nl = nc;
  return true;
}

static
bool solve(mat3 A, vec3 b, vec3 &x)
{
  float D = determinant(A);
  float D1 = determinant(mat3(b, A[1], A[2]));
  float D2 = determinant(mat3(A[0], b, A[2]));
  float D3 = determinant(mat3(A[0], A[1], b));

  if (D == 0.f)
    return false;

  x.x = D1/D;
  x.y = D2/D;
  x.z = D3/D;
  return true;
}

static
void closestLineSegmentBetweenTwoLines(vec3 na, vec3 pa, vec3 nb, vec3 pb,
                                       vec3 &pc1, vec3 &pc2)
{
  vec3 nc = normalize(cross(na, nb));
  vec3 b = pb-pa;
  mat3 A(na, -nb, nc);
  vec3 x;
  if (!solve(A, b, x))
    return;
  pc1 = pa+na*x.x;
  pc2 = pb+nb*x.y;
}

static
void offaxisStereoCameraFromTransform(mat4 projInv, mat4 viewInv,
                                      vec3 &eyeOUT, vec3 &dirOUT, vec3 &upOUT,
                                      float &fovyOUT, float &aspectOUT,
                                      box2 &imageRegionOUT)
{
  // Transform NDC unit cube corners to world/CAVE space
  vec3 v000 = unprojectNDC(projInv, viewInv, vec3(-1,-1,-1));
  vec3 v001 = unprojectNDC(projInv, viewInv, vec3(-1,-1, 1));

  vec3 v100 = unprojectNDC(projInv, viewInv, vec3( 1,-1,-1));
  vec3 v101 = unprojectNDC(projInv, viewInv, vec3( 1,-1, 1));

  vec3 v110 = unprojectNDC(projInv, viewInv, vec3( 1, 1,-1));
  vec3 v111 = unprojectNDC(projInv, viewInv, vec3( 1, 1, 1));

  vec3 v010 = unprojectNDC(projInv, viewInv, vec3(-1, 1,-1));
  vec3 v011 = unprojectNDC(projInv, viewInv, vec3(-1, 1, 1));

  // edges from +z to -z
  vec3 ez00 = normalize(v001-v000);
  vec3 ez10 = normalize(v101-v100);
  vec3 ez11 = normalize(v111-v110);
  vec3 ez01 = normalize(v011-v010);

  // edges from -y to +y
  vec3 ey00 = normalize(v010-v000);
  vec3 ey10 = normalize(v110-v100);
  vec3 ey11 = normalize(v111-v101);
  vec3 ey01 = normalize(v011-v001);

  // edges from -y to +y
  vec3 ex00 = normalize(v100-v000);
  vec3 ex10 = normalize(v110-v010);
  vec3 ex11 = normalize(v111-v011);
  vec3 ex01 = normalize(v101-v001);

  vec3 nL = normalize(cross(ey00, ez00));
  vec3 nR = normalize(cross(ez10, ey10));
  vec3 nB = normalize(cross(ez00, ex00));
  vec3 nT = normalize(cross(ex10, ez01));

  // Line of intersection between left/right planes
  vec3 pLR, nLR;
  bool isectLR = intersectPlanePlane(nL, v000, nR, v100, nLR, pLR);

  // Line of intersection between bottom/top planes
  vec3 pBT, nBT;
  bool isectBT = intersectPlanePlane(nB, v000, nT, v010, nBT, pBT);

  // Line segment connecting the two intersecint lines
  vec3 p1, p2;
  closestLineSegmentBetweenTwoLines(nLR, pLR, nBT, pBT, p1, p2);

  eyeOUT = (p1+p2)/2.f;

  vec3 LL = unprojectNDC(projInv, viewInv, vec3(-1,-1, 1));
  vec3 LR = unprojectNDC(projInv, viewInv, vec3( 1,-1, 1));
  vec3 UR = unprojectNDC(projInv, viewInv, vec3( 1, 1, 1));

  offaxisStereoCamera(LL, LR, UR, eyeOUT,
                      dirOUT, upOUT, fovyOUT, aspectOUT, imageRegionOUT);
}

} // namespace glm
