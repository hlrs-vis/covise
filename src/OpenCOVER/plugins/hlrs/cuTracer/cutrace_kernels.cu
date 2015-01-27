#include <cutil_math.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>

#include "utils.h"
#include "bb.h"

#define EPS 0.02

struct mat3 {
   float a11;
   float a12;
   float a13;

   float a21;
   float a22;
   float a23;

   float a31;
   float a32;
   float a33;
};

inline __device__ float3 rotz(const float &x, const float &y, const float &z,
                              const float &r) {

   float cosr = cos(r);
   float sinr = sin(r);

   return make_float3(x * cosr - y * sinr, x * sinr + y * cosr, z);
}

inline __device__ float3 rotz(const float3 &vec, const float &r) {
   
   return rotz(vec.x, vec.y, vec.z, r);
}


inline __device__ bool inside_bb(const float3 &pos, const BB &b) {

   return (pos.x >= b.minx && pos.x <= b.maxx &&
           pos.y >= b.miny && pos.y <= b.maxy &&
           pos.z >= b.minz && pos.z <= b.maxz);
}

__device__ __constant__ int neighbor[24] = {
   // div == 0
   LEFT, FRONT, UP,
   UP, BACK, RIGHT,
   RIGHT, FRONT, DOWN,
   BACK, LEFT, DOWN,
   // div == 1
   LEFT, BACK, UP,
   FRONT, RIGHT, UP,
   BACK, RIGHT, DOWN,
   LEFT, FRONT, DOWN,
};

__device__ __constant__ int tcl[40] = {
   // div == 0
   4, 5, 7, 0,
   6, 2, 5, 7,
   1, 0, 2, 5,
   3, 0, 2, 7,
   0, 5, 2, 7,
   // div == 1
   7, 6, 4, 3,
   5, 6, 4, 1,
   2, 1, 3, 6,
   0, 1, 3, 4,
   1, 6, 4, 3
};

__device__ int faces[6][4] = { { 0, 1, 4, 5 }, // front
                               { 1, 2, 5, 6 }, // right
                               { 2, 3, 6, 7 }, // back
                               { 0, 3, 4, 7 }, // left
                               { 0, 1, 2, 3 }, // down
                               { 4, 5, 6, 7 } }; // up


__device__ int cuda_inside_ht(const int *el, const int *cl, const float *x,
                              const float *y, const float *z, const float *vx,
                              const float *vy, const float *vz,
                              const float *bs, const int e, const float3 &pos,
                              float3 &vel, const int div) {

   //cuPrintf("(%f, %f, %f): %d\n", pos.x, pos.y, pos.z, e);
   float n[3];
   float a11, a21, a31, a12, a22, a32, a13, a23, a33;

   int direction = -1;
   float offender = FLT_MAX;

   for (int j = 0; j < 5; j++)
   {
      float3 p0 = make_float3(x[cl[el[e] + tcl[div * 20 + j * 4]]],
                              y[cl[el[e] + tcl[div * 20 + j * 4]]],
                              z[cl[el[e] + tcl[div * 20 + j * 4]]]);
      float3 p1 = make_float3(x[cl[el[e] + tcl[div * 20 + j * 4 + 1]]],
                              y[cl[el[e] + tcl[div * 20 + j * 4 + 1]]],
                              z[cl[el[e] + tcl[div * 20 + j * 4 + 1]]]);
      float3 p2 = make_float3(x[cl[el[e] + tcl[div * 20 + j * 4 + 2]]],
                              y[cl[el[e] + tcl[div * 20 + j * 4 + 2]]],
                              z[cl[el[e] + tcl[div * 20 + j * 4 + 2]]]);
      float3 p3 = make_float3(x[cl[el[e] + tcl[div * 20 + j * 4 + 3]]],
                              y[cl[el[e] + tcl[div * 20 + j * 4 + 3]]],
                              z[cl[el[e] + tcl[div * 20 + j * 4 + 3]]]);
      
      float V =
         (p1.x - p0.x) * ((p2.y - p0.y) * (p3.z - p0.z) -
                          (p2.z - p0.z) * (p3.y - p0.y)) +
         (p2.x - p0.x) * ((p0.y - p1.y) * (p3.z - p0.z) -
                          (p0.z - p1.z) * (p3.y - p0.y)) +
         (p3.x - p0.x) * ((p1.y - p0.y) * (p2.z - p0.z) -
                          (p1.z - p0.z) * (p2.y - p0.y));
         
      a11 = (p3.z - p0.z) * (p2.y - p3.y) - (p2.z - p3.z) * (p3.y - p0.y);
      a21 = (p3.z - p0.z) * (p0.y - p1.y) - (p0.z - p1.z) * (p3.y - p0.y);
      a31 = (p1.z - p2.z) * (p0.y - p1.y) - (p0.z - p1.z) * (p1.y - p2.y);
      
      a12 = (p3.x - p0.x) * (p2.z - p3.z) - (p2.x - p3.x) * (p3.z - p0.z);
      a22 = (p3.x - p0.x) * (p0.z - p1.z) - (p0.x - p1.x) * (p3.z - p0.z);
      a32 = (p1.x - p2.x) * (p0.z - p1.z) - (p0.x - p1.x) * (p1.z - p2.z);

      a13 = (p3.y - p0.y) * (p2.x - p3.x) - (p2.y - p3.y) * (p3.x - p0.x);
      a23 = (p3.y - p0.y) * (p0.x - p1.x) - (p0.y - p1.y) * (p3.x - p0.x);
      a33 = (p1.y - p2.y) * (p0.x - p1.x) - (p0.y - p1.y) * (p1.x - p2.x);

      float3 vec = pos - p0;
      n[0] = (vec.x * a11 + vec.y * a12 + vec.z * a13) / V;
      n[1] = (vec.x * a21 + vec.y * a22 + vec.z * a23) / V;
      n[2] = (vec.x * a31 + vec.y * a32 + vec.z * a33) / V;
      
      //cuPrintf("    %d: %f %f %f\n", j, n[0], n[1], n[2]);

      if (n[0] >= -EPS && n[1] >= -EPS && n[2] >= -EPS &&
          1 - n[0] - n[1] - n[2] >= -EPS) {
         float3 v0 = make_float3(vx[cl[el[e] + tcl[div * 20 + j * 4 + 0]]],
                                 vy[cl[el[e] + tcl[div * 20 + j * 4 + 0]]],
                                 vz[cl[el[e] + tcl[div * 20 + j * 4 + 0]]]);
         float3 v1 = make_float3(vx[cl[el[e] + tcl[div * 20 + j * 4 + 1]]],
                                 vy[cl[el[e] + tcl[div * 20 + j * 4 + 1]]],
                                 vz[cl[el[e] + tcl[div * 20 + j * 4 + 1]]]);
         float3 v2 = make_float3(vx[cl[el[e] + tcl[div * 20 + j * 4 + 2]]],
                                 vy[cl[el[e] + tcl[div * 20 + j * 4 + 2]]],
                                 vz[cl[el[e] + tcl[div * 20 + j * 4 + 2]]]);
         float3 v3 = make_float3(vx[cl[el[e] + tcl[div * 20 + j * 4 + 3]]],
                                 vy[cl[el[e] + tcl[div * 20 + j * 4 + 3]]],
                                 vz[cl[el[e] + tcl[div * 20 + j * 4 + 3]]]);

         vel = v0 + n[0] * (v1 - v0) + n[1] * (v2 - v0) + n[2] * (v3 - v0);
         //printf("   +(%f, %f, %f) found inside element %d tet %d\n", px, py, pz, element, j);
         return -1;
      }
      if (j != 4 && (n[0] <= 0 || n[1] <= 0 || n[2] <= 0)) {
         for (int index = 0; index < 3; index ++)
            if (n[index] <= 0 && n[index] < offender) {
               offender = n[index];
               direction = neighbor[div * 12 + j * 3 + index];
            }
      }
   }
   //cuPrintf("  dir: %d\n", direction);
   if (direction != -1)
      return direction;      

   //cuPrintf("    %d (%f %f %f) COULD NOT BE FOUND\n", e, pos.x, pos.y, pos.z);
   return -2;
}


const float k12 = 0.293328;
const float V1 = 132.72;
const float V2 = 352.99;
const float V3 = 179.29;
const float gamma0 = M_PI / 180.0 * 30;

__device__ inline float k2(const float &gamma) {

   return (gamma <= 2.0 * gamma0)?1.0:0.0;
}

__device__ inline float f(const float &gamma) {

   return pow(1 + k2(gamma) * k12 * sin(gamma * M_PI / 2.0 / gamma0), 2);
}

__device__ inline float rt(const float &vp, const float &gamma) {

   return 1 - vp / V3 * sin(gamma);
}

__device__ inline float fvpn(const float &vp, const float &gamma) {

   return pow(vp / V2 * sin(gamma), 4);
}

__device__ float cu_E(const float &vp, const float &gamma) {

   return f(gamma) * pow(vp / V1, 2) * pow(cos(gamma), 2) * (1 - pow(rt(vp, gamma), 2)) + fvpn(vp, gamma);
}

__device__ char cuda_find_ht_massy(const int *el, const int *cl,
                             const float *x, const float *y, const float *z,
                             const float *vx, const float *vy, const float *vz,
                             const float *bs, const int *neighbors,
                             const unsigned char *faceType, const BB *flat,
                             const int *cells, int &element, const float3 &pos,
                             float3 &v, int &side, int &partition,
                             const float periodicRotation, int &div) {

   float a = M_PI / 180.0 * periodicRotation * partition;
   float3 ppos = rotz(pos, -a);

   int list[64];
   int lindex = 0;
   div = 0;

   if (element == -2)
      return 0;

   //cuPrintf("(%f, %f, %f)\n", ppos.x, ppos.y, ppos.z);

   // look inside the last cell
   if (element >= 0) {

      side = cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs,
                            element, ppos, v, 0);
      if (side == -1) {
         //cuPrintf(" element %d: inside\n", element);
         v = rotz(v, a);
         return 1;
      }

      if (side == -2)
         return -2;
      
      int neighbor = neighbors[element * 6 + side];
      if (neighbor == -1 && (faceType[element] & (WALL << 6))) {
         //cuPrintf("  element %d: no neighbor & wall\n", element);
         div = (1 - div);
         return 2;
      }
      
      for (int index = 0; index < 5; index ++) {
         if (neighbor != -1) {

            int npart = neighbor >> 29;
            if (npart != 0) {
               if (npart == 2)
                  npart = -1;

               partition += npart;
               if (partition < 0)
                  partition += 11;
               partition %= 11;

               a = M_PI / 180.0 * periodicRotation * partition;
               ppos = rotz(pos, -a);
            }

            neighbor &= ((1 << 29) - 1);
            //cuPrintf("    element %d neighbor: %d\n", element, neighbor);
            int nside = cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbor, ppos, v, 0);
            
            if (nside == -1) {
               element = neighbor;
               //cuPrintf(" element %d: inside\n", element);
               v = rotz(v, a);
               return 1;
            }
            side = nside;
            element = neighbor;
            neighbor = neighbors[element * 6 + side];
         } else {
            //cuPrintf("  element %d: no more neighbors\n", element);
         }
      }
   }
   //cuPrintf("  KD: element %d\n", element);
   list[0] = 0;
   
   while (lindex >= 0) {
      
      int idx = list[lindex];
      lindex--;
      BB current = flat[idx];
      
      if (inside_bb(ppos, current)) {
         int left = current.left;
         int right = current.right;
         if (left != -1) {
            lindex++;
            list[lindex] = left;
         }
         if (right != -1) {
            lindex ++;
            list[lindex] = right;
         }
         
         if (left == -1 && right == -1) {
            int num = cells[current.cells];
            for (int i = 0; i < num; i ++) {
               int cell = cells[current.cells + i + 1];
               if (cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, cell, ppos, v, div) == -1) {
                  v = rotz(v, a);
                  element = cell;
                  return 1;
               }
            }
         }
      }
   }

   element = -2;
   return 0;
}

__device__ bool cuda_find_ht(const int *el, const int *cl, const float *x,
                             const float *y, const float *z, const float *vx,
                             const float *vy, const float *vz, const float *bs,
                             const int *neighbors, const BB *flat,
                             const int *cells, int &element, const float3 &pos,
                             float3 &v, int &partition,
                             const float periodicRotation, int &div) {

   float a = M_PI / 180.0 * periodicRotation * partition;
   float3 ppos = rotz(pos, -a);

   int list[64];
   int lindex = 0;

   if (element == -2)
      return false;

   // look inside the last cell
   if (element >= 0) {

      int side = cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, element, ppos, v, div);
      if (side == -1) {
         v = rotz(v, a);
         return true;
      }

      int neighbor = neighbors[element * 6 + side];
      for (int index = 0; index < 5; index ++) {

         if (neighbor != -1) {

            int npart = neighbor >> 29;
            if (npart != 0) {
               if (npart == 2)
                  npart = -1;
               partition += npart;
               a = M_PI / 180.0 * periodicRotation * partition;
               ppos = rotz(pos, -a);
            }

            neighbor &= ((1 << 29) - 1);

            int nside = cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbor, ppos, v, div);
            
            if (nside == -1) {
               element = neighbor;
               v = rotz(v, a);
               return true;
            }
            side = nside;
            element = neighbor;
            neighbor = neighbors[element * 6 + side];
         }
      }
   }

   list[0] = 0;
   
   while (lindex >= 0) {
      
      int idx = list[lindex];
      lindex--;
      BB current = flat[idx];
      
      if (inside_bb(ppos, current)) {
         int left = current.left;
         int right = current.right;
         if (left != -1) {
            lindex++;
            list[lindex] = left;
         }
         if (right != -1) {
            lindex ++;
            list[lindex] = right;
         }
         
         if (left == -1 && right == -1) {
            int num = cells[current.cells];
            for (int i = 0; i < num; i ++) {
               int cell = cells[current.cells + i + 1];
               if (cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, cell, ppos, v, div) == -1) {
                  v = rotz(v, a);
                  element = cell;
                  return true;
               }
            }
         }
      }
   }

   element = -2;
   return false;
}

__global__ void initial_cell(const int *el, const int *cl, const float *x, const float *y, const float *z,
                             const float *vx, const float *vy, const float *vz, const float *bs,
                             const BB *flat, const int *cells, const float3 *pos, int *outCell, float3 *outPos,
                             unsigned char *outVel, int num, int ts) {

   int div = 0;
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

   if (elemIdx < num) {
      float3 vel;
      float3 cur = pos[elemIdx];
      outPos[elemIdx] = cur;

      int list[64];
      int lindex = 0;
      
      list[0] = 0;

      while (lindex >= 0) {
         
         int idx = list[lindex];
         lindex--;
         BB current = flat[idx];

         if (inside_bb(cur, current)) {
            int left = current.left;
            int right = current.right;
            if (left != -1) {
               lindex++;
               list[lindex] = left;
            }
            if (right != -1) {
               lindex ++;
               list[lindex] = right;
            }
            
            if (left == -1 && right == -1) {
               int num = cells[current.cells];
               for (int i = 0; i < num; i ++) {
                  int cell = cells[current.cells + i + 1];
                  if (cuda_inside_ht(el, cl, x, y, z, vx, vy, vz, bs, cell, cur, vel, div) == -1) {
                     outCell[elemIdx] = cell;
                     outVel[elemIdx * 3] = 255;
                     outVel[elemIdx * 3 + 1] = 255;
                     outVel[elemIdx * 3 + 2] = 255;
                     return;
                  }
               }
            }
         }
      }
      outCell[elemIdx] = -2;
   }
}


__global__ void tr(const int *el, const int *cl, const float *x,
                   const float *y, const float *z, const float *vx,
                   const float *vy, const float *vz, const float *bs,
                   const int *neighbors, const BB *flat, const int *cells,
                   float3 *p, const float periodic, int *outCell,
                   float3 *outPos, int *outPart, unsigned char *outVel,
                   int numParticles, int steps) {

   int div = 0;
   float dt = 1.0 / 4.0;

   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   int partition = 0;
   if (elemIdx < numParticles) {
      int element = outCell[elemIdx];
      float3 pos = outPos[elemIdx];
      bool b0 = false, b1 = false, b2 = false, b3 = false;
      float3 p0, p1, p2, p3, v0, v1, v2, v3;
      int index;
      for (index = 1; index < steps; index ++) {

         b0 = cuda_find_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbors, flat, cells, element, pos, v0, partition, periodic, div);
         p0 = pos;
         if (b0) {
            p1 = p0 + (v0 * dt) / 2;
            b1 = cuda_find_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbors, flat, cells, element, p1, v1, partition, periodic, div);
            if (b1) {
               p2 = p0 + (v1 * dt) / 2;
               b2 = cuda_find_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbors, flat, cells, element, p2, v2, partition, periodic, div);
               if (b2) {
                  p3 = p0 + (v2 * dt);
                  b3 = cuda_find_ht(el, cl, x, y, z, vx, vy, vz, bs, neighbors, flat, cells, element, p3, v3, partition, periodic, div);
                  if (b3) {
                     pos = p0 + (v0 + v1 * 2 + v2 * 2 + v3) / 6.0 * dt;
                     outPos[elemIdx + index * numParticles] = pos;

                     float t = sqrt(v0.x * v0.x + v0.y * v0.y + v0.z * v0.z);
                     float3 a = make_float3(0, 0, 256);
                     float3 b = make_float3(256, 0, 0);
                     float3 s = lerp(a, b, t);
                     
                     outVel[(elemIdx + numParticles * index) * 3] =
                        clamp((int) s.x, 0, 255);
                     outVel[(elemIdx + numParticles * index) * 3 + 1] =
                        clamp((int) s.y, 0, 255);
                     outVel[(elemIdx + numParticles * index) * 3 + 2] =
                        clamp((int) s.z, 0, 255);
                  }
               }
            }
         }
         if (!(b0 && b1 && b2 && b3)) {
            element = outCell[elemIdx];
            pos = outPos[elemIdx];
            //break;
         }
      }
      outCell[elemIdx] = index * 4;
   }
}


__device__ float cu_re(const float3 &v, const float d) {

   return sqrt(v.x * v.x + v.y * v.y + v.z * v.z) * d / 1e-06;
}

__device__ void cu_dUp(float3 v0, float3 vp, float3 vold, float3 &dup,
                    float3 &duf, float dp, float rhop, float rhof,
                    float g, float dt) {

   duf = v0 - vold;
   float mp = M_PI / 6.0 * dp * dp * dp * rhop;
   
   float3 Fb = M_PI / 6 * dp * dp * dp * (1 - rhof / rhop) *
      make_float3(0.0, 0.0, g);
   float reynolds = cu_re(v0, 0.1);
   float Cd =
      fmaxf(24.0f / reynolds * (1 + 0.15 * pow(reynolds, 0.687f)), 0.44);
   float Af = M_PI * ((dp / 2) * (dp / 2));
   float3 Us = v0 - vp;
   float3 Fd = 1.0 / 2.0 * Cd * rhof * Af *
      sqrt(Us.x * Us.x + Us.y * Us.y + Us.z * Us.z) * Us;
   
   float3 Fvm = 1.0 / 2.0 * M_PI / 6.0 * dp * dp * dp * rhof *
      (duf / dt - dup / dt);
   
   float3 F = Fb + Fd + Fvm;
   dup = F * dt / mp;
}

__global__ void tr_massy(const int *el, const int *cl, const float *x,
                         const float *y, const float *z, const float *vx,
                         const float *vy, const float *vz, const float *bs,
                         const int *neighbors, unsigned char *faceType,
                         const BB *flat, const int *cells, float3 *p,
                         const float periodic, int *outCell, float3 *outPos,
                         int *outPart, unsigned char *outVel, int *outEle,
                         int numParticles, int start, int steps) {

   const float dp = 0.001;   // particle diameter in m
   const float rhof = 0.997; // water density
   const float rhop = 2.650; // particle density: quartz
   const float g = -9.81;    // gravity m/s²

   int div = 0;
   float dt = 1.0 / 10000.0;

   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x + start;

   int partition = 0;
   if (elemIdx < numParticles) {

      int element = outCell[elemIdx];
      float3 pos = outPos[elemIdx];
      char b0;
      float3 v0, vp, vold;
      int index;
      float3 dup = make_float3(0.0, 0.0, 0.0);
      float3 duf = make_float3(0.0, 0.0, 0.0);
      int side = 0;

      b0 = cuda_find_ht_massy(el, cl, x, y, z, vx, vy, vz, bs, neighbors,
                              faceType, flat, cells, element, pos, v0,
                              side, partition, periodic, div);
      outEle[elemIdx] = element;
      vp = v0;

      for (index = 1; index < steps; index ++) {

         vold = v0;
         b0 = cuda_find_ht_massy(el, cl, x, y, z, vx, vy, vz, bs,
                                 neighbors, faceType, flat, cells, element,
                                 pos, v0, side, partition, periodic, div);
         if (b0 == 1) {
            outEle[elemIdx + index * numParticles] = element;
            outVel[(elemIdx + index * numParticles) * 3] = 0;
            outVel[(elemIdx + index * numParticles) * 3 + 1] = 255;
            outVel[(elemIdx + index * numParticles) * 3 + 2] = 0;

            cu_dUp(v0, vp, vold, dup, duf, dp, rhop, rhof, g, dt);
            vp += dup;
            pos += vp * dt;
            outPos[elemIdx + index * numParticles] = pos;

         } else if (b0 == 2) {

            int v[4];
            
            for (int i = 0; i < 4; i ++)
               v[i] = cl[el[element] + faces[side][i]];
            
            float a = M_PI / 180.0 * periodic * partition;
            float3 hp0 = rotz(x[v[0]], y[v[0]], z[v[0]], a);
            float3 hp1 = rotz(x[v[1]], y[v[1]], z[v[1]], a);
            float3 hp2 = rotz(x[v[2]], y[v[2]], z[v[2]], a);
            float3 hp3 = rotz(x[v[3]], y[v[3]], z[v[3]], a);
            
            float3 n = normalize(cross(hp2 - hp0, hp1 - hp0));
            // intersection point
            float d = dot(hp0 - pos, n) / dot(normalize(vp), n);
            float3 ip = pos + d * normalize(vp);
            
            float angle = atan2(length(cross(vp, n)), dot(vp, n));
            if (angle > M_PI / 2)
               angle = M_PI - angle;
            angle = M_PI / 2 - angle;
            
            double e = cu_E(length(vp), angle);
            vp = reflect(vp, n);
            pos = ip + vp * dt;
            outPos[elemIdx + index * numParticles] = pos;
            outVel[(elemIdx + index * numParticles) * 3] = 255;
            outVel[(elemIdx + index * numParticles) * 3 + 1] = 0;
            outVel[(elemIdx + index * numParticles) * 3 + 2] = 0;
            outEle[elemIdx + index * numParticles] = -1;

         }
         if (!b0) {
            /*
            outPos[elemIdx + index * numParticles] = pos + make_float3(0.0, 0.0, +0.1);;
            outVel[(elemIdx + index * numParticles) * 3] = 0;
            outVel[(elemIdx + index * numParticles) * 3 + 1] = 0;
            outVel[(elemIdx + index * numParticles) * 3 + 2] = 255;
            outEle[elemIdx + index * numParticles] = -2;
            */
            break;
         }
      }
      outCell[elemIdx] = index;
   }
}


void
CheckErr(const char *where)
{
   cudaError_t cerror = cudaGetLastError();
   if (cerror != cudaSuccess)
      fprintf(stderr, "CudaError: %s (%s)\n", cudaGetErrorString(cerror),
              where);
}

#define BLOCK_X 64


void trace(const struct usg &usg, float3 *pos,
           const float periodic, int *outCell, float3 *outPos, int *outPart,
           unsigned char *outVel, int numParticles, int steps) {

   dim3 block(BLOCK_X);
   //dim3 grid((numParticles / block.x - 1) / block.x, 1);
   dim3 grid(numParticles / (block.x - 1), 1);

   timeval t0, t1;
   gettimeofday(&t0, NULL);

   tr<<<grid, block>>>(usg.elementList, usg.cornerList, usg.x, usg.y, usg.z,
                       usg.vx, usg.vy, usg.vz,
                       usg.boundingSpheres, usg.neighbors, usg.flat, usg.cells,
                       pos, periodic, outCell, outPos,
                       outPart, outVel, numParticles, steps);
   cudaThreadSynchronize();

   gettimeofday(&t1, NULL);

   int numIntegrations = 0;
   int num[numParticles];
   memset(num, 0, numParticles * sizeof(int));

   CUDA_SAFE_CALL(cudaMemcpy((void *) num, outCell, numParticles * sizeof(int),
                             cudaMemcpyDeviceToHost));   

   for (int index = 0; index < numParticles; index ++)
      numIntegrations += num[index];
   
   long long usec = ((long long) (t1.tv_sec - t0.tv_sec)) * 1000000 + (t1.tv_usec - t0.tv_usec);
   fprintf(stderr, "   gpu massless time: %lld usec (%f particles/s) (%f integrations/s) %d\n", usec,
           numParticles * (1000000.0 / usec),
           (1000000.0 / usec) * numIntegrations,
           numIntegrations);

   //cudaPrintfDisplay(stdout, true);
   //cudaPrintfEnd();

   CheckErr("kernel launch trace");
}



void trace_massy(const struct usg &u, const struct usg &usg, float3 *pos,
           const float periodic, int *outCell, float3 *outPos, int *outPart,
           unsigned char *outVel, int numParticles, int steps) {
   

   int iter = 128;
   int offset = numParticles / iter;

   dim3 block(BLOCK_X);
   dim3 grid((numParticles / iter + block.x - 1) / block.x, 1);

   int *outEle;
   CUDA_SAFE_CALL(cudaMalloc((void **) &outEle,
                             numParticles * steps * sizeof(int)));

   timeval t0, t1;
   gettimeofday(&t0, NULL);
   
   for (int index = 0; index < iter; index ++) {
      tr_massy<<<grid, block>>>(usg.elementList, usg.cornerList, usg.x, usg.y, usg.z, usg.vx, usg.vy, usg.vz,
                                usg.boundingSpheres, usg.neighbors, usg.faceType, usg.flat, usg.cells,
                                pos, periodic, outCell, outPos,
                                outPart, outVel, outEle, numParticles, index * offset, steps);
      //cudaThreadSynchronize();
   }

   cudaThreadSynchronize();
   gettimeofday(&t1, NULL);

   int numIntegrations = 0;
   int num[numParticles];
   CUDA_SAFE_CALL(cudaMemcpy((void *) num, outCell, numParticles * sizeof(int),
                             cudaMemcpyDeviceToHost));   

   for (int index = 0; index < numParticles; index ++)
      numIntegrations += num[index];
   
   long long usec = ((long long) (t1.tv_sec - t0.tv_sec)) * 1000000 + (t1.tv_usec - t0.tv_usec);
   fprintf(stderr, "   gpu massy time: %lld usec (%f particles/s) (%f integrations/s) %d\n", usec,
           numParticles * (1000000.0 / usec),
           (1000000.0 / usec) * numIntegrations,
           numIntegrations);
   /*
   float3 p[numParticles * steps];
   int e[numParticles * steps];
   CUDA_SAFE_CALL(cudaMemcpy((void *) p, outPos,
                             numParticles * steps * sizeof(float3),
                             cudaMemcpyDeviceToHost));      

   CUDA_SAFE_CALL(cudaMemcpy((void *) e, outEle,
                             numParticles * steps * sizeof(int),
                             cudaMemcpyDeviceToHost));      

   for (int index = 0; index < 42; index ++) {

   //printPoint(p[40 + numParticles * index].x,
   //p[40 + numParticles * index].y,
   //p[40 + numParticles * index].z);
   //printElement(&u, e[40 + numParticles * index], 0, 0);
      printf("(%f, %f, %f): %d\n", p[40 + numParticles * index].x,
             p[40 + numParticles * index].y,
             p[40 + numParticles * index].z, e[40 + numParticles * index]);
   }             
*/
   //cudaPrintfDisplay(stdout, true);
   //cudaPrintfEnd();

   CheckErr("kernel launch trace");
}

void init_trace(const struct usg &usg, float3 *pos, int *outCell, float3 *outPos, unsigned char *outVel, int num, int ts) {

   dim3 block(BLOCK_X);
   dim3 grid((num + block.x - 1) / block.x, 1);

   initial_cell<<<grid, block>>>(usg.elementList, usg.cornerList, usg.x, usg.y, usg.z, usg.vx, usg.vy, usg.vz, usg.boundingSpheres, usg.flat, usg.cells, pos, outCell, outPos, outVel,
                                 num, ts);

   CheckErr("kernel launch init_trace");
}
