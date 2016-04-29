#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/curvepoly.h"
#include "include/points.h"
#include "include/log.h"
#include "include/v.h"

struct Point *CurvePolygon(float *p1, float *s, float *p2, float m1, float m2)
{
   float d1[3], d2[3];
   struct Point *poly = NULL;

   poly = AllocPointStruct();
   // partition point d1
   if (!p1 || !p2 || !s)
   {
      dprintf(0, "ERROR: p1=0x%x p2=0x%x s=0x%x\n", p1, p2, s);
      exit(1);
   }
   dprintf(10, "CurvePolygon(): p1[]=%f,%f,%f (0x%x)\n", p1[0], p1[1], p1[2], p1);
   dprintf(10, "CurvePolygon(): p2[]=%f,%f,%f (0x%x)\n", p2[0], p2[1], p2[2], p2);
   dprintf(10, "CurvePolygon(): s[]=%f,%f,%f (0x%x)\n", s[0], s[1], s[2], s);
   dprintf(10, "CurvePolygon(): m1=%f,m2=%f (0x%x,0x%x)\n", m1, m2, &m1, &m2);
   if (poly)
   {
      d1[0] = p1[0] + m1 * (s[0] - p1[0]);
      d1[1] = p1[1] + m1 * (s[1] - p1[1]);
      d1[2] = p1[2] + m1 * (s[2] - p1[2]);
      // partition point d2
      d2[0] = p2[0] + m2 * (s[0] - p2[0]);
      d2[1] = p2[1] + m2 * (s[1] - p2[1]);
      d2[2] = p2[2] + m2 * (s[2] - p2[2]);
      // assign points to curve polygon
      AddVPoint(poly, p1);
      AddVPoint(poly, d1);
      AddVPoint(poly, d2);
      AddVPoint(poly, p2);
   }
   else
   {
      dprintf(0, "CurvePolygon(): No more memory\n");
      exit(1);
   }

   return(poly);
}


struct Point *CurvePolygon2(float *p1, float *s, float *p2,
float m1, float m2, float m3)
{
   float d1[3], d2[3], d3[3], n[3];
   struct Point *poly = NULL;

   poly = AllocPointStruct();
   // partition point d1
   if (!p1 || !p2 || !s)
   {
      dprintf(0, "ERROR: p1=0x%x p2=0x%x s=0x%x\n", p1, p2, s);
      exit(1);
   }
   dprintf(10, "CurvePolygon(): p1[]=%f,%f,%f (0x%x)\n", p1[0], p1[1], p1[2], p1);
   dprintf(10, "CurvePolygon(): p2[]=%f,%f,%f (0x%x)\n", p2[0], p2[1], p2[2], p2);
   dprintf(10, "CurvePolygon(): s[]=%f,%f,%f (0x%x)\n", s[0], s[1], s[2], s);
   dprintf(10, "CurvePolygon(): m1=%f,m2=%f (0x%x,0x%x)\n", m1, m2, &m1, &m2);
   if (poly)
   {
      d1[0] = p1[0] + m1 * (s[0] - p1[0]);
      d1[1] = p1[1] + m1 * (s[1] - p1[1]);
      d1[2] = p1[2] + m1 * (s[2] - p1[2]);
      // partition point d2
      d2[0] = p2[0] + m2 * (s[0] - p2[0]);
      d2[1] = p2[1] + m2 * (s[1] - p2[1]);
      d2[2] = p2[2] + m2 * (s[2] - p2[2]);
      // move center point between d1 & d2 in xy-plane
      d3[0] =  0.5*(d2[0]+d1[0]);
      d3[1] =  0.5*(d2[1]+d1[1]);
      d3[2] =  0.5*(d2[2]+d1[2]);
      n[0]  = -(d2[1]-d1[1]);
      n[1]  =   d2[0]-d1[0];
      n[2]  =  0.0;
      d3[0]+=  m3*n[0];
      d3[1]+=  m3*n[1];
      // assign points to curve polygon
      AddVPoint(poly, p1);
      AddVPoint(poly, d1);
      AddVPoint(poly, d3);
      AddVPoint(poly, d2);
      AddVPoint(poly, p2);
   }
   else
   {
      dprintf(0, "CurvePolygon(): No more memory\n");
      exit(1);
   }

   return(poly);
}
