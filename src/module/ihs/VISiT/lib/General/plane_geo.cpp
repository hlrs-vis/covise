#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/points.h"
#include "include/plane_geo.h"

#define RAD(x)          ((x)*M_PI/180.0)
#define GRAD(x)         ((x)*180.0/M_PI)

int LineIntersect(float *p1, float *v1, float *p2, float *v2, float *s)
{
   float t, r;

   s[0] = s[1] = s[2] = 0.0;
   // intersection parameter
   t  = (p2[0] - p1[0]) * v2[1] - (p2[1] - p1[1]) * v2[0];
   if( fabs(r = (v1[0] * v2[1] - v1[1] * v2[0])) < 1.0E-6)
   {
      s[0] = (p1[0] + p2[0])*0.5;
      s[1] = (p1[1] + p2[1])*0.5;
   }
   else
   {
      t /= r;
      // intersection point s
      s[0] = p1[0] + t * v1[0];
      s[1] = p1[1] + t * v1[1];
   }
   return(1);
}


int LineIntersectXZ(float *p1, float *v1, float *p2, float *v2, float *s)
{
   float t, r;

   s[0] = s[1] = s[2] = 0.0;
   // intersection parameter
   t  = (p2[0] - p1[0]) * v2[2] - (p2[2] - p1[2]) * v2[0];
   if( fabs(r = (v1[0] * v2[2] - v1[2] * v2[0])) < 1.0E-6)
   {
      s[0] = (p1[0] + p2[0])*0.5;
      s[1] = (p1[1] + p2[1])*0.5;
      s[2] = (p1[2] + p2[2])*0.5;
   }
   else
   {
      t /= r;
      // intersection point s
      s[0] = p1[0] + t * v1[0];
      s[1] = p1[1] + t * v1[1];
      s[2] = p1[2] + t * v1[2];
   }
   return(1);
}


struct Point *ArcSegmentsXZ(float *m, float *s, float *e, int n)
{
   int i;
   float a[3], b[3], mag_a, mag_b, angle, angle_seg;
   float p[3];
   struct Point *p_arc = NULL;

   p_arc = AllocPointStruct();
   // vectors arc centre to start/end point
   a[0] = s[0] - m[0];
   a[1] = 0.0;
   a[2] = s[2] - m[2];
   b[0] = e[0] - m[0];
   b[1] = 0.0;
   b[2] = e[2] - m[2];
   // enclosed angle, segment angle
   mag_a     = sqrt(pow(a[0], 2) + pow(a[2], 2));
   mag_b     = sqrt(pow(b[0], 2) + pow(b[2], 2));
   angle     = acos((a[0] * b[0] + a[2] * b[2]) / (mag_a * mag_b));
   angle_seg = angle / n;
   // segment arc points, assign to struct point
   //AddVPoint(p_arc, m);
   AddVPoint(p_arc, s);
   for (i = 0; i < (n-1); i++)
   {
      angle = (i + 1) * angle_seg;
      p[0]  = (s[0] - m[0]) * cos(angle) + (s[2] - m[2]) * sin(angle) + m[0];
      p[1]  = 0.0;
      p[2]  = (s[0] - m[0]) * -sin(angle) + (s[2] - m[2]) * cos(angle) + m[2];
      AddVPoint(p_arc, p);
   }
   AddVPoint(p_arc, e);

   return p_arc;
}
