#ifndef PLANE_GEO_H_INCLUDED
#define PLANE_GEO_H_INCLUDED

extern  int LineIntersect(float *p1, float *v1, float *p2, float *v2, float *s);
extern  int LineIntersectXZ(float *p1, float *v1, float *p2, float *v2, float *s);
extern  float Distance(float *p1, float *p2);
extern  struct Point *ArcSegmentsXZ(float *m, float *s, float *e, int n);
#endif                                            // PLANE_GEO_H_INCLUDED
