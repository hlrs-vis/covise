#ifndef BSPLINE_H_INCLUDED
#define BSPLINE_H_INCLUDED

struct Flist *BSplineKnot(struct Point *d, int deg);
void BSplinePoint(int deg, struct Point *d, struct Flist *t, float t0, float *x);
float deBoor(float *coord, struct Flist *t, int deg, float t0, int intvl, float **D);
void BSplineNormal(int deg, struct Point *d, struct Flist *t, float t0, float *grad);
void BSplineNormalXZ(int deg, struct Point *d, struct Flist *t, float t0, float *grad);
#endif                                            // BSPLINE_H_INCLUDED
