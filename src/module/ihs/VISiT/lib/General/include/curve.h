#ifndef  CURVE_H_INCLUDED
#define  CURVE_H_INCLUDED

#include <stdio.h>

struct curve
{
   struct Point *p;                               // curve point coordinates
   float *len;                                    // absolute curve arc length
   float *par;                                    // curve parameter, rel. arc length [0..1]
   int arclen;
};

extern  struct curve *AllocCurveStruct(void);
extern  void FreeCurveStruct(struct curve *c);
extern  int AddCurvePoint(struct curve *c, float x, float y, float z, float len, float par);
extern  int CalcCurveArclen(struct curve *c);
extern  int CalcCurveArclen2(struct curve *c);
extern  struct curve *GetCurveMemory(struct curve *c);
extern  void DumpCurve(struct curve *c, FILE *fp);
#endif                                            // CURVE_H_INCLUDED
