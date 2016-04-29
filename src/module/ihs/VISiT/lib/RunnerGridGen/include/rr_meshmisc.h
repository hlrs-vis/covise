#ifndef RR_MESHMISC_INCLUDE
#define RR_MESHMISC_INCLUDE

#include "../../General/include/curve.h"
#include "../../General/include/points.h"
#include "rr_grid.h"

#ifdef RADIAL_RUNNER
#include "../../RadialRunner/include/radial.h"
#endif
#ifdef DIAGONAL_RUNNER
#include "../../RadialRunner/include/diagonal.h"
#endif

struct Flist *GetCircleArclen(struct Point *line);
struct Flist *GetCircleArclenXY(struct Point *line);
struct region **GetRegionsMemory(struct region **reg, int rnum);
struct cgrid **SwitchCGridElements(struct cgrid **cge, int index1, int index2);
struct ge **SwitchBladeElements(struct ge **ge, int index1, int index2);
int GetPointRadius(struct Point *line, struct curve *ml);
int CalcLinearCurve(struct Point *line, struct Flist *para, float *u, float *p1);
int XShiftCurve(struct Point *srcline, struct Point *destline, float shift);
int CalcPointCoords(struct Point *line, struct Flist **arc, struct curve *ml);
int InterpolBlade(float *bppara, struct Point *bp, struct Flist *rpara,
struct Point *rp, float dphi);
int InterpolBladeSpline(struct Point *src, struct Flist *tgtpara,
struct Point *tgt, float *n, float dphi);
int   CalcEnvelopeCurve(struct Point *envline, struct Flist *learc, struct Point *leline,
struct Flist *blarc, struct Point *blline, struct Point *cl,
struct Flist *para, float dphi, float lscale, int sign);
int   CalcEnvelopeCurveSpline(struct Point *envline, struct Flist *learc, struct Point *leline,
struct Flist *blarc, struct Point *blline, struct Point *cl,
struct Flist *para, float dphi, float lscale, int sign);
int   CalcEnvelopeCurve2(struct Point *envline, struct Flist *learc, struct Point *leline,
struct Flist *blarc, struct Point *blline, struct Point *cl,
struct Flist *para, int le_dis, float dphi, float lscale, int sign);
int   CalcEnvelopeCurveSpline2(struct Point *envline, struct Flist *learc, struct Point *leline,
struct Flist *blarc, struct Point *blline, struct Point *cl,
struct Flist *para, int le_dis, float dphi, float lscale, int sign);
int CalcTERatio(struct region *reg, struct Flist *clarc,
struct Point *cl, int iblade, int ienv, float dphi);
float CalcTEParameter(struct region *reg, struct Flist *clarc,
struct Point *cl, int iblade, int ienv, float dphi);
#ifdef PARA_OUT
int PutRR_GridParams(struct rr_grid *grid);
#endif
#ifdef PARA_IN
int GetRR_GridParams(struct rr_grid *grid);
#endif
struct region *AddMeshLines(struct region *reg, int start, int end, int first,
int last, int initnuml, int addnuml);
#endif                                            // RR_MESHMISC_INCLUDE
