#ifndef  GA_COV_H_INCLUDED
#define  GA_COV_H_INCLUDED

#include <General/include/cov.h>

struct covise_info *Gate2Covise(struct gate *ga);
int CreateGA_BEPolygons(struct covise_info *ci, int be, int offset);
#ifdef TIP_POLYGONS
void CreateGA_TipPolygons(struct covise_info *ci, int np, int npopoin, int npol, int te);
#endif                                            // TIP_POLYGONS
void RotateGABlade4Covise(struct covise_info *ci, int nob);
void CreateGA_CoviseContours(struct covise_info *ci, struct gate *ga);
void CreateGAContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_vx, int sec, int np, int npb, int part);
#endif                                            // GA_COV_H_INCLUDED
