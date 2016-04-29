#ifndef AR_INTERSECT_H_INCLUDED
#define AR_INTERSECT_H_INCLUDED

#include <AxialRunner/include/axial.h>

extern  int BladeContourIntersect(struct axial *ar);
extern  int CalcExtension(struct margin *ma, struct axial *ar, int side);
extern  int CalcIntersection(struct margin *ma, struct curve *c, struct be *be, struct model *mod);
#endif                                            // AXIAL_H_INCLUDED
