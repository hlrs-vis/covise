#ifndef  T2C_INCLUDED
#define  T2C_INCLUDED

extern  struct covise_info *Tube2Covise(struct tube *tu);
extern  void T_AddPoints2CI(struct Point *points, struct cs *cs);
extern  struct Point * CS_BorderPoints(struct cs *cs);

#ifdef   DEBUG
#include <include/cov.h>

extern  void CiCsDump(FILE *fp, struct ci_cs *c);
extern  void Tube2CoviseDump(struct covise_info *ci);
#endif                                            // DEBUG
#endif                                            // T2C_INCLUDED
