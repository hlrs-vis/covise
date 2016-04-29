#ifndef RR_MESHINIT_INCLUDE
#define RR_MESHINIT_INCLUDE

#include "rr_grid.h"

int InitRR_Grid(struct rr_grid *grid);
int InitRR_GridParams(struct rr_grid *grid);
int TranslateBladeProfiles(struct rr_grid *grid);
#ifdef AXIAL_RUNNER
int InterpolMeridianPlanes(struct meridian **be, int be_num,
struct rr_grid *grid);

#elif RADIAL_RUNNER
int InterpolMeridianPlanes(struct be **be, int be_num, struct rr_grid *grid);
#endif
#ifdef GAP
int AddGAP(struct be *gp, struct rr_grid *grid);
#endif
#endif                                            // RR_MESHINIT_INCLUDE
