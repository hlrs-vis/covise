#ifndef RR_MESH_INCLUDE
#define RR_MESH_INCLUDE

#include "../../General/include/curve.h"
#include "../../General/include/points.h"
#include "rr_grid.h"

int CreateRR_GridRegions(int nob, struct rr_grid *grid);
#ifdef GAP
int AddGAP(struct be *gp, struct rr_grid *grid);
#endif
#endif                                            // RR_MESH_INCLUDE
