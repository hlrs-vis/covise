#ifndef NEW_MESH_INCLUDE
#define NEW_MESH_INCLUDE

#include "../../General/include/curve.h"
#include "../../General/include/points.h"
#include "rr_grid.h"

int CreateNew_GridRegions(int nob, struct rr_grid *grid);
#ifdef GAP
int AddGAP(struct be *gp, struct rr_grid *grid);
#endif
#endif                                            // NEW_MESH_INCLUDE
