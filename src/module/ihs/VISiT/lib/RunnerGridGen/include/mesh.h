#ifndef MESH_INCLUDE
#define MESH_INCLUDE

#if AXIAL_RUNNER
#include "../../AxialRunner/include/axial.h"
#include "rr_grid.h"
struct rr_grid *CreateAR_Mesh(struct axial *);

#elif RADIAL_RUNNER
#include "../../RadialRunner/include/radial.h"
#include "rr_grid.h"
struct rr_grid *CreateRR_Mesh(struct radial *);

#elif DIAGONAL_RUNNER
#include "../../RadialRunner/include/diagonal.h"
#include "rr_grid.h"
struct rr_grid *CreateRR_Mesh(struct radial *);
#endif

const char *GetLastGridErr();
#endif                                            // MESH_INCLUDE
