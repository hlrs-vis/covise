/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BOOTH_H_INCLUDED
#define BOOTH_H_INCLUDED

#include <vector>

#include "cov.h"
#include "elements.h"
#include "flist.h"
#define WALL 1
#define INLET 100
#define INLET_FLOOR_4HOLES 150
#define INLET_FLOOR_1HOLE 151
#define INLET_FLOOR_OPEN_SX9 152
#define INLET_FLOOR_LOCHBLECH 153
#define INLET_FLOOR_4HOLESOPEN 154
#define INLET_FLOOR_OPEN_NEC_CLUSTER 155
#define OUTLET 200
#define OUTLET_CEILING 250
#define VEN 4

#define MIN_ELEMY_FOR_AIRCONDITIONING 25
#define MIN_ELEM_FOR_VEN 25

#define RG_COL_NODE 2
#define RG_COL_ELEM 2
#define RG_COL_DIRICLET 2
#define RG_COL_WALL 7
#define RG_COL_BALANCE 7
#define RG_COL_PRESS 7

struct rech_model
{
    int nobjects; // number of objects
    float size[3]; // size of complete model
    float Q; // flow rate [m3/s]
    float v_sx9;
    float v_NEC_cluster;
    float Q_ven; // flow rate ventilators mounted in cube
    float spacing;
    const char *BCFile;

    struct covise_info *ci; // surface geometry

    // bcs
    int ilo; // z-direction
    int ihi;
    int jlo; // y-direction
    int jhi;
    int klo; // x-direction
    int khi;

    int bc_type_plusx;
    int bc_type_minusx;
    int bc_type_plusy;
    int bc_type_minusy;
    int bc_type_plusz;
    int bc_type_minusz;

    float bcin_velo[3];

    struct cubus **cubes;
    float zScale;

    // floor inlet variables
    float Q_total;
    float Q_SX9;
    float Q_NEC_cluster;

    int n_floor_4holes; // number of floor squares with 4 holes
    int n_floor_4holesopen; // number of floor squares with 4 completely open holes
    int n_floor_1hole; // number of floor squares with 1 hole
    int n_floor_open_sx9; // number of floor squares completely open under sx9
    int n_floor_open_nec_cluster; // number of floor squares completely open under nec cluster
    int n_floor_lochblech; // number of lochblech floor squares
    float Ain_floor_total; // total floor inlet area
    float Ain_floor_4holes;
    float Ain_floor_4holesopen;
    float Ain_floor_1hole;
    float Ain_floor_open_sx9;
    float Ain_floor_open_NEC_cluster;
    float Ain_floor_lochblech;
    float vin_4holes;
    float vin_4holesopen;
    float vin_1hole;
    float vin_open_sx9;
    float vin_lochblech;
    float vin_open_nec_cluster;
};

struct rechgrid
{
    struct Point *p;
    struct Element *e;

    int *e_to_remove;
    int *p_to_remove;
    int *new_node;
    int *new_elem;

    float spacing_x; // the distance between to neighbour nodes
    float spacing_y;
    float spacing_z;

    int nelem;
    int nelem_x;
    int nelem_y;
    int nelem_z;

    int npoi;
    int npoi_x;
    int npoi_y;
    int npoi_z;

    // bcs
    struct Ilist *bcwall; // vertex list for wall boundary surfaces
    struct Ilist *bcwallpol; // polygon list for wall boundary surfaces
    struct Ilist *bcwallvol; // the inner volume element on the wall boundary surfaces
    struct Ilist *bcwallvol_outer; // the outer volume element on the wall boundary surfaces

    // inlet floor squares
    int *floorbcs; // floor bc type: 0=wall,
    //                1=IN_FLOOR_4HOLES,
    //                2=IN_FLOOR_1HOLE,
    //                3=IN_FLOOR_OPEN_SX9,
    //                4=IN_FLOOR_LOCHBLECH,
    //                5=IN_FLOOR_4HOLESOPEN
    //                6=IN_FLOOR_OPEN_NEC_CLUSTER

    std::vector<int> bcinlet_4holes;
    std::vector<int> bcinlet_4holes_pol;
    std::vector<int> bcinlet_4holes_vol;
    std::vector<int> bcinlet_4holes_outer;

    std::vector<int> bcinlet_4holesopen;
    std::vector<int> bcinlet_4holesopen_pol;
    std::vector<int> bcinlet_4holesopen_vol;
    std::vector<int> bcinlet_4holesopen_outer;

    std::vector<int> bcinlet_1hole;
    std::vector<int> bcinlet_1hole_pol;
    std::vector<int> bcinlet_1hole_vol;
    std::vector<int> bcinlet_1hole_outer;

    std::vector<int> bcinlet_open_sx9;
    std::vector<int> bcinlet_open_sx9_pol;
    std::vector<int> bcinlet_open_sx9_vol;
    std::vector<int> bcinlet_open_sx9_outer;

    std::vector<int> bcinlet_lochblech;
    std::vector<int> bcinlet_lochblech_pol;
    std::vector<int> bcinlet_lochblech_vol;
    std::vector<int> bcinlet_lochblech_outer;

    std::vector<int> bcinlet_open_nec_cluster;
    std::vector<int> bcinlet_open_nec_cluster_pol;
    std::vector<int> bcinlet_open_nec_cluster_vol;
    std::vector<int> bcinlet_open_nec_cluster_outer;

    struct Ilist *bcin; // vertex list for inlet boundary surfaces
    struct Ilist *bcinpol; // polygon list for inlet boundary surfaces
    std::vector<int> bcinvol; // the inner volume element on the inlet boundary surfaces
    struct Ilist *bcinvol_outer; // the outer volume element on the inlet boundary surfaces
    std::vector<int> bcin_type; // 100-129: racks  150-153: inlet floor squares
    std::vector<float> bcin_type2; // 0: racks  1-5: inlet floor squares

    int *ceilingbcs; // ceiling bc type: 0=wall, 1-8=outlet clima nr.
    std::vector<int> bcoutlet_air;
    std::vector<int> bcoutlet_air_pol;
    std::vector<int> bcoutlet_air_vol;
    std::vector<int> bcoutlet_air_outer;

    struct Ilist *bcout; // vertex list for outlet boundary surfaces
    struct Ilist *bcoutpol; // polygon list for outlet boundary surfaces
    struct Ilist *bcoutvol; // the inner volume element on the outlet boundary surfaces
    struct Ilist *bcoutvol_outer; // the outer volume element on the outlet boundary surfaces
    std::vector<int> bcout_type; // 200-229: racks  250-257: ceiling outlets

    std::vector<int> bcin_nodes; // inlet nodes list
    std::vector<float> bcin_velos; // inlet nodes

    int bc_inval; // bila inlet
    int bc_outval; // bila outlet
};

struct cubus
{
    float pos[3];
    float size[3];

    int ilo;
    int ihi;
    int jlo;
    int jhi;
    int klo;
    int khi;

    int bc_type_plusx;
    int bc_type_minusx;
    int bc_type_plusy;
    int bc_type_minusy;
    int bc_type_plusz;
    int bc_type_minusz;
    int bc_special; // here we store if there is a scpecial bc like aircondition-slots ...

    int hasbcwall;
    int hasbcin;
    int hasbcout;

    // bcs
    float v_bcin;
    float v_bcout;
    float v_bcven;

    float Q;
    float inlet_area; // not used so far ...
};

int ReadStartfile(const char *fn, struct rech_model *model);
int DefineBCs(struct rechgrid *grid, struct rech_model *model, int number);
int GenerateFloorBCs(struct rechgrid *grid, struct rech_model *model);
int GenerateCeilingBCs(struct rechgrid *grid, struct rech_model *model);

int GenerateBCs(struct rechgrid *grid, struct rech_model *model, int number);

struct rech_model *AllocModel(void);
void FreeModel(struct rech_model *model);

struct rechgrid *CreateRechGrid(struct rech_model *model);
struct covise_info *CreateGeometry4Covise(struct rech_model *model);
int CreateCubePolygons4Covise(struct covise_info *ci, struct rech_model *model, int number, int ipoi, int ipol);
int subtractCube(struct rechgrid *grid, struct rech_model *model, int number);
int CreateGeoRbFile(struct rechgrid *grid, const char *geofile, const char *rbfile);

int read_string(char *buf, char **str, const char *separator);
int read_int(char *buf, int *izahl, const char *separator);
int read_double(char *buf, double *dzahl, const char *separator);

#endif
