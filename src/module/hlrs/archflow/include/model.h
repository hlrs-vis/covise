/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BOOTH_H_INCLUDED
#define BOOTH_H_INCLUDED

#include "cov.h"
#include "elements.h"
#include "flist.h"
#define WALL 1
#define INLET 2
#define OUTLET 3
#define VEN 4

#define MIN_ELEMY_FOR_AIRCONDITIONING 25
#define MIN_ELEM_FOR_VEN 25

#define AG_COL_NODE 2
#define AG_COL_ELEM 2
#define AG_COL_DIRICLET 2
#define AG_COL_WALL 7
#define AG_COL_BALANCE 7
#define AG_COL_PRESS 7

struct arch_model
{
    int nobjects; // number of objects
    float size[3]; // size of complete model
    float Q; // flow rate [m3/s]
    float Q_ven; // flow rate ventilators mounted in cube
    int dooropen;
    float spacing;

    struct covise_info *ci; // surface geometry

    // bcs
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

    float bcin_velo[3];

    struct cubus **cubes;
    float zScale;
};

struct archgrid
{
    struct Point *p;
    struct Element *e;

    int *e_to_remove;
    int *p_to_remove;
    int *new_node;
    int *new_elem;

    float spacing_x;
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

    struct Ilist *bcin; // vertex list for inlet boundary surfaces
    struct Ilist *bcinpol; // polygon list for inlet boundary surfaces
    struct Ilist *bcinvol; // the inner volume element on the inlet boundary surfaces
    struct Ilist *bcinvol_outer; // the outer volume element on the inlet boundary surfaces

    struct Ilist *bcout; // vertex list for outlet boundary surfaces
    struct Ilist *bcoutpol; // polygon list for outlet boundary surfaces
    struct Ilist *bcoutvol; // the inner volume element on the outlet boundary surfaces
    struct Ilist *bcoutvol_outer; // the outer volume element on the outlet boundary surfaces

    struct Ilist *bcin_nodes; // inlet nodes list
    struct Flist *bcin_velos; // inlet nodes

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
};

int ReadStartfile(const char *fn, struct arch_model *model);
int DefineBCs(struct archgrid *grid, struct arch_model *model, int number);
int GenerateBCs(struct archgrid *grid, struct arch_model *model, int number);

struct arch_model *AllocModel(void);
void FreeModel(struct arch_model *model);

struct archgrid *CreateArchGrid(struct arch_model *model);
struct covise_info *CreateGeometry4Covise(struct arch_model *model);
int CreateCubePolygons4Covise(struct covise_info *ci, struct arch_model *model, int number, int ipoi, int ipol);
int subtractCube(struct archgrid *grid, struct arch_model *model, int number);
int CreateGeoRbFile(struct archgrid *grid, const char *geofile, const char *rbfile);

int read_string(char *buf, char **str, const char *separator);
int read_int(char *buf, int *izahl, const char *separator);
int read_double(char *buf, double *dzahl, const char *separator);

#endif
