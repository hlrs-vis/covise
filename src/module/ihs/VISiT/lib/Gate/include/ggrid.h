#ifndef  GGRID_INCLUDED
#define  GGRID_INCLUDED

#define  GG_COL_NODE        2
#define  GG_COL_ELEM        2
#define  GG_COL_DIRICLET       2
#define  GG_COL_WALL        7
#define  GG_COL_BALANCE     7
#define  GG_COL_PRESS      6

#include <Gate/include/gate.h>
#include <General/include/points.h>
#include <General/include/elements.h>
#include <General/include/vector.h>

struct ggrid
{

   // geometry + connectivity
   struct Point   *p;
   struct Element *e;

   // boundary conditions
   struct Ilist *bcin;                            // cells at entry (corner list)
   struct Ilist *bcinpol;                         // polygon list
   struct Ilist *bcinvol;                         // referring hexa-volume-element

   struct Ilist *bcout;                           // cells at exit (corner list)
   struct Ilist *bcoutpol;                        // polygon list
   struct Ilist *bcoutvol;                        // referring hexa-volume-element
   struct Flist *bcpressval;                      // pressure value list for bcout corners

   struct Ilist *bcwall;                          // hub, shroud and blade cells (corner list)
   struct Ilist *bcwallpol;                       // polygon list
   struct Ilist *bcwallvol;                       // referring hexa-volume-element

   struct Ilist *bcperiodic;                      // periodic boundaries (corner list)
   struct Ilist *bcperiodicpol;                   // polygon list
   struct Ilist *bcperiodicvol;                   // polygon list
   struct Ilist *bcperiodicval;                   // value (left or right periodic border?)

//   struct Vector  *in;
//   struct Vector  *out;
//   struct Vector  *wall;

   float epsilon;
   float k;
   float T;                                       // temperature
   int bc_inval;
   int bc_outval;
   int bc_wall;
   int bc_periodic_left;
   int bc_periodic_right;

};

int   parameter_from_covise(struct stf *stf_para, struct gate *ga,
int *anz_punkte_ls,
double *n_r, double *n_z,
double *k_r, double *k_z,
double *ls_x, double *ls_y,
double *ls_r, double *ls_phi);

void FreeStructGGrid(struct ggrid *gg);
int WriteGGrid(struct ggrid *gg, const char *fn);
int WriteGBoundaryConditions(struct ggrid *gg, const char *fn);

struct ggrid *CreateGGrid(struct gate *ga);

#ifdef   DEBUG
//void DumpGGrid(struct ggrid *gg);
#endif                                            // DEBUG
#endif                                            // GGRID_INCLUDED
