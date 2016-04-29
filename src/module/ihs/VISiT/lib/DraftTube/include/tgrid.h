#ifndef  TGRID_INCLUDED
#define  TGRID_INCLUDED

#define  TG_COL_NODE    2
#define  TG_COL_ELEM    2
#define  TG_COL_DIRICLET   2
#define  TG_COL_WALL    6
#define  TG_COL_BALANCE 6

#include "tube.h"
#include "../../General/include/points.h"
#include "../../General/include/elements.h"
#include "../../General/include/vector.h"

struct gs
{
   float rr[8][3];                                // from (0,0,0) to the rectangle points
   float dr[8][3];                                // dr[i] = rr[i+1] - rr[i] / cols
   float ro[8][3];                                // from (0,0,0) to the outer points
   float rm[8][3];                                // from (0,0,0) to the inner points
   float o[8][3];                                 // o[i] = ro[i+1] - ro[i]
   float m[8][3];                                 // m[i] = rm[i+1] - rm[i]
   float dm[8][3];                                // m[i] / cols
   float part[8];
   float linfact;                                 // the length of the innerst element of the outer section
   // should be linfact-times as the first
   int num_elems;
   struct Point *p;
};

struct tgrid
{
   struct Point   *p;
   struct Element *e;
   struct Vector  *in;
   struct Vector  *out;
   struct Vector  *wall;

   // now Params for Grid-generation are following
   int num_i[4];                                  // num of inner nodes
   int num_o;                                     // num of outer nodes
   int num_cs;                                    // num of points between the cs
   int numiP[4];                                  // number of grid points in this inner section
   int numisP[4];                                 // number (sum) of grid points in every inner section
   int numiSP[4];                                 // startindex (relativ) of th first point of this section
   int numoP[8];                                  // number of grid points in this outer section
   int numosP[8];                                 // number (sum) of grid points in every outer section
   int numoSP[8];                                 // startindex (relativ) of the first point of this section
   float epsilon;
   float k;
   float T;                                       // temperature
   int bc_inval;
   int bc_outval;
   int bc_wall;

   int gs_num;
   int gs_max;
   struct gs **gs;
};

void FreeStructTGrid(struct tgrid *tg);
void FreeStructT_GS(struct gs *gs);
int WriteTGrid(struct tgrid *tg, const char *fn);
int WriteTBoundaryConditions(struct tgrid *tg, const char *fn);
int GetiRowPoints(struct tgrid *tg, int ind);
int GetiColPoints(struct tgrid *tg, int ind);
int GetoRowPoints(struct tgrid *tg);
int GetoColPoints(struct tgrid *tg, int ind);
int GetiRowElems(struct tgrid *tg, int ind);
int GetiColElems(struct tgrid *tg, int ind);
int GetoRowElems(struct tgrid *tg);
int GetoColElems(struct tgrid *tg, int ind);
struct tgrid *CreateTGrid(struct tube *tu);
#ifdef   DEBUG
void DumpTGrid(struct tgrid *tg);
#endif                                            // DEBUG
#endif                                            // TGRID_INCLUDED
