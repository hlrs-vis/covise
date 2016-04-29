#ifndef  COVISE_H_INCLUDED
#define  COVISE_H_INCLUDED

#include "points.h"
#include "ilist.h"

struct ci_cs
{
   struct Point *p;
   struct Ilist *cvx;
};

struct covise_info
{
   // points are for both
   struct Point *p;

   // this is for the crossections
   int num_cs;
   struct ci_cs **ci_cs;

   struct Ilist *cpol;
   struct Ilist *cvx;

   // this is for the entry surface
   int bcinnumPoints;
   struct Ilist *bcinpol;
   struct Ilist *bcinvx;

   // this is the geometry
   struct Ilist *pol;
   struct Ilist *vx;

   // this is the geometry (in line format)
   struct Ilist *lpol;
   struct Ilist *lvx;

};

extern  struct covise_info *AllocCoviseInfo(int cs_num);
extern  struct ci_cs *AllocCiCsStruct(void);
extern  void FreeCiCsStruct(struct ci_cs *c);
extern  void FreeCoviseInfo(struct covise_info *ci);
#endif                                            // COVISE_H_INCLUDED
