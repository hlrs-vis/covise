#ifndef NODES_INCLUDE
#define NODES_INCLUDE

#include "curve.h"

#define ARC 0
#define PHI 1

struct node
{
   int id;
   int index;
   float phi;                                     // circumf. angle
   float l;                                       // meridional length
   float r;                                       // radius
   float arc;                                     // arc length on circumference
   float x;                                       // x,y,z in cartesian system
   float y;
   float z;
};

struct Nodelist
{
   int max;                                       // max. id number
   int num;                                       // number of nodes
   int   offset;                                  // node number offset between meridian planes
   int portion;
   struct node **n;
};

struct Nodelist *AllocNodelistStruct(void);
int AddNode(struct Nodelist *n, float phi, float l, float r, int flag);
int AddVNode(struct Nodelist *n, float x[3], int flag);
void FreeNodelistStruct(struct Nodelist *n);
int CalcNodeRadius(struct node **n, struct curve *ml, int nnum);
int CalcNodeAngle(struct node **n, struct curve *ml, int nnum);
int CalcNodeCoords(struct node **n, int nnum, int clock);

#ifdef DEBUG_NODES
int DumpNodes(struct node **n, int nnum, FILE *fp);
#endif
#endif
