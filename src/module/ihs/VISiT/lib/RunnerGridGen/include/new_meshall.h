#ifndef NEW_MESHALL_INCLUDE
#define NEW_MESHALL_INCLUDE

#include "../../General/include/curve.h"

int MeshNew_InletRegion(struct Nodelist *n, struct curve *ml, struct Ilist *inlet, struct Ilist *psle,
struct Ilist *ssle, struct region *reg);
int MeshNew_SSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *ssnod, struct Ilist *ssle,
struct region *reg, struct region *reg0, int le_dis);
int MeshNew_PSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *psnod, struct Ilist *psle,
struct region *reg, struct region *reg0, int le_dis);
int MeshNew_CoreRegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg0,
struct region *reg1, struct region *reg3,
float angle14);
int   MeshNew_SSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct Ilist *ssnod, struct Ilist *sste, struct Ilist *outlet);
int MeshNew_OutletRegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct region *reg4, struct region *reg2, struct Ilist *outlet);
int MeshNew_PSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg3,
struct region *reg5, struct Ilist *psnod, struct Ilist *pste, struct Ilist *outlet);
#ifndef NO_INLET_EXT
int MeshNew_ExtRegion(struct Nodelist *n, struct curve *ml, struct Ilist *inlet, struct Ilist *psle,
struct Ilist *ssle, struct region *reg, struct region *reg0);
#endif
#ifdef GAP
int MeshNew_SSGapRegion(struct Nodelist *n, struct curve *ml,
struct Ilist *sste, struct Ilist *ssnod,
struct region *reg, struct region *reg1,
struct region *reg4, int itip);
int MeshNew_PSGapRegion(struct Nodelist *n, struct curve *ml,
struct Ilist *pste, struct Ilist *psnod,
struct region *reg, struct region *reg3,
struct region *reg6, int itip);
#endif
#endif
