#ifndef RR_MESHALL_INCLUDE
#define RR_MESHALL_INCLUDE

#include "../../General/include/curve.h"

int MeshRR_InletRegion(struct Nodelist *n, struct curve *ml, struct Ilist *inlet, struct Ilist *psle,
struct Ilist *ssle, struct region *reg);
int MeshRR_SSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *ssnod, struct Ilist *ssle,
struct region *reg, struct region *reg0, int jadd);
int MeshRR_PSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *psnod, struct Ilist *psle,
struct region *reg, struct region *reg0, int jadd);
int MeshRR_CoreRegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg0,
struct region *reg1, struct region *reg3,
float angle14, int jadd);
int MeshRR_SSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct Ilist *ssnod, struct Ilist *sste, struct Ilist *outlet);
int MeshRR_OutletRegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg1,
struct region *reg4, struct region *reg2, struct Ilist *outlet);
int MeshRR_PSTERegion(struct Nodelist *n, struct curve *ml, struct region *reg, struct region *reg3,
struct region *reg5, struct Ilist *psnod, struct Ilist *pste, struct Ilist *outlet);
int MeshMod_SSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *ssnod,
struct Ilist *ssle, struct region *reg, struct region *reg0, int ii1);
int MeshMod_PSRegion(struct Nodelist *n, struct curve *ml, struct Ilist *psnod, struct Ilist *psle,
struct region *reg, struct region *reg0, int ii1);
#ifndef NO_INLET_EXT
int MeshRR_ExtRegion(struct Nodelist *n, struct curve *ml, struct Ilist *inlet, struct Ilist *psle,
struct Ilist *ssle, struct region *reg, struct region *reg0);
#endif
#ifdef GAP
int MeshRR_SSGapRegion(struct Nodelist *n, struct curve *ml,
struct Ilist *sste, struct Ilist *ssnod,
struct region *reg, struct region *reg1,
struct region *reg4, int itip);
int MeshRR_PSGapRegion(struct Nodelist *n, struct curve *ml,
struct Ilist *pste, struct Ilist *psnod,
struct region *reg, struct region *reg3,
struct region *reg6, int itip);
#endif
#endif
