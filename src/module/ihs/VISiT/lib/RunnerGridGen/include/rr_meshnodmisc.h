#ifndef RR_MESHNODMISC_INCLUDED
#define RR_MESHNODMISC_INCLUDED

#include "../../General/include/nodes.h"
#include "../../General/include/ilist.h"

#ifdef BLNODES_OUT
int PutBladeNodes(struct Nodelist *n, struct Ilist *ssnod,
struct Ilist *psnod, int i, float para);
#endif
#ifdef READJUST_PERIODIC
int ReadjustPeriodic(struct Nodelist *n, struct Ilist *psle,
struct Ilist *ssle,struct Ilist *pste,
struct Ilist *sste, int ge_num, int clock);
#endif
#ifdef DEBUG_NODES
int EquivCheck(struct node **n, int offset);
#endif

#ifdef DEBUG_BC
int DumpBoundaries(struct Nodelist *n, struct Ilist *inlet,
struct Ilist *psle, struct Ilist *ssle,
struct Ilist *psnod, struct Ilist *ssnod,
struct Ilist *pste, struct Ilist *sste,
struct Ilist *outlet, int ge_num);
#endif
#endif                                            // RR_MESHNODMISC_INCLUDED
