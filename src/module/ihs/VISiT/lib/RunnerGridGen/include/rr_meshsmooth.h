#ifndef RR_MESHSMOOTH_INCLUDE
#define RR_MESHSMOOTH_INCLUDE
int SmoothRR_Mesh(struct Nodelist *nn, struct Element *e, int ge_num,
struct Ilist *psnod, struct Ilist *ssnod,
struct Ilist *psle, struct Ilist *ssle, struct Ilist *pste,
struct Ilist *sste,struct Ilist *inlet,struct Ilist *outlet);
#endif
