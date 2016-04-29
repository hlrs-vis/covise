#ifndef BIAS_H_INCLUDED
#define BIAS_H_INCLUDED

extern  struct Flist *CalcBladeElementBias(int nodes, float t1, float t2, int type, float ratio);
extern  struct Flist *Add2Bias(struct Flist *bias, int nodes, float t1, float t2,
int type, float ratio, int first);
#endif                                            // BIAS_H_INCLUDED
