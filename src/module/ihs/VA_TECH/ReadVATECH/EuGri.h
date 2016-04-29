#ifndef  EuGri_Included

#define  EuGri_Included

#define  COV_IHS_EULERGRIDPATH   "IHS.EulerGridPath"
#define  ENV_IHS_EULERGRIDPATH   "IHS_EULERGRIDPATH"
#define  COV_IHS_EULERDATAPATH   "IHS.EulerDataPath"
#define  ENV_IHS_EULERDATAPATH   "IHS_EULERDATAPATH"

extern  struct EuGri *ReadEuler(char *grid, char *res, float omega);
extern  void NormEulerGrid(struct EuGri *s, float norm);
extern  struct EuGri *ReadEulerResults(char *fn, struct EuGri *s, float omega);
extern  struct EuGri *ReadEulerGrid(char *fn, float alpha);
extern  void FreeStructEuler(struct EuGri *eu);
struct EuGri *MultiRotateGrid(struct EuGri *s, int nob);

struct EuGri
{
   int i;
   int j;
   int k;
   int num;                                       // = i*j*k
   float norm;
   float *x;
   float *y;
   float *z;
   float *p;
   float *u;                                      // Absolutgeschwindigkeit
   float *v;                                      // Absolutgeschwindigkeit
   float *w;                                      // Absolutgeschwindigkeit
   float omega;                                   // Laufrad
   float *ur;                                     // Relativgeschwindigkeit
   float *vr;                                     // Relativgeschwindigkeit
   float *wr;                                     // Relativgeschwindigkeit
};
#endif                                            // EuGri_Included
