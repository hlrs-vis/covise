#ifndef AR_CREATEBE_H_INCLUDED
#define AR_CREATEBE_H_INCLUDED

extern  int CreateAR_BladeElements(struct axial *ar);
extern  int ModifyAR_BladeElements4Covise(struct axial *ar);
extern  int SurfacesAR_BladeElement(struct be *be, float angle,
float ref, int clock, int clspline);
extern  void DetermineCoefficients(float *x, float *y, float *a);
extern  float EvaluateParameter(float x, float *a);
#endif                                            // AR_CREATEBE_H_INCLUDED
