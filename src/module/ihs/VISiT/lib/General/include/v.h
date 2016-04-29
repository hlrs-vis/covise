#ifndef  V_H_INCLUDED
#define  V_H_INCLUDED

#ifdef   DEBUG
#define  V_DEB(x) fprintf(stderr, "%s = (%f,%f,%f)\n", #x, x[0], x[1], x[2])
#else
#define  V_DEB(x)
#endif
void V_Add(float *a, float *b, float *res);
void V_Sub(float *a, float *b, float *res);
void V_Norm(float *a);
float V_Angle(float *a, float *b);
float V_ScalarProduct(float *a, float *b);
void V_MultScal(float *a, float s);
float V_Len(float *a);
void V_Copy(float *s, float *d);
void V_0(float *a);
#endif                                            // V_H_INCLUDED
