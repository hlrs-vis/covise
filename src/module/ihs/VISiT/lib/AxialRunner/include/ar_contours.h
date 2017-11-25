#ifndef AR_CONTOURS_H_INCLUDED
#define AR_CONTOURS_H_INCLUDED

#define POST_HUB_CORE      0.8f

#define NPOIN_MAX       15
#define NPOIN_LINEAR    4
#define NPOIN_OUTLET    10
#define SMOOTH_HBEND    16
#define SMOOTH_COUNTER     10
#define NOS_SPHERE_ARC     12

#define ELLIPSE_BIAS    4.0f

#define NPOIN_SPLN_INLET   10
#define NPOIN_SPLN_BEND    20
#define NPOIN_SPLN_CORE    20
#define NPOIN_SPLN_OUTLET   8

extern  int CreateAR_Contours(struct axial *ar);
extern  void CreateContourWireframe(struct curve *c, int nump);
#endif                                            // AR_CONTOURS_H_INCLUDED
