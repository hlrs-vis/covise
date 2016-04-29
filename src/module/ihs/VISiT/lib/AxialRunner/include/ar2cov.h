#ifndef  AR_COV_H_INCLUDED
#define  AR_COV_H_INCLUDED

#include "../../General/include/cov.h"

extern  struct covise_info *Axial2Covise(struct axial *ar);
extern  void CreateAR_BEPolygons(struct covise_info *ci, int be, int offset, int te);
extern  void CreateAR_TipPolygons(struct covise_info *ci, int npoin, int te);
extern  void RotateBlade4Covise(struct covise_info *ci, int nob);
extern  void CreateAR_CoviseContours(struct covise_info *ci, struct axial *ar);
extern  void CreateContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_vx, int sec, int np, int npb, int hub);
void GetXMGRCommands(char *plbuf, float *xy,const char *title,const char *xlabel,const char *ylabel, int q_flag);
void GetMeridianContourPlotData(struct axial *ar, float *xpl, float *ypl,
int ext_flag);
void GetMeridianContourNumbers(int *num_points, float *xy, struct axial *ar,
int ext_flag);
void GetEulerAnglesPlotData(struct axial *ar, float *xpl,
float *ypl, float *xy);
void GetCamberPlotData(struct axial *ar, float *xpl, float *ypl, float *xy,
int c, int v_count);
void GetNormalizedCamber(struct axial *ar, float *xpl, float *ypl, float *xy,
int c, int v_count);
void GetConformalViewPlotData(struct axial *ar, float *xpl, float *ypl,
float *xy, int c, int v_count);
void GetMaxThicknessData(struct axial *ar, float *xpl, float *ypl, float *xy);
void GetMaxThicknessDistrib(struct axial *ar,float *xpl, float *ypl,float *xy);
void GetBladeAnglesPlotData(struct axial *ar, float *xpl,float *ypl,float *xy);
void GetOverlapPlotData(struct axial *ar, float *xpl, float *ypl, float *xy);
void GetChordAnglesPlotData(struct axial *ar,float *xpl,float *ypl,float *xy);
void GetParamPlotData(struct axial *ar, float *xpl, float *ypl,float *xy,
int ival);
char *GetLastErr(void);
int PutBladeData(struct axial *ar);
#endif                                            // AR_COV_H_INCLUDED
