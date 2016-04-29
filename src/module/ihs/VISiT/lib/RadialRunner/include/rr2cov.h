#ifndef  RR_COV_H_INCLUDED
#define  RR_COV_H_INCLUDED

#include "../../General/include/cov.h"

struct covise_info *Radial2Covise(struct radial *rr);
char *GetLastErr(void);
void CreateRR_BEPolygons(struct covise_info *ci, int be, int offset, int npol);
void CreateRR_TipPolygons(struct covise_info *ci, int npoin, int te);
int RotateBlade4Covise(struct covise_info *ci, int nob);
int CreateRR_Contours(struct covise_info *ci, struct radial *rr);
void CreateContourPolygons(struct Ilist *ci_pol, struct Ilist *ci_xv, int sec, int np, int npb, int hub);
void GetXMGRCommands(char *plbuf, float *xy,const char *title,const char *xlabel,const char *ylabel, int q_flag);
void GetMeridianContourNumbers(int *num_points, float *xy, struct radial *rr, int ext_flag);
void GetMeridianContourPlotData(struct radial *rr, float *xpl, float *ypl, int num_points, int ext_flag);
void GetMeridianContourPlotData2(struct radial *rr, float *xpl, float *ypl, int num_points, int ext_flag);
void GetConformalViewPlotData(struct radial *rr, float *xpl, float *ypl, float *xy, int c, int v_count);
void GetCamberPlotData(struct radial *rr, float *xpl, float *ypl, float *xy,
int c, int v_count);
void GetNormalizedCamber(struct radial *rr, float *xpl, float *ypl, float *xy,
int c, int v_count);
void GetMaxThicknessData(struct radial *rr, float *xpl, float *ypl, float *xy);
void GetOverlapPlotData(struct radial *rr, float *xpl, float *ypl, float *xy);
void GetBladeAnglesPlotData(struct radial *rr, float *xpl, float *ypl,
float *xy);
void GetEulerAnglesPlotData(struct radial *rr, float *xpl,
							float *ypl, float *xy);
void GetMeridianVelocityPlotData(struct radial *rr, float *xpl,
								 float *ypl, float *xy);
void GetCircumferentialVelocityPlotData(struct radial *rr, float *xpl,
										float *ypl, float *xy);
void AddXMGRSet2Plot(char *plbuf, float *xy, float *x, float *y, int num,
char *title, char *xlabel, char *ylabel,
int q_flag, int graph, int set);
int PutBladeData(struct radial *rr);
#endif                                            // RR_COV_H_INCLUDED
