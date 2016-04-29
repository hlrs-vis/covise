#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <windows.h>
#else
#include <strings.h>
#endif
#include <math.h>

#include "../General/include/ihs_cfg.h"
#include "../General/include/geo.h"
#include "../RadialRunner/include/radial.h"
#include "../RadialRunner/include/rr2cov.h"
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/parameter.h"
#include "../General/include/profile.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

#define RR		  "[runner data]"
#define RR_FORCECAMB	"force camber"
#define RR_EULER	"euler equation"
#define RR_PUMP  	"pump"
#define DR_RUNNER	"diagonal runner"
#define RR_CAMB_FUNC	"camber function"
#define RR_NOB	  "number of blades"
#define RR_IDIAM  "shroud inlet diameter"
#define RR_ODIAM  "shroud outlet diameter"
#define RR_SHRD_DIF	 "shroud height difference"
#define RR_IWIDTH "inlet conduit width"
#define RR_OWIDTH "outlet conduit width"
#define RR_EXT_IHDIF "shroud inlet height difference"
#define RR_EXT_OHDIF "shroud outlet height difference"
#define RR_EXT_IDIAM "extension, shroud inlet diameter"
#define RR_EXT_ODIAM "extension, shroud outlet diameter"
#define RR_EXT_IWIDTH	"extension, inlet conduit width"
#define RR_EXT_OWIDTH	"extension, outlet conduit width"
#define RR_EXT_IANGLE	"extension, inlet angle"
#define RR_HSPARA_INEXT "extension, hub params."
#define RR_SSPARA_INEXT "extension, shroud params."
#define RR_SSTPARAM		 "shroud, straight line parameters"
#define RR_HSTPARAM		 "hub, straight line parameters"
#define RR_STCONTOUR	"straight line flags"
#define RR_ICANGLE	 "inlet contour angle"
#define RR_IOPANGLE	 "inlet opening angles"
#define RR_OCANGLE	 "outlet contour angle"
#define RR_OOPANGLE	 "outlet opening angles"
#define RR_HSPARA		"hub spline parameters"
#define RR_SSPARA		"shroud spline parameters"
#define RR_LE	  "[leading edge]"
#define RR_TE	  "[trailing edge]"
#define ED_HCPARAM	 "hub contour parameter"
#define ED_HCANGLE	 "hub off-contour angle"
#define ED_SPARA  "spline parameters"
#define ED_SCPARAM	 "shroud contour parameter"
#define ED_SCANGLE	 "shroud off-contour angle"
#define RR_DES	  "[design data]"
#define DD_DIS	  "machine discharge"
#define DD_HEAD		 "machine head"
#define DD_REVS		 "machine revolutions"
#define DD_VRATIO "inlet velocity ratio"
#define RR_IANGLE "[inlet angle]"
#define RR_MOD_IANGLE  "[inlet angle modification]"
#define RR_OANGLE "[outlet angle]"
#define RR_MOD_OANGLE  "[outlet angle modification]"
#define RR_OROT_ABS "[remaining curl]"
#define RR_PTHICK "[blade thickness]"
#define RR_TETHICK	 "[trailing edge thickness]"
#define RR_BLLENPARA	"[blade length parametre]"
#define RR_CAMBPARA	"[camber parameter]"
#define RR_CAMB		 "[centre line camber]"
#define RR_CAMBPOS	 "[centre line camber pos]"
#define RR_BLADE_LESPLINE_PARA	 "[le spline parameters]"
#define RR_BLADE_TESPLINE_PARA	 "[te spline parameters]"
#define RR_TEWRAP "[trailing edge wrap angle]"
#define RR_BLWRAP "[blade wrap angle]"
#define RR_PROF		 "[blade profile]"
#define RR_BPSHIFT	 "[blade profile shift]"
#define RR_BE	  "[blade element bias]"
#define RR_BENUM  "number of elements"
#define RR_BIAS		"bias factor"
#define RR_BTYPE	"bias type"
#define RR_EXTRA  "extrapolation"
#define DR_ISPHERE	"sphere inlet diameter"
#define DR_OSPHERE	"sphere outlet diameter"
#define DR_SWIDTH	"sphere conduit width"
#define DR_SHEIGHT	"sphere height difference"
#define DR_SOHEIGHT "sphere outlet height"
#define DR_IHUBPARA "hub section inlet stretch parameter"
#define DR_OHUBPARA "hub section outlet stretch parameter"
#ifdef GAP
#define RR_GAP	  "gap width"
#endif											  // GAP
#define INIT_PORTION 20
#define STAT "stat%d"
#define L_LEN 45

#define RAD(x) ((x) * M_PI/180.0)
#define GRAD(x)	  ((x) * 180.0/M_PI)

#ifndef COVISE_MODULE
struct covise_info *Radial2Covise(struct radial *rr);
#endif

struct radial *AllocRadialRunner(void)
{
	struct radial *rr;

	if ((rr = (struct radial *)calloc(1, sizeof(struct radial))) != NULL) {
		if ((rr->le = (struct edge *)calloc(1, sizeof(struct edge))) == NULL)
			fatal("memory for (struct edge *)le");
		if ((rr->te = (struct edge *)calloc(1, sizeof(struct edge))) == NULL)
			fatal("memory for (struct edge *)te");
		if ((rr->des = (struct design *)calloc(1, sizeof(struct design))) == NULL)
			fatal("memory for (struct design *)des");
	}
	else
		fatal("memory for (struct radial *)");

	return rr;
}


int ReadRadialRunner(struct geometry *g, const char *fn)
{
	int err;
	char *tmp;

#ifdef DEBUG_COVISE_PLOT
	void GetCamberPlotData(struct radial *rr, float *xpl, float *ypl, float *xy,
						   int c, int v_count);
	int i, num_points, v_count;
	float *x, *y, xy[4];
	char *fndebug = "plotdata.txt";
	FILE *fpdebug;
#endif

	g->rr = (struct radial *)AllocRadialRunner();

	// runner data
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_NOB)) != NULL) {
		sscanf(tmp, "%d", &g->rr->nob);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_PUMP)) != NULL) {
		sscanf(tmp, "%d", &g->rr->pump);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EULER)) != NULL) {
		sscanf(tmp, "%d", &g->rr->euler);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_RUNNER)) != NULL) {
		sscanf(tmp, "%d", &g->rr->diagonal);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_CAMB_FUNC)) != NULL) {
		sscanf(tmp, "%d", &g->rr->camb2surf);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_FORCECAMB)) != NULL) {
		sscanf(tmp, "%d", &g->rr->camb_flag);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_IDIAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->diam[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_ODIAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ref);
		free(tmp);
		//g->rr->ref	*= 0.5;
		g->rr->diam[1] = 1.0;
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_SHRD_DIF)) != NULL) {
		sscanf(tmp, "%f", &g->rr->height);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_IWIDTH)) != NULL) {
		sscanf(tmp, "%f", &g->rr->cond[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_OWIDTH)) != NULL) {
		sscanf(tmp, "%f", &g->rr->cond[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_STCONTOUR)) != NULL) {
		sscanf(tmp, "%d, %d", &g->rr->straight_cont[0],
			   &g->rr->straight_cont[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_HSTPARAM)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->hstparam[0],&g->rr->hstparam[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_SSTPARAM)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->sstparam[0],&g->rr->sstparam[1]);
		free(tmp);
	}
	// extended geometry modelling
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_IANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_iangle);
		free(tmp);
		g->rr->ext_iangle *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_IHDIF)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_height[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_OHDIF)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_height[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_IDIAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_diam[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_ODIAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_diam[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_IWIDTH)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_cond[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_EXT_OWIDTH)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ext_cond[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_HSPARA_INEXT)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->hspara_inext[0],
			   &g->rr->hspara_inext[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_SSPARA_INEXT)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->sspara_inext[0],
			   &g->rr->sspara_inext[1]);
		free(tmp);
	}
	//
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_ICANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->angle[0]);
		free(tmp);
		g->rr->angle[0] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_IOPANGLE)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->iop_angle[0], &g->rr->iop_angle[1]);
		free(tmp);
		g->rr->iop_angle[0] *= ((float)M_PI/180.0f);
		g->rr->iop_angle[1] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_OCANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->angle[1]);
		free(tmp);
		g->rr->angle[1] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_OOPANGLE)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->oop_angle[0], &g->rr->oop_angle[1]);
		free(tmp);
		g->rr->oop_angle[0] *= ((float)M_PI/180.0f);
		g->rr->oop_angle[1] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_HSPARA)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->hspara[0], &g->rr->hspara[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_SSPARA)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->sspara[0], &g->rr->sspara[1]);
		free(tmp);
	}
	// values for sphere of diagonal runner
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_ISPHERE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->sphdiam[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_OSPHERE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->sphdiam[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_SWIDTH)) != NULL) {
		sscanf(tmp, "%f", &g->rr->sphcond);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_SHEIGHT)) != NULL) {
		sscanf(tmp, "%f", &g->rr->spheight);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_SOHEIGHT)) != NULL) {
		sscanf(tmp, "%f", &g->rr->ospheight);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_IHUBPARA)) != NULL) {
		sscanf(tmp, "%f", &g->rr->stpara[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR, DR_OHUBPARA)) != NULL) {
		sscanf(tmp, "%f", &g->rr->stpara[1]);
		free(tmp);
	}

#ifdef GAP
	if ((tmp = IHS_GetCFGValue(fn, RR, RR_GAP)) != NULL) {
		sscanf(tmp, "%f", &g->rr->gap);
		free(tmp);
	}
#endif

	// leading edge data
	if ((tmp = IHS_GetCFGValue(fn, RR_LE, ED_HCPARAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->le->para[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_LE, ED_HCANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->le->angle[0]);
		free(tmp);
		g->rr->le->angle[0] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_LE, ED_SPARA)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->le->spara[0], &g->rr->le->spara[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_LE, ED_SCPARAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->le->para[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_LE, ED_SCANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->le->angle[1]);
		free(tmp);
		g->rr->le->angle[1] *= ((float)M_PI/180.0f);
	}
	// trailing edge data
	if ((tmp = IHS_GetCFGValue(fn, RR_TE, ED_HCPARAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->te->para[0]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_TE, ED_HCANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->te->angle[0]);
		free(tmp);
		g->rr->te->angle[0] *= ((float)M_PI/180.0f);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_TE, ED_SPARA)) != NULL) {
		sscanf(tmp, "%f, %f", &g->rr->te->spara[0], &g->rr->te->spara[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_TE, ED_SCPARAM)) != NULL) {
		sscanf(tmp, "%f", &g->rr->te->para[1]);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_TE, ED_SCANGLE)) != NULL) {
		sscanf(tmp, "%f", &g->rr->te->angle[1]);
		free(tmp);
		g->rr->te->angle[1] *= ((float)M_PI/180.0f);
	}
	// design data
	if ((tmp = IHS_GetCFGValue(fn, RR_DES, DD_DIS)) != NULL) {
		sscanf(tmp, "%f", &g->rr->des->dis);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_DES, DD_HEAD)) != NULL) {
		sscanf(tmp, "%f", &g->rr->des->head);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_DES, DD_REVS)) != NULL) {
		sscanf(tmp, "%f", &g->rr->des->revs);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_DES, DD_VRATIO)) != NULL) {
		sscanf(tmp, "%f", &g->rr->des->vratio);
		free(tmp);
	}

	// parameters sets: angles, thickness, te, camber, wrap angles, shift
	g->rr->iang = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->iang, RR_IANGLE, fn);
	Parameter2Radians(g->rr->iang);
	g->rr->mod_iang = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->mod_iang,RR_MOD_IANGLE,fn);
	Parameter2Radians(g->rr->mod_iang);
	g->rr->oang = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->oang, RR_OANGLE, fn);
	Parameter2Radians(g->rr->oang);
	g->rr->mod_oang = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->mod_oang,RR_MOD_OANGLE,fn);
	Parameter2Radians(g->rr->mod_oang);
	g->rr->orot_abs = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->orot_abs,RR_OROT_ABS,fn);
	g->rr->t	= AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->t, RR_PTHICK, fn);
	g->rr->tet	= AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->tet, RR_TETHICK, fn);
	g->rr->camb = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->camb, RR_CAMB, fn);
	g->rr->cambpara = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->cambpara, RR_CAMBPARA, fn);
	g->rr->camb_pos = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->camb_pos, RR_CAMBPOS, fn);
	g->rr->bl_lenpara = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->bl_lenpara, RR_BLLENPARA, fn);
	g->rr->tewr = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->tewr, RR_TEWRAP, fn);
	Parameter2Radians(g->rr->tewr);
	g->rr->blwr = AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->blwr, RR_BLWRAP, fn);
	Parameter2Radians(g->rr->blwr);
	g->rr->bps	= AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->bps, RR_BPSHIFT, fn);
	g->rr->le_para	= AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->le_para, RR_BLADE_LESPLINE_PARA, fn);
	g->rr->te_para	= AllocParameterStruct(INIT_PORTION);
	ReadParameterSet(g->rr->te_para, RR_BLADE_TESPLINE_PARA, fn);
#ifdef DDEBUG
	fprintf(stderr, "\ninlet angle data:\n");
	DumpParameterSet(g->rr->iang);
	fprintf(stderr, "\ninlet angle modification:\n");
	DumpParameterSet(g->rr->mod_iang);
	fprintf(stderr, "\noutlet angle data:\n");
	DumpParameterSet(g->rr->oang);
	fprintf(stderr, "\noutlet angle modification:\n");
	DumpParameterSet(g->rr->mod_oang);
	fprintf(stderr, "\nblade thickness data:\n");
	DumpParameterSet(g->rr->t);
	fprintf(stderr, "\ntrailing edge thickness data:\n");
	DumpParameterSet(g->rr->tet);
	fprintf(stderr, "\ncentre line camber data:\n");
	DumpParameterSet(g->rr->camb);
	fprintf(stderr, "\ncentre line camber data position:\n");
	DumpParameterSet(g->rr->camb_pos);
	fprintf(stderr, "\ntrailing edge wrap angle data:\n");
	DumpParameterSet(g->rr->tewr);
	fprintf(stderr, "\nblade wrap angle data:\n");
	DumpParameterSet(g->rr->blwr);
	fprintf(stderr, "\nblade profile shift data:\n");
	DumpParameterSet(g->rr->bps);
#endif										   // DEBUG

	// blade element bias
	if ((tmp = IHS_GetCFGValue(fn, RR_BE, RR_BENUM)) != NULL) {
		sscanf(tmp, "%d", &g->rr->be_num);
		free(tmp);
		if (g->rr->be_num < 2)
			fatal("number of blade elements less than two!");
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_BE, RR_BIAS)) != NULL ) {
		sscanf(tmp, "%f", &g->rr->be_bias);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_BE, RR_BTYPE)) != NULL ) {
		sscanf(tmp, "%d", &g->rr->be_type);
		free(tmp);
	}
	if ((tmp = IHS_GetCFGValue(fn, RR_BE, RR_EXTRA)) != NULL) {
		sscanf(tmp, "%d", &g->rr->extrapol);
		free(tmp);
	}

	// blade profile data
	g->rr->bp = AllocBladeProfile();
	ReadBladeProfile(g->rr->bp, RR_PROF, fn);
#ifdef DDEBUG
	fprintf(stderr, "\nblade profile data:\n");
	DumpBladeProfile(g->rr->bp);
#endif										   // DDEBUG

	// BEGIN OF GEOMERTY CALCULATION
	// IMPLEMENT TO rr2cov.c !!!

	// blade element calculation
	if( (err = CreateRR_BladeElements(g->rr)) ) {
		fprintf(stderr,"ERROR %d in CreateRR_BladeElements\n",err);
	}

#ifndef COVISE_MODULE
	Radial2Covise(g->rr);
#endif										   // COVISE_MODULE

#ifdef DEBUG_COVISE_PLOT
	fprintf(stderr,"ReadRadialRunner(): DEBUG_COVISE_PLOT ...\n\n");
	num_points = 3*g->rr->be_num * (2*(g->rr->be[0]->cl->nump));
	if( (x = (float*)calloc(num_points,sizeof(float))) == NULL) {
		fprintf(stderr,"\n calloc on %d*sizeof(float) failed!\n\n",num_points);
		exit(-1);
	}
	if( (y = (float*)calloc(num_points,sizeof(float))) == NULL) {
		fprintf(stderr,"\n calloc on %d*sizeof(float) failed!\n\n",num_points);
		exit(-1);
	}
	for(i = 0; i < g->rr->be_num; i++) {
		//GetCamberPlotData(g->rr, x, y, xy, i, v_count);
		GetConformalViewPlotData(g->rr, x, y, xy, i, v_count);
		v_count++;
	}
	if( (fpdebug = fopen(fndebug,"w+")) == NULL) {
		fprintf(stderr,"\n could not open file '%s'!\n\n",fndebug);
		exit(-1);
	}
	for(i = 0; i < num_points; i++) {
		fprintf(fpdebug," %16.8f  %16.8f\n",x[i], y[i]);
	}

	fclose(fpdebug);
	free(x);
	free(y);
#endif										   // DEBUG_COVISE_PLOT

#ifdef DEBUG
	DumpRR(g->rr);
#endif

	return 0;
}


int WriteRadialRunner(struct radial *rr, FILE *fp)
{
	int i;
	char buf[200];

	fprintf(stderr," WriteRadialRunner() ...\n");

	// write data to file
	fprintf(fp, "\n%s\n",RR);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_NOB, rr->nob);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_EULER, rr->euler);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_PUMP, rr->pump);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_CAMB_FUNC, rr->camb2surf);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_FORCECAMB, rr->camb_flag);
	fprintf(fp, "\n%s\n", "# reference diameter [m]");
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_ODIAM, rr->ref);
	fprintf(fp, "\n%s\n", "# relative dimensions [-]");
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_IDIAM, rr->diam[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_SHRD_DIF, rr->height);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_IWIDTH, rr->cond[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_OWIDTH, rr->cond[1]);
#ifdef GAP
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_GAP, rr->gap);
#endif
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_ICANGLE, GRAD(rr->angle[0]));
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_IOPANGLE,
			GRAD(rr->iop_angle[0]), GRAD(rr->iop_angle[1]) );
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_OCANGLE, GRAD(rr->angle[1]));
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_OOPANGLE,
			GRAD(rr->oop_angle[0]), GRAD(rr->oop_angle[1]) );
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_HSPARA,
			rr->hspara[0], rr->hspara[1]);
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_SSPARA,
			rr->sspara[0], rr->sspara[1]);
	fprintf(fp, "%*s = %d, %d\n", L_LEN, RR_STCONTOUR,
			rr->straight_cont[0], rr->straight_cont[1]);
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_HSTPARAM,
			rr->hstparam[0], rr->hstparam[1]);
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_SSTPARAM,
			rr->sstparam[0], rr->sstparam[1]);

	// diagonal runner part
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DR_ISPHERE, rr->sphdiam[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DR_OSPHERE, rr->sphdiam[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DR_SWIDTH,  rr->sphcond);
	

	// extension
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_IDIAM, rr->ext_diam[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_ODIAM, rr->ext_diam[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_IHDIF, rr->ext_height[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_OHDIF, rr->ext_height[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_IWIDTH, rr->ext_cond[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_OWIDTH, rr->ext_cond[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_EXT_IANGLE, GRAD(rr->ext_iangle));
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_SSPARA_INEXT,
			rr->sspara_inext[0], rr->sspara_inext[1]);
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, RR_HSPARA_INEXT,
			rr->hspara_inext[0], rr->hspara_inext[1]);

	fprintf(fp, "\n%s\n",RR_LE);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_HCPARAM, rr->le->para[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_HCANGLE, GRAD(rr->le->angle[0]));
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_SCPARAM, rr->le->para[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_SCANGLE, GRAD(rr->le->angle[1]));
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, ED_SPARA,
			rr->le->spara[0], rr->le->spara[1]);
	fprintf(fp, "\n%s\n",RR_TE);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_HCPARAM, rr->te->para[0]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_HCANGLE, GRAD(rr->te->angle[0]));
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_SCPARAM, rr->te->para[1]);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, ED_SCANGLE, GRAD(rr->te->angle[1]) );
	fprintf(fp, "%*s = %9.4f, %9.4f\n", L_LEN, ED_SPARA,
			rr->te->spara[0], rr->te->spara[1]);
	fprintf(fp, "\n%s\n",RR_DES);
	fprintf(fp, "%s\n","# discharge [m3/s], head [m], revs. [rpm]");
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DD_DIS, rr->des->dis);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DD_HEAD, rr->des->head);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DD_REVS, rr->des->revs);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, DD_VRATIO, rr->des->vratio);

	fprintf(fp, "\n%s\n",RR_MOD_IANGLE);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->mod_angle[0]) );
	}

	fprintf(fp, "\n%s\n",RR_MOD_OANGLE);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->mod_angle[1]) );
	}

	fprintf(fp, "\n%s\n",RR_OROT_ABS);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->rot_abs[1]);
	}

	fprintf(fp, "\n%s\n",RR_PTHICK);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->p_thick);
	}

	fprintf(fp, "\n%s\n",RR_TETHICK);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->te_thick);
	}

	fprintf(fp, "\n%s\n",RR_CAMBPARA);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf, 
				rr->be[i]->para, rr->be[i]->camb_para);
	}

	fprintf(fp, "\n%s\n",RR_CAMB);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->camb);
	}

	fprintf(fp, "\n%s\n",RR_CAMBPOS);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->camb_pos);
	}

	fprintf(fp, "\n%s\n",RR_BLLENPARA);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf, 
				rr->be[i]->para, rr->be[i]->bl_lenpara);
	}

	fprintf(fp, "\n%s\n",RR_TEWRAP);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->te_wrap) );
	}

	fprintf(fp, "\n%s\n",RR_BLWRAP);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->bl_wrap) );
	}

	fprintf(fp, "\n%s\n",RR_PROF);
	fprintf(fp, "%*s = %d\n", L_LEN, "naca style", rr->bp->naca);
	for(i = 0; i < rr->bp->num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->bp->c[i], rr->bp->t[i]);
	}

	fprintf(fp, "\n%s\n",RR_BPSHIFT);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->bp_shift);
	}
	fprintf(fp, "\n%s\n",RR_BLADE_LESPLINE_PARA);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->le_para);
	}
	fprintf(fp, "\n%s\n",RR_BLADE_TESPLINE_PARA);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, rr->be[i]->te_para);
	}
	fprintf(fp, "\n%s\n",RR_BE);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_BENUM, rr->be_num);
	fprintf(fp, "%*s = %9.4f\n", L_LEN, RR_BIAS, rr->be_bias);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_BTYPE, rr->be_type);
	fprintf(fp, "%*s = %d\n", L_LEN, RR_EXTRA, rr->extrapol);

	fprintf(fp, "\n%s\n",RR_IANGLE);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->angle[0]) );
	}

	fprintf(fp, "\n%s\n",RR_OANGLE);
	for(i = 0; i < rr->be_num; i++) {
		sprintf(buf, STAT, i);
		fprintf(fp, "%*s = %6.4f, %9.4f\n", L_LEN, buf,
				rr->be[i]->para, GRAD(rr->be[i]->angle[1]) );
	}

	fprintf(stderr," WriteRadialRunner() ... done!\n");
	return (1);
}


#ifdef	DEBUG
void DumpRR(struct radial *rr)
{
	static int fcount = 0;
	char fname[255];
	FILE *ferr;

	sprintf(fname, "rr_struct_%02d.txt", fcount++);
	ferr = fopen(fname, "w");
	if (ferr) {
		fprintf(ferr, "global runner data:\n");
		fprintf(ferr, "rr->nob			= %d\n", rr->nob);
		fprintf(ferr, "rr->diam[0]		= %7.4f\n", rr->diam[0]);
		fprintf(ferr, "rr->diam[1](ref) = %7.4f\n", rr->diam[1]);
		fprintf(ferr, "rr->ref (abs)	= %7.4f\n", rr->ref);
		fprintf(ferr, "rr->height		= %7.4f\n", rr->height);
		fprintf(ferr, "rr->cond[0]		= %7.4f\n", rr->cond[0]);
		fprintf(ferr, "rr->cond[1]		= %7.4f\n", rr->cond[1]);
		fprintf(ferr, "rr->angle[0]		= %7.4f ", rr->angle[0]);
		fprintf(ferr, "(%6.2f)\n", rr->angle[0] * 180.0/M_PI);
		fprintf(ferr, "rr->iop_angle[0]	 = %7.4f ", rr->iop_angle[0]);
		fprintf(ferr, "(%6.2f)\n", rr->iop_angle[0] * 180.0/M_PI);
		fprintf(ferr, "rr->iop_angle[1]	 = %7.4f ", rr->iop_angle[1]);
		fprintf(ferr, "(%6.2f)\n", rr->iop_angle[1] * 180.0/M_PI);
		fprintf(ferr, "rr->angle[1]		= %7.4f ", rr->angle[1]);
		fprintf(ferr, "(%6.2f)\n", rr->angle[1] * 180.0/M_PI);
		fprintf(ferr, "rr->oop_angle[0]	 = %7.4f ", rr->oop_angle[0]);
		fprintf(ferr, "(%6.2f)\n", rr->oop_angle[0] * 180.0/M_PI);
		fprintf(ferr, "rr->oop_angle[1]	 = %7.4f ", rr->oop_angle[1]);
		fprintf(ferr, "(%6.2f)\n", rr->oop_angle[1] * 180.0/M_PI);
		fprintf(ferr, "\nleading edge data:\n");
		fprintf(ferr, "rr->le->para[0]	= %7.4f\n", rr->le->para[0]);
		fprintf(ferr, "rr->le->angle[0] = %7.4f ", rr->le->angle[0]);
		fprintf(ferr, "(%6.2f)\n", rr->le->angle[0] * 180.0/M_PI);
		fprintf(ferr, "rr->le->para[1]	= %7.4f\n", rr->le->para[1]);
		fprintf(ferr, "rr->le->angle[1] = %7.4f ", rr->le->angle[1]);
		fprintf(ferr, "(%6.2f)\n", rr->le->angle[1] * 180.0/M_PI);
		fprintf(ferr, "\ntrailing edge data:\n");
		fprintf(ferr, "rr->te->para[0]	= %7.4f\n", rr->te->para[0]);
		fprintf(ferr, "rr->te->angle[0] = %7.4f ", rr->te->angle[0]);
		fprintf(ferr, "(%6.2f)\n", rr->te->angle[0] * 180.0/M_PI);
		fprintf(ferr, "rr->te->para[1]	= %7.4f\n", rr->te->para[1]);
		fprintf(ferr, "rr->le->angle[1] = %7.4f ", rr->te->angle[1]);
		fprintf(ferr, "(%6.2f)\n", rr->te->angle[1] * 180.0/M_PI);
		fprintf(ferr, "\ndesign data:\n");
		fprintf(ferr, "rr->des->dis		= %8.3f\n", rr->des->dis);
		fprintf(ferr, "rr->des->head	= %8.4f\n", rr->des->head);
		fprintf(ferr, "rr->des->revs	= %8.1f\n", rr->des->revs);
		fprintf(ferr, "\nblade element bias:\n");
		fprintf(ferr, "rr->be_num		= %d\n", rr->be_num);
		fprintf(ferr, "rr->be_bias		= %7.4f (%d)\n", rr->be_bias, rr->be_type);
		fprintf(ferr, "rr->extrapol		= %d\n\n", rr->extrapol);
#ifdef BLADE_ELEMENTS
		fprintf(ferr, "blade elements:\n");
		for(i = 0; i < rr->be_num; i++) {
			fprintf(ferr, "rr->be[%d]->para		 = %7.4f\n", i, rr->be[i]->para);
			fprintf(ferr, "rr->be[%d]->angle[0] = %7.4f ", i, rr->be[i]->angle[0]);
			fprintf(ferr, "(%6.2f)\n", rr->be[i]->angle[0] * 180.0/M_PI);
			fprintf(ferr, "rr->be[%d]->angle[1] = %7.4f ", i, rr->be[i]->angle[1]);
			fprintf(ferr, "(%6.2f)\n", rr->be[i]->angle[1] * 180.0/M_PI);
			fprintf(ferr, "rr->be[%d]->p_thick	= %7.4f\n", i, rr->be[i]->p_thick);
			fprintf(ferr, "rr->be[%d]->te_thick = %7.4f\n", i, rr->be[i]->te_thick);
			fprintf(ferr, "rr->be[%d]->camb		= %7.4f\n", i, rr->be[i]->camb);
			fprintf(ferr, "rr->be[%d]->te_wrap	= %7.4f ", i, rr->be[i]->te_wrap);
			fprintf(ferr, "(%6.2f)\n", rr->be[i]->te_wrap * 180.0/M_PI);
			fprintf(ferr, "rr->be[%d]->bl_wrap	= %7.4f ", i, rr->be[i]->bl_wrap);
			fprintf(ferr, "(%6.2f)\n\n", rr->be[i]->bl_wrap * 180.0/M_PI);
		}
#endif										// BLADE_ELEMENTS
	}
}
#endif											  // DEBUG

#ifdef GNUPLOT
void WriteGNU_RR(struct radial *rr)
{
	int i,j;
	static int ncall = 0;
	float x, y, z;
	FILE *fp = NULL;
	char fn[255];

	sprintf(fn, "rr_blade3d_%02d.txt", ncall++);
	if( (fp = fopen(fn, "w")) == NULL) {
		fprintf(stderr,"Could NOT open file '%s'!\n src: %s, line: %d\n",
				fn, __FILE__, __LINE__);
		exit(-1);
	}
	fprintf(fp, "# centre line\n");
	for (i = 0; i < rr->be_num; i++) {
		for (j = 0; j < rr->be[i]->bp->num; j++) {
			x = rr->be[i]->cl_cart->x[j];
			y = rr->be[i]->cl_cart->y[j];
			z = rr->be[i]->cl_cart->z[j];
			fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");

	fprintf(fp, "# pressure side\n");
	for (i = 0; i < rr->be_num; i++) {
		for (j = 0; j < rr->be[i]->bp->num; j++) {
			x = rr->be[i]->ps_cart->x[j];
			y = rr->be[i]->ps_cart->y[j];
			z = rr->be[i]->ps_cart->z[j];
			fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");

	fprintf(fp, "# suction side\n");
	for (i = 0; i < rr->be_num; i++) {
		for (j = 0; j < rr->be[i]->bp->num; j++) {
			x = rr->be[i]->ss_cart->x[j];
			y = rr->be[i]->ss_cart->y[j];
			z = rr->be[i]->ss_cart->z[j];
			fprintf(fp, "%8.6f %8.6f %8.6f\n", x, y, z);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	fclose(fp);

	// write stations to file
	sprintf(fn,"%s","rr_stations.txt");
	if( (fp = fopen(fn, "w+")) == NULL) {
		fprintf(stderr,"could not open file '%s'!\n",fn);
		exit(-1);
	}
	fprintf(stdout,"stations to gnuplot file '%s'\n", fn);

	fprintf(fp, "# %s\n", RR_MOD_IANGLE);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->mod_angle[0]));

	fprintf(fp, "\n\n# %s\n", RR_MOD_OANGLE);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->mod_angle[1]));

	fprintf(fp, "\n\n# %s\n", RR_OROT_ABS);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->rot_abs[1]);

	fprintf(fp, "\n\n# %s\n", RR_PTHICK);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->p_thick);

	fprintf(fp, "\n\n# %s\n", RR_TETHICK);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->te_thick);

	fprintf(fp, "\n\n# %s\n", RR_CAMB);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->camb);

	fprintf(fp, "\n\n# %s\n", RR_CAMBPOS);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->camb_pos);

	fprintf(fp, "\n\n# %s\n", RR_TEWRAP);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->te_wrap));

	fprintf(fp, "\n\n# %s\n", RR_BLWRAP);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->bl_wrap));

	fprintf(fp, "\n\n# %s\n", RR_BPSHIFT);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, rr->be[i]->bp_shift);

	fprintf(fp, "\n\n# %s\n", RR_IANGLE);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->angle[0]));

	fprintf(fp, "\n\n# %s\n", RR_OANGLE);
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f\n",rr->be[i]->para, GRAD(rr->be[i]->angle[1]));

	// areas
	fprintf(fp, "\n\n# cross section area (para, length, cross sect.\n");
	for(i = 0; i < rr->be[0]->ml->p->nump; i++)
		fprintf(fp,"%6.4f	%7.4f	%9.4f\n",rr->be[0]->ml->par[i],
				rr->be[0]->ml->len[i], rr->be[0]->area->list[i]);

	fprintf(fp, "\n\n# cross section area, only runner\n");
	fprintf(fp,"%6.4f	%7.4f	%9.4f\n",rr->be[0]->ml->par[NPOIN_EXT-1],
			rr->be[0]->ml->len[NPOIN_EXT-1], 0.0);
	for(i = NPOIN_EXT-1; i < (rr->be[0]->ml->p->nump-NPOIN_EXT+1); i++)
		fprintf(fp,"%6.4f	%7.4f	%9.4f\n",rr->be[0]->ml->par[i],
				rr->be[0]->ml->len[i], rr->be[0]->area->list[i]);
	fprintf(fp,"%6.4f	%7.4f	%9.4f\n",rr->be[0]->ml->par[i-1],
			rr->be[0]->ml->len[i-1], 0.0);

	fprintf(fp, "\n\n# cross section area, only runner, normalized param.\n");
	for(i = NPOIN_EXT-1; i < (rr->be[0]->ml->p->nump-NPOIN_EXT+1); i++)
		fprintf(fp,"%6.4f	%7.4f	%9.4f\n",
				(rr->be[0]->ml->par[i] - rr->be[0]->ml->par[NPOIN_EXT-1])
				/(rr->be[0]->ml->par[rr->be[0]->ml->p->nump-NPOIN_EXT]
				  - rr->be[0]->ml->par[NPOIN_EXT-1]),
				rr->be[0]->ml->len[i], rr->be[0]->area->list[i]);

	fprintf(fp, "\n\n# cross section area le/te\n");
	for(i = 0; i < rr->be_num; i++)
		fprintf(fp,"%6.4f	%9.4f	%9.4f\n", rr->be[i]->para,
				rr->be[i]->con_area[0], rr->be[i]->con_area[1]);

	fclose(fp);

}
#endif											  // GNUPLOT
