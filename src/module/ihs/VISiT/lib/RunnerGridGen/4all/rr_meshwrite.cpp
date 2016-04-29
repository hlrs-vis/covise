#include <stdio.h>
#include <stdlib.h>
#ifdef WIN32
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>

#include <General/include/flist.h>
#include <General/include/ilist.h>
#include <General/include/points.h>
#include <General/include/nodes.h>
#include <General/include/elements.h>
#include <General/include/fatal.h>

#include <RunnerGridGen/include/rr_grid.h>

#ifdef FENFLOSS_OUT

#define ROT 1
#define FIX 0

#define ROTLABEL 11
#define FIXLABEL 10

#define INLET		100
#define OUTLET		200
#define ALLWALL		150
#define PSSURFACE	202
#define SSSURFACE	202

#ifdef RR_IONODES
#define RRINLET		101
#define RROUTLET	102
#endif

#if !defined(AXIAL_RUNNER) && !defined(RADIAL_RUNNER) && !defined(DIAGONAL_RUNNER)
#error One of AXIAL_RUNNER, RADIAL_RUNNER or DIAGONAL_RUNNER has to be defined
#endif

#ifdef AXIAL_RUNNER
#define PSLEPERIOD	110
#define SSLEPERIOD	120
#define PSTEPERIOD	110
#define SSTEPERIOD	120
#endif
#ifdef RADIAL_RUNNER
#define PSLEPERIOD	110
#define SSLEPERIOD	120
#define PSTEPERIOD	110
#define SSTEPERIOD	120
#endif
#ifdef DIAGONAL_RUNNER
#define PSLEPERIOD	110
#define SSLEPERIOD	120
#define PSTEPERIOD	110
#define SSTEPERIOD	120
#endif
#define PRESS_NUM	10
#define PRESS_VAL	0.0
#ifndef MIN
#define MIN(a,b)  ( (a) <  (b) ? (a) :	(b) )
#endif
#ifndef MAX
#define MAX(a,b)  ( (a) >  (b) ? (a) :	(b) )
#endif

static int PutDispNodes(struct Ilist *inlet, float **bcval, FILE *fp);
static int PutWallElement(struct Element *e, FILE *fp, int flag);
static int PutBilaElement(struct Element *e, FILE *fp, float flag);
static int PutPressElement(struct Element *e, FILE *fp, int flag);
static int PutBCElement(struct Element *e, FILE *fp, int flag, const char *type);

int WriteFENFLOSS_Geofile(struct Nodelist *n, struct Element *e)
{
	int i, j;

	struct node **nod;
	int **elem;

	const char *fn = "complete.geo";
	FILE *fp = NULL;

	fprintf(stderr,"\nWriteFENFLOSS_Geofile: FENFLOSS-Geofile '%s' ... ",fn);fflush(stderr);

	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"\n Shit happened opening file '%s'!\n\n", fn);
		exit(-1);
	}

	fprintf(fp,"######################################################################\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#  Automatically created FENFLOSS-Geometry file						 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"######################################################################\n");
	fprintf(fp,"%7d %7d %7d %7d %7d %7d %7d %7d\n", n->num, e->nume,
			0, 0, 0, 0, n->num, e->nume);

	nod = n->n;
	for(i = 0; i < n->num; i++) {
		fprintf(fp,"%8d %16.6f %16.6f %16.6f %6d\n",
				(*nod)->id, (*nod)->x, (*nod)->y, (*nod)->z, 0);
		nod++;
	}
	elem = e->e;
	for(i = 0; i < e->nume; i++) {
		fprintf(fp,"%8d",i+1);
		for(j = 0; j < 8; j++) {
			fprintf(fp,"  %8d",(*elem)[j]+1);
		}
		fprintf(fp,"  %8d\n", 0);
		elem++;
	}

	fprintf(stderr,"done!\n");
	fclose(fp);
	return 0;
}


int WriteFENFLOSS_BCfile(struct Nodelist *, struct Element *,
						 struct Element *wall, struct Element *frictless,
						 struct Element *shroud, struct Element *shroudext,
						 struct Element *psblade, struct Element *ssblade,
						 struct Element *psleperiodic,
						 struct Element *ssleperiodic,
						 struct Element *psteperiodic,
						 struct Element *ssteperiodic,
						 struct Element *inlet, struct Element *outlet,
						 struct Element *rrinlet, struct Element *rroutlet,
						 struct Ilist *innodes, float **bcval, int rot_flag)
{
	int inletnum, wallnum, bilanum, pressnum;

	const char *fn = "complete.bc";
	FILE *fp = NULL;

	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"\n Shit happened opening file '%s'!\n\n", fn);
		exit(-1);
	}

	if(bcval) inletnum = innodes->num*5;
	else inletnum = 0;
	wallnum	 = wall->nume +
		shroud->nume +
		shroudext->nume +
		frictless->nume;
	pressnum = outlet->nume;
	bilanum	 = inlet->nume +
		outlet->nume +
		psleperiodic->nume +
		ssleperiodic->nume +
		psteperiodic->nume +
		ssteperiodic->nume +
		psblade->nume +
		ssblade->nume +
		wall->nume +
		shroud->nume +
		shroudext->nume +
		frictless->nume;
#ifdef RR_IONODES
	bilanum += (rrinlet->nume + rroutlet->nume);
#endif

	fprintf(fp,"######################################################################\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#  Automatically created FENFLOSS-BC file							 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#  Format Version 6.0												 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"######################################################################\n");
	fprintf(fp," %7d %7d %7d %7d %7d %7d %7d\n", inletnum, wallnum,
			pressnum, 0, 0, 0, bilanum);

	fprintf(stderr,"\nWriteFENFLOSS_BCfile: FENFLOSS-BCfile: '%s'\n",fn);
#ifdef DEBUG_BC
	fprintf(stderr," WriteFENFLOSS_BCfile: inlet->nume				  = %d\n",inlet->nume);
	fprintf(stderr," WriteFENFLOSS_BCfile: inlet->e[inlet->nume-1][0] = %d\n",inlet->e[inlet->nume-1][0]);
	fprintf(stderr," WriteFENFLOSS_BCfile: wall->nume				= %d\n",wall->nume);
	fprintf(stderr," WriteFENFLOSS_BCfile: wall->e[wall->nume-1][0] = %d\n",wall->e[wall->nume-1][0]);
	fprintf(stderr," WriteFENFLOSS_BCfile: shroud->nume					= %d\n",shroud->nume);
	fprintf(stderr," WriteFENFLOSS_BCfile: shroud->e[shroud->nume-1][0] = %d\n",shroud->e[shroud->nume-1][0]);
#endif

	if(inletnum) PutDispNodes(innodes,bcval,fp);
	PutWallElement(wall, fp, ROT);
	PutWallElement(frictless, fp, ROT);
#ifdef GAP
	PutWallElement(shroud, fp, FIX);
#else
#ifdef AXIAL_RUNNER
	if(rot_flag)
		PutWallElement(shroud, fp, ROT);
	else
		PutWallElement(shroud, fp, FIX);
#endif										   // AXIAL_RUNNER
#ifdef RADIAL_RUNNER
	PutWallElement(shroud, fp, ROT);
#endif										   // RADIAL_RUNNER
#endif										   // GAP
	if(rot_flag)
		PutWallElement(shroudext, fp, ROT);
	else
		PutWallElement(shroudext, fp, FIX);
	PutPressElement(outlet, fp, PRESS_NUM);
	PutBilaElement(inlet, fp, INLET);
	PutBilaElement(outlet, fp, OUTLET);
	PutBilaElement(psleperiodic, fp, PSLEPERIOD);
	PutBilaElement(ssleperiodic, fp, SSLEPERIOD);
	PutBilaElement(psteperiodic, fp, PSTEPERIOD);
	PutBilaElement(ssteperiodic, fp, SSTEPERIOD);
	PutBilaElement(psblade, fp, PSSURFACE);
	PutBilaElement(ssblade, fp, SSSURFACE);
	PutBilaElement(rrinlet, fp, RRINLET);
	PutBilaElement(rroutlet, fp, RROUTLET);
	PutBilaElement(wall, fp, ALLWALL);
	PutBilaElement(frictless, fp, ALLWALL);
	PutBilaElement(shroud, fp, ALLWALL);
	PutBilaElement(shroudext, fp, ALLWALL);

	fclose(fp);
	return 0;
}


static int PutDispNodes(struct Ilist *inlet, float **bcval, FILE *fp)
{
	int i, j, node_id;
	int *ilist;

	float *val;

	if(!bcval || !inlet || !inlet->list) {
		fatal(" One of the objects needed disappeared somehow!");
		return 1;
	}

	ilist = inlet->list;
	// !! bcval[0] has to be allocated in ONE chunk!! see rr_grid.c
	val	  = bcval[0];
	for(i = 0; i < inlet->num; i++, ilist++) {
		node_id = (*ilist) + 1;
		for(j = 0; j < 5; j++, val++)
			fprintf(fp," %8d %3d %16.6f\n",node_id,j+1,*val);
	}
	return 0;
}


static int PutWallElement(struct Element *e, FILE *fp, int flag)
{
	int i;
	int **elem;

	elem = e->e;
	for(i = 0; i < e->nume; i++, elem++) {
		fprintf(fp," %8d %8d %8d %8d %2d %2d %2d %8d\n", (*elem)[1]+1, (*elem)[2]+1, (*elem)[4]+1,
				(*elem)[3]+1, flag, 0, 0, (*elem)[0]+1);
	}
	return(1);
}


static int PutBilaElement(struct Element *e, FILE *fp, float flag)
{
	int i;
	int **elem;

	elem = e->e;
	for(i = 0; i < e->nume; i++, elem++) {
		fprintf(fp," %8d %8d %8d %8d %8d %16.6f\n", (*elem)[1]+1, (*elem)[2]+1,
				(*elem)[4]+1, (*elem)[3]+1, (*elem)[0]+1, flag);
	}
	return 1;
}


static int PutPressElement(struct Element *e, FILE *fp, int flag)
{
	int i;
	int **elem;

	elem = e->e;
	for(i = 0; i < e->nume; i++, elem++) {
		fprintf(fp," %8d %8d %8d %8d  %8d  %14.6f %8d\n",(*elem)[1]+1, (*elem)[2]+1,
				(*elem)[4]+1, (*elem)[3]+1, flag, PRESS_VAL, (*elem)[0]+1);
	}
	return 1;
}


int WriteFENFLOSS62x_BCfile(struct Nodelist *, struct Element *,
							struct Element *wall, struct Element *frictless,
							struct Element *shroud, struct Element *shroudext,
							struct Element *psblade, struct Element *ssblade,
							struct Element *psleperiodic,
							struct Element *ssleperiodic,
							struct Element *psteperiodic,
							struct Element *ssteperiodic,
							struct Element *inlet, struct Element *outlet,
							struct Element *rrinlet, struct Element *rroutlet,
							struct Ilist *innodes, float **bcval, int rot_flag)
{
	int inletnum, wallnum, bilanum, pressnum;

	const char *fn = "complete62x.bc";
	FILE *fp = NULL;

	if( (fp = fopen(fn,"w+")) == NULL) {
		fprintf(stderr,"\n Shit happened opening file '%s'!\n\n", fn);
		exit(-1);
	}
	fprintf(stderr," WriteFENFLOSS62x_BCfile()... \n");fflush(stderr);

	if(bcval) inletnum = innodes->num*5;
	else inletnum = 0;
	wallnum	 = wall->nume +
		shroud->nume +
		shroudext->nume +
		frictless->nume;
	pressnum = outlet->nume;
	bilanum	 = inlet->nume +
		outlet->nume +
		psleperiodic->nume +
		ssleperiodic->nume +
		psteperiodic->nume +
		ssteperiodic->nume +
		psblade->nume +
		ssblade->nume +
		wall->nume +
		shroud->nume +
		shroudext->nume +
		frictless->nume;
#ifdef RR_IONODES
	bilanum += (rrinlet->nume + rroutlet->nume);
#endif

	fprintf(fp,"######################################################################\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#  Automatically created FENFLOSS-BC file							 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#  Format Version 6.2.x												 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"#																	 #\n");
	fprintf(fp,"######################################################################\n");
	fprintf(fp," %7d %7d %7d %7d %7d %7d %7d\n",
			inletnum, wallnum, pressnum, 0, 0, 0, bilanum);

	fprintf(stderr,"\nWriteFENFLOSS_BCfile: FENFLOSS-BCfile: '%s'\n",fn);

	if(inletnum) PutDispNodes(innodes,bcval,fp);
	PutBCElement(wall, fp, ROTLABEL, "wand");
	PutBCElement(frictless, fp, ROTLABEL, "wand");
#ifdef GAP
	PutBCElement(shroud, fp, FIXLABEL, "wand");
#else
	PutBCElement(shroud, fp, FIXLABEL, "wand");
#endif
	if(rot_flag)
		PutBCElement(shroudext, fp, ROTLABEL, "wand");
	else
		PutBCElement(shroudext, fp, FIXLABEL, "wand");
	PutBCElement(outlet, fp, PRESS_NUM, "pres");
	PutBCElement(inlet, fp, INLET, "bila");
	PutBCElement(outlet, fp, OUTLET, "bila");
	PutBCElement(psleperiodic, fp, PSLEPERIOD, "bila");
	PutBCElement(ssleperiodic, fp, SSLEPERIOD, "bila");
	PutBCElement(psteperiodic, fp, PSTEPERIOD, "bila");
	PutBCElement(ssteperiodic, fp, SSTEPERIOD, "bila");
	PutBCElement(psblade, fp, PSSURFACE, "bila");
	PutBCElement(ssblade, fp, SSSURFACE, "bila");
	PutBCElement(rrinlet, fp, RRINLET, "bila");
	PutBCElement(rroutlet, fp, RROUTLET, "bila");
	PutBCElement(wall, fp, ALLWALL, "bila");
	PutBCElement(frictless, fp, ALLWALL, "bila");
	PutBCElement(shroud, fp, ALLWALL, "bila");
	PutBCElement(shroudext, fp, ALLWALL, "bila");

	fclose(fp);
	return 0;
}


static int PutBCElement(struct Element *e, FILE *fp, int flag, const char *type)
{
	int i;
	int **elem;

	elem = e->e;
	for(i = 0; i < e->nume; i++, elem++) {
		fprintf(fp," %8d %8d %8d %8d %8d %4d %10.4f %s\n", (*elem)[1]+1, (*elem)[2]+1,
				(*elem)[4]+1, (*elem)[3]+1, (*elem)[0]+1, flag, 0.0, type);
	}
	return 0 ;
}
#endif

#define SMALL 1.0e-5f
#define VISCO 1.0e-5f
static float getExp(float);
int CreateInletBCs(struct Ilist *inlet, struct Nodelist *n,struct bc *inbc,
				   float ***bcval, int alpha_const, int turb_prof)
{
	int i;
	int *ilist;

	float rmax,rmin,rmid,deltar, zmax,zmin,deltaz, area, deltal;
        float beta, cm, cu, cm_r, cu_r; //, crel;
	float ca, curmid, bccm, **tmpbcval, *val;
	float Re, exp=0.0f, cein, cmmax, cunod, zmid, dz=0.0f, tf;
	
	struct node *nod;
	
	const char *fn = "inbc.dat";
	FILE *fp = NULL;
	
	if( (fp = fopen(fn,"w+")) == NULL)
		fprintf(stderr,"Could not open file '%s'!\n",fn);

	fprintf(stderr," CreateInletBCs() ... \n");

	// find intermediate radius and delta z.
	// straight inlet supposed
	if(!(ilist = inlet->list)) {
		fatal("list with inlet nodes does not exist!!");
		return -1;
	}
	rmax  = rmin = n->n[*ilist]->r;
	zmax  = zmin = n->n[*ilist]->z;
	ilist++;
	for(i = 1; i < inlet->num; i++,ilist++) {
		nod = n->n[*ilist];
		if(rmax < nod->r)	   rmax = nod->r;
		else if(rmin > nod->r) rmin = nod->r;
		if(zmax < nod->z)	   zmax = nod->z;
		else if(zmin > nod->z) zmin = nod->z;
	}
	rmid   = 0.5*(rmin+rmax);
	zmid   = 0.5*(zmax+zmin);
	deltar = rmax-rmin;
	deltaz = zmax-zmin;
	deltal = sqrt(pow(deltar,2)+pow(deltaz,2));
	if(deltaz > SMALL) beta = atan(deltar/deltaz);
	else beta = (float)M_PI/2.0;

	fprintf(stderr," rmid = %f, deltar = %f, deltaz = %f, deltal = %f\n",
			rmid, deltar, deltaz, deltal);
	fprintf(stderr," Q = %f, H = %f, N = %f\n",
			inbc->bcQ,inbc->bcH,inbc->bcN);
	fprintf(fp,"# Q=%.4e, H=%.4e, N=%.4e\n",inbc->bcQ,inbc->bcH,inbc->bcN);

	// medium inlet velocities (= atan(cm0/cu0))
	// and turbulence values
	if ( (area = 2.0f*(float)M_PI*rmid*deltal) < SMALL) area = SMALL;
	cm = inbc->bcQ/area;
	inbc->cm = cm;
	if(inbc->useAlpha)
		cu = cm/tan(inbc->bcAlpha);
	else cu = -((9.81*inbc->bcH) / (M_PI/30.0*inbc->bcN*rmid));

	// inflow bc values
	// memory for values
	fprintf(stderr," cu = %f, cm = %f, alpha = %f\n",cu,cm,
			180.0/M_PI*atan(cm/cu));
	if( (tmpbcval = (float**)calloc(inlet->num,sizeof(float*))) == NULL) {
		fatal(" no memory for bcvalues!");
		return -1;
	}
	if( (tmpbcval[0] = (float*)calloc(inlet->num*5,sizeof(float))) == NULL) {
		fatal(" no memory for bcvalues!");
		return -1;
	}
	for(i = 1; i < inlet->num; i++)
		tmpbcval[i] = tmpbcval[i-1] + 5;

	// set values, see ~lippold/perl/rotein4.pl as reference.
	ilist  =  inlet->list;
	ca	   =  cos(beta);
	bccm   = -cm*sin(beta);
	curmid =  cu*rmid;
	Re	   =  cm*deltal / VISCO;
	exp	   =  getExp(Re);
	cmmax  =   cm*(1./exp+1.)*(2./exp+1.)/(2.*pow(1./exp,2.0));
	fprintf(fp,"# alpha_const=%d, turb_prof=%d, inlet->num=%d\n",
			alpha_const,turb_prof,inlet->num);
	fprintf(fp,"# deltar=%.4e, deltaz=%.4e, deltal=%.4e\n",
			deltar, deltaz, deltal);
	fprintf(fp,"# rmid=%.4e, zmid=%.4e, area=%.4e, exp=%f, Re=%.1f\n",
			rmid,zmid,area,exp,Re);
	fprintf(fp,"# cu=%f, cm=%f, alpha=%f, curmid=%f,bccm=%f\n",
			cu,cm,180.0/M_PI*atan(cm/cu),curmid,bccm);
	fprintf(fp,"# id,r,z,bccm,cu_r,cm_r,val[3](k),val[4](eps)\n");
	if(alpha_const == 1) {
		for(i = 0; i < inlet->num; i++, ilist++) {
			val	   = tmpbcval[i];
			nod	   = n->n[*ilist];
			cein   = cm;
			cunod  = cu;
			if(turb_prof) {
				dz = fabs(nod->z-zmid);
				if(deltaz > SMALL)
					tf = pow((float)fabs(1.-2.*dz/deltaz),exp);

				else
					tf = pow((float)fabs(1.-2.*fabs(nod->r-rmid)/deltar),exp);
				cein = cmmax*tf;
				cunod = cu*tf;
				bccm = -cein*sin(beta);
			}
			cm_r   = cein*ca;
			cu_r   = cunod;
			val[0] = (-nod->x*cm_r + nod->y*cu_r)/nod->r;
			val[1] = (-nod->x*cu_r - nod->y*cm_r)/nod->r;
			val[2] = bccm;
                        //crel   = sqrt(pow(val[0],2) +
                        //			  pow(val[1],2) +
                        //			  pow(val[2],2) );
			if(deltaz > SMALL) {
				val[3] = 3.75e-3 * pow(cm,2)*
					(1.0+5.0*pow(2.1*(nod->z-zmid)/deltaz,20.0));
				val[4] = MIN(9.4e-4	 * pow(cm,3) / deltal*
							 (1.0+10.0*pow(2.1*(nod->z-zmid)/deltaz,30.0)),1.e3);

			}
			else {
				val[3] = 3.75e-3 * pow(cm,2)*
					(1.0+5.0*pow(2.1*(nod->r-rmid)/deltar,20.0));
				val[4] = MIN(9.4e-4 * pow(cm,3) / deltal*
							 (1.0+10.0*pow(2.1*(nod->r-rmid)/deltar,30.0)),1.e3);
			}
			if(fp)
				fprintf(fp,"%6d %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e\n",
						nod->id,nod->r,nod->z, bccm, cu_r,cm_r,val[3],val[4]);
		}											// end i
	}
	else {
		for(i = 0; i < inlet->num; i++, ilist++) {
			val	   = tmpbcval[i];
			nod	   = n->n[*ilist];
			cein   = cm;
			cunod  = curmid/nod->r;
			if(turb_prof) {
				if(deltaz > SMALL)
					dz = fabs(nod->z-zmid);
				if(deltaz > SMALL)
					tf = pow((float)fabs(1.-2.*dz/deltaz),exp);
				else
					tf = pow((float)fabs(1.-2.*fabs(nod->r-rmid)/deltar),exp);
				cein = cmmax*tf;
				cunod = cu*tf;
				bccm = -cein*sin(beta);
			}
			cm_r   = cein*ca;
			cu_r   = cunod;
			val[0] = (-nod->x*cm_r + nod->y*cu_r)/nod->r;
			val[1] = (-nod->x*cu_r - nod->y*cm_r)/nod->r;
			val[2] = bccm;
                        //crel   = sqrt(pow(val[0],2) +
                        //			  pow(val[1],2) +
                        //			  pow(val[2],2) );
			if(deltaz > SMALL) {
				val[3] = 3.75e-3 * pow(cm,2)*
					(1.0+5.0*pow(2.1*(nod->z-zmid)/deltaz,20.0));
				val[4] = MIN(9.4e-4	 * pow(cm,3) / deltal*
							 (1.0+10.0*pow((float)2.1*(nod->z-zmid)/deltaz,30)),1.e3);

			}
			else {
				val[3] = 3.75e-3 * pow(cm,2)*
					(1.0+5.0*pow(2.1*(nod->r-rmid)/deltar,20.0));
				val[4] = MIN(9.4e-4 * pow(cm,3) / deltal*
							 (1.0+10.0*pow(2.1*(nod->r-rmid)/deltar,30.0)),1.e3);
			}
			if(fp)
				fprintf(fp,"%6d %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e %14.5e\n",
						nod->id,nod->r,nod->z, bccm, cu_r,cm_r,val[3],val[4]);
		}											// end i
	}
	*bcval = &tmpbcval[0];
	
	if(fp) fclose(fp);

	return 0;
}

#ifdef  INLET_BC2
// **************************************************
// new and nicer version of creating inlet BCs
// first get profile depending on the radius, then get values for each node
// and translate to xyz-coords.
#define NCOMPS 6  // r,phi,z,  vu,vm,vtotal
int CreateInletBCs2(struct Ilist *inlet, struct Nodelist *n,struct bc *inbc,
		    float ***bcval, int alpha_const, int turb_prof, int ge_num)
{
	int i, j, rnum, noffset;
	float *vv[NCOMPS];
	float rmax,rmin,rmid,deltar, zmax,zmin,deltaz, area, deltal;
	float beta, cm, cu, cm_r, cu_r, crel;
	float ca, curmid, bccm, **tmpbcval, *val;
	float Re, exp, cein, cmmax, cunod, zmid, dz, tf;
	
	char *fn = "inbc.dat";
	FILE *fp = NULL;
	
	if( (fp = fopen(fn,"w+")) == NULL)
		fprintf(stderr,"Could not open file '%s'!\n",fn);

	fprintf(stderr," CreateInletBCs() ... \n");

	// ****************************************
	// inits
	if(!(inlet->list)) {
		fatal("list with inlet nodes does not exist!!");
		return -1;
	}

	for(i = 0; i < NCOMPS; i++) {
		if( (vv[i] = (float*)calloc(ge_num,sizeof(float))) == NULL)
			fprintf(stdout,"\n%s (%d): no memory for %d*float!\n\n",
					__FILE__,__LINE__,ge_num);
			return -1;
	}

	noffset = inlet->num/ge_num;
	for(i = 0; i < ge_num; i++) {
		vv[0][i] = n->n[inlet->list[i*noffset]]->r;
		vv[1][i] = n->n[inlet->list[i*noffset]]->phi;
		vv[2][i] = n->n[inlet->list[i*noffset]]->z;
	}
	// ****************************************
	// intermediate radius and delta z.
	// straight inlet supposed
	rmax  = MAX(vv[0][ge_num-1], vv[0][0]);
	rmin  = MIN(vv[0][ge_num-1], vv[0][0]);
	zmax  = MAX(vv[2][ge_num-1], vv[2][0]);
	zmin  = MIN(vv[2][ge_num-1], vv[2][0]);
	rmid   = 0.5*(rmin+rmax);
	zmid   = 0.5*(zmax+zmin);
	deltar = rmax-rmin;
	deltaz = zmax-zmin;
	deltal = sqrt(pow(deltar,2)+pow(deltaz,2));
	if(deltaz > SMALL) beta = atan(deltar/deltaz);
	else beta = M_PI/2.0;
	


	// ****************************************
	// cleanup
	for(i = 0; i < NCOMPS; i++) free(vv[i]);


	return 0;
}
#undef NCOMPS
#undef SMALL

#endif // INLET_BC2

// get exponent for turbulent profile.
#define SAMPLE_PTS 4
static float getExp(float Re)
{
	int i;
	int imax = SAMPLE_PTS-1;

	float exp=0.0f;

	float r[] = {4.0e+3f, 1.0e+5f, 1.0e+6f, 3.0e+6f };
	float e[] = {0.16666f, 0.14285f, 0.11111f, 0.1f };

	if(Re >= r[imax]) return e[imax];
	else if(Re <= r[0]) return e[0];

	for(i = 1; i < SAMPLE_PTS; i++) {
		if(Re < r[i]) {
			exp = (e[i-1]-e[i])/(r[i-1]-r[i])*(Re-r[i]) + e[i];
			break;
		}
	}
	return exp;
}


#undef SAMPLE_PTS
#undef VISCO

#ifdef PATRAN_SES_OUT
// writes .ses file for PATRAN for each meridional plane (elems)
int WritePATRAN_SESfile(int nnum, int elnum, int ge_num, int nstart,
						int elstart, const char *efile, const char *nfile,
						const char *egroup, const char *ngroup)
{
	int i;
	int nstep, elstep;

	char fn[200], meridian[100];
	FILE *fp = NULL;

	nstep  = nnum / ge_num;
	elstep = elnum / (ge_num - 1);

	// session files for elements + nodes
	for(i = 0; i < ge_num-1; i++) {
		sprintf(fn,"%s_%02d.ses",efile,i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr," Can not open file '%s'!\n",fn);
			exit(-1);
		}
		sprintf(meridian,"%s%02d",egroup,i);
		fprintf(fp,"uil_list_a.clear()\n");
		fprintf(fp,"list_create_target_list(\"lista\",\"elm %d:%d\")\n",
				elstart+1+i*elstep, elstart+(i+1)*elstep);
		fprintf(fp,"list_create_target_list(\"lista\",\"node %d:%d\")\n",
				nstart+1+i*nstep, nstart+(i+2)*nstep);
		fprintf(fp,"list_save_group(\"lista\",\"%s\",FALSE)\n", meridian);
		fprintf(fp,"uil_viewport_post_groups.posted_groups(\"default_viewport\",1,[\"%s\"])\n",meridian);
		fprintf(fp,"gu_fit_view()\n");
		fclose(fp);
	}

	// session files, one meridian plane's nodes
	for(i = 0; i < ge_num; i++) {
		sprintf(fn,"%s_%02d.ses",nfile, i);
		if( (fp = fopen(fn,"w+")) == NULL) {
			fprintf(stderr," Can not open file '%s'!\n",fn);
			exit(-1);
		}
		sprintf(meridian,"%s%02d",ngroup, i);
		fprintf(fp,"uil_list_a.clear()\n");
		fprintf(fp,"list_create_target_list(\"lista\",\"node %d:%d\")\n",
				nstart+1+i*nstep, nstart+(i+1)*nstep);
		fprintf(fp,"list_save_group(\"lista\",\"%s\",FALSE)\n", meridian);
		fprintf(fp,"uil_viewport_post_groups.posted_groups(\"default_viewport\",1,[\"%s\"])\n",meridian);
		fprintf(fp,"gu_fit_view()\n");
		fclose(fp);
	}

	return 0;
}
#endif
