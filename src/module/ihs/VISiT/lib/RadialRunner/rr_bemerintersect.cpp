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
#include "../General/include/geo.h"

#ifdef RADIAL_RUNNER
#include "include/radial.h"
#endif
#ifdef DIAGONAL_RUNNER
#include "include/diagonal.h"
#endif
#include "../General/include/points.h"
#include "../General/include/curve.h"
#include "../General/include/flist.h"
#include "../General/include/plane_geo.h"
#include "../General/include/parameter.h"
#include "../General/include/profile.h"
#include "../General/include/bias.h"
#include "../General/include/curvepoly.h"
#include "../BSpline/include/bspline.h"
#include "../General/include/common.h"
#include "../General/include/log.h"
#include "../General/include/fatal.h"
#include "../General/include/v.h"

int BladeEdgeMeridianIntersection(struct edge *e, struct be **be, int be_num)
{
	int i, j, k, ii;
	int found = 0;				// flag if segment is found
	int next_edge_segment = 0;	// flag if segment has no intersection
	int jstart = 0;				// meridian curve to start next search
	int kfound = 0;				// number of last found segment
	int depth = 2;				// search depth in one direction
	int upper, lower;			// upper and lower search start bounds
	float b[3], bvec[3];		// edge segment start point and vector
	float bmin, bmax;			// min/max y coord of edge segment
	float m[3], mvec[3];		// meridian segment start point and vector
	float mmin, mmax;			// min/max y coord of meridian segment
	float inter[3];				// segment intersection
	float dvec1[3], dvec2[3];	// edge start point to meridian start point vectors
	float vprod1, vprod2;		// vector product z coordinate
	float par, dz, dr;				// parameter value of found intersection
#ifdef DEBUG_INTERSECTION
	FILE *fp, *fm;
	char fname[255];
	int nedge = 0;
	static int ncall = 0;
#endif	// DEBUG_INTERSECTION

#ifdef GAP
	be_num++;
#endif

	// memory for intersection data
	if (e->bmint) {
		FreePointStruct(e->bmint);
		e->bmint = NULL;
	}
	e->bmint = AllocPointStruct();
	if (e->bmpar) {
		FreeFlistStruct(e->bmpar);
		e->bmpar = NULL;
	}
	e->bmpar = AllocFlistStruct(be_num);
#ifdef DEBUG_INTERSECTION
	sprintf(fname, "rr_bemerint_%02d.txt", ncall++);
	if ((fm = fopen(fname, "w")) == NULL)
		fprintf(stderr, "cannot open file %s\n", fname);
#endif	// DEBUG_INTERSECTION

	// search all blade edge contour points
#ifdef NO_INLET_EXT
	kfound = (int)((NPOIN_MERIDIAN) * e->para[0]);
#else //NO_INLET_EXT
	kfound = (int)((NPOIN_MERIDIAN) * e->para[0] + NPOIN_EXT - 1);
#endif

	// determine intersection coordinate
	dr = float(fabs(e->c->p->x[0] - e->c->p->x[e->c->p->nump-1]));
	dz = float(fabs(e->c->p->z[0] - e->c->p->z[e->c->p->nump-1]));
	if(dr < dz) ii = 0;	 // use radius for intersection
	else ii = 2;

	for (i = 0; i < NPOIN_EDGE; i++) {
#ifdef DEBUG_INTERSECTION
		sprintf(fname, "rr_intersect%02d_%02d.txt", nedge++, (ncall-1));
		if ((fp = fopen(fname, "w")) == NULL)
			fprintf(stderr, "cannot open file %s\n", fname);
		fprintf(fp, "\nBLADE EDGE SEGMENT %d\n", i);
#endif	// DEBUG_INTERSECTION
		found = 0;
		next_edge_segment = 0;

		

		// blade edge segment vector
		b[0]	= e->c->p->x[i];
		b[1]	= e->c->p->y[i];
		b[2]	= e->c->p->z[i];
		bmin = bmax = b[ii];
		bvec[0] = e->c->p->x[i+1] - b[0];
		bvec[1] = e->c->p->y[i+1] - b[1];
		bvec[2] = e->c->p->z[i+1] - b[2];
		((b[ii] + bvec[ii]) > bmin) ? (bmax = b[ii] + bvec[ii]) : (bmin = b[ii] + bvec[ii]);
#ifdef DEBUG_INTERSECTION
		fprintf(fp," b = [%f  %f  %f]\n",b[0], b[1], b[2]);
#endif

		//search all meridian contour curves
		for (j = jstart; j < be_num; j++) {
#ifdef DEBUG_INTERSECTION
			fprintf(fp, "\nMERIDIAN CURVE %d\n", j);
			fprintf(fp, "jstart = %d\n", jstart);
#endif	// DEBUG_INTERSECTION
			found = 0;
			upper = lower = kfound;

			// search meridian contour curve until
			//	a) intersection is found, or
			//	b) sign change of vector product (no intersection)
			while (!next_edge_segment && !found) {

				// search ascending
				for (k = upper; k < (upper+depth); k++) {
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "##################################################\n");
					fprintf(fp, "# ascending search:\n");
					fprintf(fp, "#	kfound = %d\t", kfound);
					fprintf(fp, "kstart/kend = %d/%d\t", upper, (upper+depth));
					fprintf(fp, "k = %d\n", k);
					fprintf(fp, "##################################################\n\n");
#endif	// DEBUG_INTERSECTION
					// meridian segment vector
					m[0]	= be[j]->ml->p->x[k];
					m[1]	= be[j]->ml->p->y[k];
					m[2]	= be[j]->ml->p->z[k];
					mmin = mmax = m[ii];
					mvec[0] = be[j]->ml->p->x[k+1] - m[0];
					mvec[1] = be[j]->ml->p->y[k+1] - m[1];
					mvec[2] = be[j]->ml->p->z[k+1] - m[2];
					((m[ii] + mvec[ii]) > mmin) ? (mmax = m[ii] + mvec[ii]) : (mmin = m[ii] + mvec[ii]);
					mmax +=1.e-6f; mmin -= 1.e-6f;
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "edge segment i=%d\n", i);
					fprintf(fp, " b(i)		: %8.6f, %8.6f, %8.6f\n", b[0], b[1], b[2]);
					fprintf(fp, " b(i+1)	: %8.6f, %8.6f, %8.6f\n", b[0]+bvec[0], b[1]+bvec[1], b[2]+bvec[2]);
					fprintf(fp, " bmin, bmax: %8.6f, %8.6f\n", bmin, bmax);
					fprintf(fp, "meridian segment k=%d\n", k);
					fprintf(fp, " m(k)		: %8.6f, %8.6f\n", m[0], m[2]);
					fprintf(fp, " m(k+1)	: %8.6f, %8.6f\n", m[0]+mvec[0], m[2]+mvec[2]);
					fprintf(fp, " mmin, mmax: %8.6f, %8.6f\n", mmin, mmax);
#endif	// DEBUG_INTERSECTION
					// intersection and location check on segments
					LineIntersectXZ(b, bvec, m, mvec, inter);
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "intersection\n");
					fprintf(fp, " inter		: %8.6f, %8.6f, %8.6f\n", inter[0], inter[1], inter[2]);
#endif	// DEBUG_INTERSECTION
					if ((inter[ii] >= bmin) && (inter[ii] <= bmax) &&
						(inter[ii] >= mmin) && (inter[ii] <= mmax)) {
						found  = 1;
						AddVPoint(e->bmint, inter);
						if(fabs(dz = be[j]->ml->p->z[k+1] - be[j]->ml->p->z[k]) > 1.e-6) {
							par	 = (be[j]->ml->par[k+1] - be[j]->ml->par[k]);
							par /= (dz);
							par *= (inter[2] - be[j]->ml->p->z[k]);
							par += be[j]->ml->par[k];
						}
						else {
							par	 = (be[j]->ml->par[k+1] - be[j]->ml->par[k]);
							par /= (be[j]->ml->p->x[k+1] - be[j]->ml->p->x[k]);
							par *= (inter[0] - be[j]->ml->p->x[k]);
							par += be[j]->ml->par[k];
						}
						Add2Flist(e->bmpar, par);
						upper = lower = kfound = k;
						jstart++;
#ifdef DEBUG_INTERSECTION
						fprintf(fm, "%8.6f %8.6f %8.6f %8.6f\n", e->bmint->x[j], e->bmint->y[j], e->bmint->z[j], par);
						fprintf(fp, " par = %8.6f\n", par);
#else
						break;
#endif	// DEBUG_INTERSECTION
					}
#ifdef DEBUG_INTERSECTION
					fprintf(fp, " FOUND = %d\n\n", found);
					if (found) break;
#endif	// DEBUG_INTERSECTION
					// vectors to meridian points from blade segment start point
					dvec1[0] = m[0] - b[0];
					dvec1[1] = m[1] - b[1];
					dvec1[2] = m[2] - b[2];
					dvec2[0] = m[0] + mvec[0] - b[0];
					dvec2[1] = m[1] + mvec[1] - b[1];
					dvec2[2] = m[2] + mvec[2] - b[2];
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "edge meridian vectors:\n");
					fprintf(fp, " dvec1: %8.6f %8.6f %8.6f\n", dvec1[0], dvec1[1], dvec1[2]);
					fprintf(fp, " dvec2: %8.6f %8.6f %8.6f\n", dvec2[0], dvec2[1], dvec2[2]);
#endif	// DEBUG_INTERSECTION
					// vector products with blade segment vector
					vprod1	 = bvec[0] * dvec1[2] - bvec[2] * dvec1[0];
					vprod2	 = bvec[0] * dvec2[2] - bvec[2] * dvec2[0];
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "vector product (z coord):\n");
					fprintf(fp, " vprod1 = %8.6f\n vprod2 = %8.6f\n", vprod1, vprod2);
#endif	// DEBUG_INTERSECTION
					if ((vprod1 * vprod2) < 0.0) {
						next_edge_segment = 1;
#ifndef DEBUG_INTERSECTION
						break;
#endif
					}
#ifdef DEBUG_INTERSECTION
					
					fprintf(fp, " NEXT_EDGE_SEGMENT = %d\n\n", next_edge_segment);
					//??????????????????????????????????????????????
#endif	// DEBUG_INTERSECTION
					//??????????????????????????????????????????????
					if (next_edge_segment)	break;
				}
				if (next_edge_segment || found) break;

#ifdef DEBUG_INTERSECTION
				fprintf(fp, "\n\n");
#endif	// DEBUG_INTERSECTION
		// search descending
				for (k = (lower-1); k > (lower-1-depth); k--) {
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "##################################################\n");
					fprintf(fp, "# descending search:\n");
					fprintf(fp, "#	kfound = %d\t", kfound);
					fprintf(fp, "kstart/kend = %d/%d\t", (lower-1), (lower-1-depth));
					fprintf(fp, "k = %d\n", k);
					fprintf(fp, "##################################################\n\n");
#endif	// DEBUG_INTERSECTION
					// meridian segment vector
					m[0]	= be[j]->ml->p->x[k];
					m[1]	= be[j]->ml->p->y[k];
					m[2]	= be[j]->ml->p->z[k];
					mmin = mmax = m[ii];
					mvec[0] = be[j]->ml->p->x[k+1] - m[0];
					mvec[1] = be[j]->ml->p->y[k+1] - m[1];
					mvec[2] = be[j]->ml->p->z[k+1] - m[2];
					((m[ii] + mvec[ii]) > mmin) ? (mmax = m[ii] + mvec[ii]) : (mmin = m[ii] + mvec[ii]);
					mmax +=1.e-6f; mmin -= 1.e-6f;
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "edge segment i=%d\n", i);
					fprintf(fp, " b(i)		: %8.6f, %8.6f, %8.6f\n", b[0], b[1], b[2]);
					fprintf(fp, " b(i+1)	: %8.6f, %8.6f, %8.6f\n", b[0]+bvec[0], b[1]+bvec[1], b[2]+bvec[2]);
					fprintf(fp, " bmin, bmax: %8.6f, %8.6f\n", bmin, bmax);
					fprintf(fp, "meridian segment k=%d\n", k);
					fprintf(fp, " m(k)		: %8.6f, %8.6f, %8.6f\n", m[0], m[1], m[2]);
					fprintf(fp, " m(k+1)	: %8.6f, %8.6f, %8.6f\n", m[0]+mvec[0], m[1]+mvec[1], m[2]+mvec[2]);
					fprintf(fp, " mmin, mmax: %8.6f, %8.6f\n", mmin, mmax);
#endif	// DEBUG_INTERSECTION
					// intersection and location check on segments
					LineIntersectXZ(b, bvec, m, mvec, inter);
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "intersection\n");
					fprintf(fp, " inter		: %8.6f, %8.6f, %8.6f\n", inter[0], inter[1], inter[2]);
#endif	// DEBUG_INTERSECTION
					if ((inter[ii] >= bmin) && (inter[ii] <= bmax) &&
						(inter[ii] >= mmin) && (inter[ii] <= mmax)) {
						found  = 1;
						AddVPoint(e->bmint, inter);
						if(fabs(dz = be[j]->ml->p->z[k+1] - be[j]->ml->p->z[k]) > 1.e-6) {
							par	 = (be[j]->ml->par[k+1] - be[j]->ml->par[k]);
							par /= (dz);
							par *= (inter[2] - be[j]->ml->p->z[k]);
							par += be[j]->ml->par[k];
						}
						else {
							par	 = (be[j]->ml->par[k+1] - be[j]->ml->par[k]);
							par /= (be[j]->ml->p->x[k+1] - be[j]->ml->p->x[k]);
							par *= (inter[0] - be[j]->ml->p->x[k]);
							par += be[j]->ml->par[k];
						}
						Add2Flist(e->bmpar, par);
						upper = lower = kfound = k;
						jstart++;
#ifdef DEBUG_INTERSECTION
						fprintf(fm, "%8.6f %8.6f %8.6f\n", e->bmint->x[j], e->bmint->y[j], e->bmint->z[j]);
						fprintf(fp, " par = %8.6f\n", par);
#else
						break;
#endif	// DEBUG_INTERSECTION
					}
#ifdef DEBUG_INTERSECTION
					fprintf(fp, " FOUND = %d\n\n", found);
					if (found) break;
#endif	// DEBUG_INTERSECTION
					// vectors to meridian points from blade segment start point
					dvec1[0] = m[0] - b[0];
					dvec1[1] = m[1] - b[1];
					dvec1[2] = m[2] - b[2];
					dvec2[0] = m[0] + mvec[0] - b[0];
					dvec2[1] = m[1] + mvec[1] - b[1];
					dvec2[2] = m[2] + mvec[2] - b[2];
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "edge meridian vectors:\n");
					fprintf(fp, " dvec1: %8.6f %8.6f %8.6f\n", dvec1[0], dvec1[1], dvec1[2]);
					fprintf(fp, " dvec2: %8.6f %8.6f %8.6f\n", dvec2[0], dvec2[1], dvec2[2]);
#endif	// DEBUG_INTERSECTION
					// vector products with blade segment vector
					vprod1	 = bvec[0] * dvec1[2] - bvec[2] * dvec1[0];
					vprod2	 = bvec[0] * dvec2[2] - bvec[2] * dvec2[0];
#ifdef DEBUG_INTERSECTION
					fprintf(fp, "vector product (z coord):\n");
					fprintf(fp, " vprod1 = %8.6f\n vprod2 = %8.6f\n", vprod1, vprod2);
#endif	// DEBUG_INTERSECTION
					if ((vprod1 * vprod2) < 0.0) {
						next_edge_segment = 1;
#ifdef DEBUG_INTERSECTION
						fprintf(fp, " NEXT_EDGE_SEGMENT = %d\n\n", next_edge_segment);
#endif	// DEBUG_INTERSECTION
						break;
					}
				}
				if (next_edge_segment)	break;
				// restart search with shifted upper/lower start point, same i,j
				if (!found) {
					upper += depth;
					lower -= depth;
				}
			}
			if (next_edge_segment)
				break;
		}
#ifdef DEBUG_INTERSECTION
		fclose(fp);
#endif	// DEBUG_INTERSECTION
	}
#ifdef DEBUG_INTERSECTION
	fclose(fm);
#endif	// DEBUG_INTERSECTION
	return 0;
}
