#ifdef GRID4COV

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#include "../General/include/nodes.h"
#include "../General/include/ptlist.h"
#include "../General/include/ilist.h"
#include "include/rr_grid.h"
#include "../General/include/log.h"
#include "../General/include/common.h"

// **************************************************
int CreateRR_Grid4Covise(struct Nodelist *n, float *x, float *y, float *z,
						 int istart, int iend)
{
   int i;
   float *xp, *yp, *zp;

   struct node **nod;

   xp = x; yp = y; zp = z;
   nod = n->n+istart;
   for(i = istart; i < iend; i++,nod++, xp++,yp++,zp++) {
      (*xp) = (*nod)->x;
      (*yp) = (*nod)->y;
      (*zp) = (*nod)->z;
   }
   return 0;
}

// **************************************************
// append surface elements to a list (coDoPolygons)
// caller: XXXRunner::CreateGrid
int CreateRR_BClist4Covise(struct Element *e, int *corners, int *poly_list,
						   int *i_corn,int *i_poly,
						   int num_corn, int num_poly, char *buf)
{
	int i, j, **elnod, ip, ip0;
	int *c, *p, *cc;

	int npe4[] = {1,2,4,3}; // nodes are ordered x-over!!

	c   = &corners[*i_corn];
	p   = &poly_list[*i_poly];

	elnod = e->e;

	ip = ip0 = 0;
	if(*i_poly > 0) ip0  = poly_list[(*i_poly)-1]+NPE_BC;

	// init connectivity pointer
	switch(NPE_BC) {
		case 4:
			cc = npe4;
			break;
		default:
			sprintf(buf," CreateRR_BClist4Covise(): Case npe = %d not implemented!\n",NPE_BC);
			return 1;
			break;
	}

	// set start pointer for next element set
	// makes appending of some more sets possible
	if( (*i_corn += NPE_BC * e->nume) > num_corn) {
		sprintf(buf," CreateRR_BClist4Covise(): Too many corners!\n");
		return 1;
	}
	if( (*i_poly += e->nume) > num_poly) {
		sprintf(buf," CreateRR_BClist4Covise(): Too many polygons!\n");
		return 1;
	}

	// append/set data
	for(i = 0; i < e->nume; i++, elnod++) {
		for(j = 0; j < NPE_BC; j++) {
			*(c++) = (*elnod)[cc[j]];
		}
		*(p++) = ip+ip0;
		ip    += NPE_BC;
	}


	return 0;
}
// **************************************************
int CreateRR_BClistbyElemset(struct Nodelist *n,
							 struct Element **e, int num,
							 float *xc, float *yc, float *zc,
							 int *corners, int *poly_list,
							 int num_corn, int num_poly, int num_node,
							 char *buf)
{
	int i, j, k, id, ip, nix, nid;
	int maxid;
	int *ix;
	int *c, *p;
	int npe4[] = {1,2,4,3}; // nodes are ordered x-over!!
	float *x, *y, *z;

	x = xc; y = yc; z = zc;
	c = corners;
	p = poly_list;


	// get maxid
	maxid = 0;
	for(i = 0; i < num; i++) {
		for(j = 0; j < e[i]->nume; j++) {
			for(k = 1; k <= NPE_BC; k++) {
				maxid = MAX(e[i]->e[j][k], maxid);
			}
		}
	}
	if( (ix = (int *)calloc(maxid+1,sizeof(int))) == NULL) {
		sprintf(buf," CreateRR_BClistbyElemset(): no memory!\n");
		return 1;
	}
	memset(ix,-1,(maxid+1)*sizeof(int));

	// get inverse pointer and elements
	id = 0;
	ip = 0;
	for(i = 0; i < num; i++) {
		for(j = 0; j < e[i]->nume; j++) {
			for(k = 0; k < NPE_BC; k++) {
				nix = e[i]->e[j][npe4[k]];
				// check for new node and set pointer
				if(ix[nix] == -1) {
					ix[nix] = id++;
					if( (id > num_node) ) {
						sprintf(buf," CreateRR_BClistbyElemset(): too many nodes!\n");
						free(ix);
						return 1;
					}
					*(x++) = n->n[nix]->x;
					*(y++) = n->n[nix]->y;
					*(z++) = n->n[nix]->z;
				}
				// get pointer and add node to corner list
				if( (nid = ix[nix]) > num_corn) {
					sprintf(buf," CreateRR_BClistbyElemset(): too many corners!\n");
					free(ix);
					return 1;
				}
				*(c++) = nid;
			}
			// increase polygon list
			*(p++) = ip;
			ip    += NPE_BC;
			if( (ip/NPE_BC) > num_poly) {
				sprintf(buf," CreateRR_BClistbyElemset(): too many polygons!\n");
				free(ix);
				return 1;
			}
		}
	}

	free(ix);

	return 0;
}

struct ptlist *elist;
// **************************************************
void **Create_ElemSet(int *num, ...)
{
	va_list ap;
	struct Element *ee;

	va_start(ap,num);

	elist = AllocPtListStruct();
	while( (ee = va_arg(ap, struct Element *) )) {
		Add2PtList(elist,ee);
	}

	va_end(ap);
	*num = elist->num;
	return elist->pt;
}
void FreeElemSetPtr(void)
{
	FreePtListStruct(elist);
}

// **************************************************
int SetElement(int **data, struct Element *e, int flag)
{
	int i, *dd;
	int **elem;

	elem = e->e;

	dd = *data;

	dprintf(2," SetElement(): Entering ...\n");
	// elements are x-over (3 <-> 4)
	for (i = 0; i < e->nume; i++, elem++) {
		*(dd++) = (*elem)[1]+1;
		*(dd++) = (*elem)[2]+1;
		*(dd++) = (*elem)[4]+1;
		*(dd++) = (*elem)[3]+1;
		*(dd++) = (*elem)[0]+1; // volume element
		dprintf(4," SetElement(): i = %7d: %8d %8d %8d %8d %8d(%8d)\n",
				i,*(dd-5),*(dd-4),*(dd-3),*(dd-2),*(dd-1),
				(*elem)[0]+1);
		*(dd++) = flag;
		*(dd++) = 0;               // dummy number
	}

	*data = &dd[0];

	return 0;
}
#endif                                            // GRID4COV
