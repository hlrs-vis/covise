#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "include/ptlist.h"
#include "include/log.h"
#include "include/fatal.h"

struct ptlist *AllocPtListStruct(void)
{
	char buf[222];
	struct ptlist *ptlist;

	if( (ptlist = (struct ptlist*)calloc(1,sizeof(struct ptlist))) == NULL) {
		sprintf(buf,"No memory for sizeof(struct ptlist)!");
		fatal(buf);
	}
	ptlist->portion = 10;
	return ptlist;
}

void FreePtListStruct(struct ptlist *ptlist)
{
	if(ptlist) {
		if(ptlist->num > 0) free(ptlist->pt);
		free(ptlist);
	}
}

int Add2PtList(struct ptlist *ptlist, void *pt)
{
	char buf[222];

	if(ptlist->num == ptlist->max) {
		ptlist->max += ptlist->portion;
		if( (ptlist->pt = (void **)realloc(ptlist->pt, 
									 ptlist->max*sizeof(void*))) == NULL) {
			sprintf(buf,"Memory for %d*sizeof(void*)",ptlist->max);
			fatal(buf);
		}
	}
	ptlist->pt[ptlist->num] = pt;
	ptlist->num++;
	return ptlist->num-1;
}
