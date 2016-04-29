#include <stdlib.h>
#include <string.h>
#include "include/elements.h"
#include "include/fatal.h"

int *GetElement(struct Element *e, int r[8], int ind)
{
	int i;
	if (ind < e->nume) {
		for (i = 0; i < 8; i++) {
			r[i] = e->e[ind][i];
		}
		return r;
	}
	return NULL;
}


struct Element * AllocElementStruct(void)
{
	return (struct Element *)(calloc(1, sizeof(struct Element)));
}


int AddElement(struct Element *e, int elem[8])
{
	int i;
	if ((e->nume+1) >= e->maxe) {
		e->maxe += 100;
		if ((e->e = (int **)realloc(e->e, e->maxe*sizeof(int *))) == NULL)
			fatal("Space in AddElement(): e->e");
		for (i = e->maxe-100; i < e->maxe; i++)
			if ((e->e[i] = (int *)(calloc(8, sizeof(int)))) == NULL)
				fatal("Space in AddElement(): e->e[i]");
	}
	memcpy(e->e[e->nume], elem, 8*sizeof(int));
	e->nume++;
	return (e->nume-1);
}


void FreeElementStruct(struct Element *e)
{
	int i;
	if(e) {
		for (i = 0; i < e->maxe; i++)
			free(e->e[i]);
		free(e->e);
		free(e);
	}
}
