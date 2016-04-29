#include <string.h>
#include <stdlib.h>
#include <string.h>
#include "include/vector.h"
#include "include/fatal.h"

struct Vector * AllocVectorStruct(int vlen)
{
   struct Vector *v;

   if ((v = (struct Vector *)(calloc(1, sizeof(struct Vector)))) != NULL)
   {
      v->len = vlen;
   }
   return v;
}


int AddVector(struct Vector *v, int vec[])
{
   int i;

   if ((v->numv+1) >= v->maxv)
   {
      v->maxv += 100;
      if ((v->v = (int **)realloc(v->v, v->maxv*sizeof(int *))) == NULL)
         fatal("Space in AddVector(): v->v");
      for (i = v->maxv-100; i < v->maxv; i++)
         if ((v->v[i] = (int *)(calloc(v->len, sizeof(int)))) == NULL)
            fatal("Space in AddVector(): v->v[i]");
   }
   memcpy(v->v[v->numv], vec, v->len*sizeof(int));
   v->numv++;
   return (v->numv-1);
}


void FreeVectorStruct(struct Vector *v)
{
   int i;
   for (i = 0; i < v->maxv; i++)
      free(v->v[i]);
   free(v->v);
   free(v);
}
