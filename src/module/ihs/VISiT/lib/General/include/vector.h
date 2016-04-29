#ifndef  VECTOR_H_INCLUDED
#define  VECTOR_H_INCLUDED

struct Vector
{
   int numv;
   int maxv;
   int len;
   int **v;
};

struct Vector * AllocVectorStruct(int vlen);
int AddVector(struct Vector *v, int *vec);
void FreeVectorStruct(struct Vector *v);
#ifdef   DEBUG
#endif                                            // DEBUG
#endif                                            // VECTOR_H_INCLUDED
