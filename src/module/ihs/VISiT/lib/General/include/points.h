#ifndef  POINT_H_INCLUDED
#define  POINT_H_INCLUDED

#include <stdio.h>

struct Point
{
   int nump;
   int maxp;
   int portion;
   float *x;
   float *y;
   float *z;
};

struct Point *AllocPointStruct(void);
float *GetPoint(struct Point *p, float r[3], int ind);
int GetPointIndex(int num, float *fpara, float par, int istart);
int AddVPoint(struct Point *p, float P[3]);
void FreePointStruct(struct Point *p);
int AddPoint(struct Point *p, float x, float y, float z);
struct Point *GetPointMemory(struct Point *p);
struct Point *CopyPointStruct(struct Point *src);
struct Point *nCopyPointStruct(struct Point *src, int srcnum);
struct Point *nCopynPointStruct(struct Point *src, int istart, int srcnum);
void DumpPoints(struct Point *p, FILE *fp);
#endif                                            // POINT_H_INCLUDED
