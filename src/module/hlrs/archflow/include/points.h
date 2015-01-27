/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef POINT_H_INCLUDED
#define POINT_H_INCLUDED

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
void DumpPoints(struct Point *p, FILE *fp);
#endif // POINT_H_INCLUDED
