/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "include/points.h"
#include "include/log.h"
#include "include/fatal.h"

#define POINT "point%d"

float *GetPoint(struct Point *p, float r[3], int ind)
{
    if (ind < p->nump)
    {
        r[0] = p->x[ind];
        r[1] = p->y[ind];
        r[2] = p->z[ind];
        return r;
    }
    return NULL;
}

// returns point index or -1 if no point found.
int GetPointIndex(int num, float *fpara, float par, int istart)
{
    int i;

    num--;
    for (i = istart; i < num; i++)
    {
        if (fpara[i + 1] >= par)
        {
            return (i);
        }
    }
    return (-1);
}

struct Point *AllocPointStruct(void)
{
    struct Point *p;

    if ((p = (struct Point *)calloc(1, sizeof(struct Point))) == NULL)
        fatal("memory for (struct Point *)");
    p->portion = 100;
    return (p);
}

int AddVPoint(struct Point *p, float P[3])
{
    return AddPoint(p, P[0], P[1], P[2]);
}

int AddPoint(struct Point *p, float x, float y, float z)
{
    if ((p->nump + 1) >= p->maxp)
    {
        p->maxp += p->portion;
        if ((p->x = (float *)realloc(p->x, p->maxp * sizeof(float))) == NULL)
            fatal("Space in AddPoint(): p->x");
        if ((p->y = (float *)realloc(p->y, p->maxp * sizeof(float))) == NULL)
            fatal("Space in AddPoint(): p->y");
        if ((p->z = (float *)realloc(p->z, p->maxp * sizeof(float))) == NULL)
            fatal("Space in AddPoint(): p->z");
    }
    p->x[p->nump] = x;
    p->y[p->nump] = y;
    p->z[p->nump] = z;
    p->nump++;
    return (p->nump - 1);
}

void NormPointStruct(struct Point *p)
{
    int i;
    float mag;

    for (i = 0; i < p->nump; i++)
    {
        mag = sqrt(pow(p->x[i], 2) + pow(p->y[i], 2) + pow(p->z[i], 2));
        if (mag > 0.0)
        {
            p->x[i] /= mag;
            p->y[i] /= mag;
            p->z[i] /= mag;
        }
    }
}

struct Point *GetPointMemory(struct Point *p)
{
    if (p)
    {
        FreePointStruct(p);
        p = NULL;
    }
    return (AllocPointStruct());
}

struct Point *CopyPointStruct(struct Point *src)
{
    struct Point *p;
    p = AllocPointStruct();
    p->nump = src->nump;
    p->maxp = src->maxp;
    if ((p->x = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->x");
    if ((p->y = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->y");
    if ((p->z = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->z");
    memcpy(p->x, src->x, src->nump * sizeof(float));
    memcpy(p->y, src->y, src->nump * sizeof(float));
    memcpy(p->z, src->z, src->nump * sizeof(float));

    return p;
}

struct Point *nCopyPointStruct(struct Point *src, int srcnum)
{
    struct Point *p;
    p = AllocPointStruct();
    p->nump = srcnum;
    p->maxp = srcnum;
    if ((p->x = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->x");
    if ((p->y = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->y");
    if ((p->z = (float *)calloc(p->maxp, sizeof(float))) == NULL)
        fatal("Space in CopyPointStruct: p->z");
    memcpy(p->x, src->x, srcnum * sizeof(float));
    memcpy(p->y, src->y, srcnum * sizeof(float));
    memcpy(p->z, src->z, srcnum * sizeof(float));

    return p;
}

void FreePointStruct(struct Point *p)
{
    if (p)
    {
        if (p->nump && p->x)
            free(p->x);
        if (p->nump && p->y)
            free(p->y);
        if (p->nump && p->z)
            free(p->z);
        free(p);
    }
}

void DumpPoints(struct Point *p, FILE *fp)
{
    int j;

    dprintf(5, "nump = %d\n", p->nump);

    for (j = 0; j < p->nump; j++)
    {
        if (fp)
            fprintf(fp, "%10.6f %10.6f %10.6f\n", p->x[j], p->y[j], p->z[j]);
        dprintf(5, "  j=%3d: x=%10.6f y=%10.6f z=%10.6f\n", j, p->x[j],
                p->y[j], p->z[j]);
    }
}
