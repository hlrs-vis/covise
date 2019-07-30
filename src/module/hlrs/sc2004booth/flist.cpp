/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/flist.h"
#include "include/fatal.h"
#include "include/log.h"

void Add2Flist(struct Flist *flist, float ind)
{
    char buf[200];

    if ((flist->num + 1) >= flist->max)
    {
        flist->max += flist->portion;
        if ((flist->list = (float *)realloc(flist->list, flist->max * sizeof(float))) == NULL)
        {
            sprintf(buf, "Realloc failed in Add2Flist: %s, %d\n", __FILE__, __LINE__);
            fatal(buf);
        }
    }
    flist->list[(flist->num)++] = ind;
}

struct Flist *AllocFlistStruct(int portion)
{
    struct Flist *flist = NULL;

    if ((flist = (struct Flist *)calloc(1, sizeof(struct Flist))) == NULL)
    {
        fatal("memory for (struct Flist!");
    }
    if (portion > 0 && portion < 1024 * 1024)
        flist->portion = portion;
    else
        flist->portion = 100;
    return flist;
}

void FreeFlistStruct(struct Flist *flist)
{
    char buf[128];

    if (flist)
    {
        if (flist->list)
        {
            free(flist->list);
        }
        free(flist);
    }
    else
    {
        sprintf(buf, "Free on NULL: %s, %d\n", __FILE__, __LINE__);
        fatal(buf);
    }
}

struct Flist *CopyFlistStruct(struct Flist *src)
{
    struct Flist *flist = NULL;

    flist = AllocFlistStruct(src->portion);
    flist->num = src->num;
    flist->max = src->max;
    if ((flist->list = (float *)calloc(flist->max, sizeof(float))) == NULL)
    {
        fatal("calloc failed on (float)!");
    }
    memcpy(flist->list, src->list, src->max * sizeof(float));

    return flist;
}

struct Flist *nCopyFlistStruct(struct Flist *src, int srcnum)
{
    struct Flist *flist = NULL;

    flist = AllocFlistStruct(src->portion);
    flist->num = srcnum;
    flist->max = srcnum;
    if ((flist->list = (float *)calloc(flist->max, sizeof(float))) == NULL)
    {
        fatal("calloc failed on (float)!");
    }
    memcpy(flist->list, src->list, srcnum * sizeof(float));

    return flist;
}

void DumpFlist(struct Flist *flist)
{
    int i, lc;

    dprintf(5, "flist->num = %d\t", flist->num);
    dprintf(5, "flist->max = %d\t", flist->max);
    dprintf(5, "flist->portion = %d\n", flist->portion);
    for (lc = 0, i = 0; i < flist->num; i++)
    {
        dprintf(5, "  flist->list[%4d] = %7.4f |", i, flist->list[i]);
        if (++lc == 3)
        {
            dprintf(5, "\n");
            lc = 0;
        }
    }
    if (lc)
        dprintf(5, "\n");
}

void DumpFlist2File(struct Flist *flist, FILE *fp)
{
    int i, lc;

    fprintf(fp, "flist->num = %d\t", flist->num);
    fprintf(fp, "flist->max = %d\t", flist->max);
    fprintf(fp, "flist->portion = %d\n", flist->portion);
    for (lc = 0, i = 0; i < flist->num; i++)
    {
        fprintf(fp, "  flist->list[%4d] = %7.4f |", i, flist->list[i]);
        if (++lc == 3)
        {
            fputs("\n", fp);
            lc = 0;
        }
    }
    if (lc)
        fputs("\n", fp);
}
