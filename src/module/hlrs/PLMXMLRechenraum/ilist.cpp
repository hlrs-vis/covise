/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/log.h"
#include "include/ilist.h"
#include "include/fatal.h"

void Add2Ilist(struct Ilist *ilist, int ind)
{
    char buf[200];

    if ((ilist->num + 1) >= ilist->max)
    {
        ilist->max += ilist->portion;
        if ((ilist->list = (int *)realloc(ilist->list, ilist->max * sizeof(int))) == NULL)
        {
            sprintf(buf, "Realloc failed in Add2Ilist: %s, %d\n", __FILE__, __LINE__);
            fatal(buf);
        }
    }
    ilist->list[(ilist->num)++] = ind;
}

struct Ilist *AllocIlistStruct(int portion)
{
    struct Ilist *ilist;

    if ((ilist = (struct Ilist *)calloc(1, sizeof(struct Ilist))) != NULL)
    {
        if (portion > 0 && portion < 1024 * 1024)
            ilist->portion = portion;
        else
            ilist->portion = 100;
    }
    return ilist;
}

void FreeIlistStruct(struct Ilist *ilist)
{
    char buf[50];

    if (ilist)
    {
        if (ilist->list)
            free(ilist->list);
        free(ilist);
    }
    else
    {
        sprintf(buf, "Free on NULL: %s, %d\n", __FILE__, __LINE__);
        fatal(buf);
    }
}

struct Ilist *CopyIlistStruct(struct Ilist *src)
{
    struct Ilist *ilist = NULL;

    ilist = AllocIlistStruct(src->portion);
    ilist->num = src->num;
    ilist->max = src->max;
    if ((ilist->list = (int *)calloc(ilist->max, sizeof(int))) == NULL)
    {
        fatal("calloc failed on (float)!");
    }
    memcpy(ilist->list, src->list, src->max * sizeof(int));

    return ilist;
}

struct Ilist *nCopyIlistStruct(struct Ilist *src, int srcnum)
{
    struct Ilist *ilist = NULL;

    ilist = AllocIlistStruct(src->portion);
    ilist->num = srcnum;
    ilist->max = src->max;
    if ((ilist->list = (int *)calloc(ilist->max, sizeof(int))) == NULL)
    {
        fatal("calloc failed on (float)!");
    }
    memcpy(ilist->list, src->list, srcnum * sizeof(int));

    return ilist;
}

void DumpIlist(struct Ilist *ilist)
{
    int i, lc;

    dprintf(5, "ilist->num = %d\t", ilist->num);
    dprintf(5, "ilist->max = %d\t", ilist->max);
    dprintf(5, "ilist->portion = %d\n", ilist->portion);
    for (lc = 0, i = 0; i < ilist->num; i++)
    {
        dprintf(5, "  ilist->list[%6d] = %6d |", i, ilist->list[i]);
        if (++lc == 3)
        {
            dprintf(5, "\n");
            lc = 0;
        }
    }
    if (lc)
        dprintf(5, "\n");
}

void DumpIlist2File(struct Ilist *ilist, FILE *fp)
{
    int i, lc;

    fprintf(fp, "ilist->num = %d\t", ilist->num);
    fprintf(fp, "ilist->max = %d\t", ilist->max);
    fprintf(fp, "ilist->portion = %d\n", ilist->portion);
    for (lc = 0, i = 0; i < ilist->num; i++)
    {
        fprintf(fp, "  ilist->list[%6d] = %6d |", i, ilist->list[i]);
        if (++lc == 3)
        {
            fputs("\n", fp);
            lc = 0;
        }
    }
    if (lc)
        fputs("\n", fp);
}
