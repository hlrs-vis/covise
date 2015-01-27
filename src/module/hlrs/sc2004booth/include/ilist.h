/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ILIST_H_INCLUDED
#define ILIST_H_INCLUDED

struct Ilist
{
    int num;
    int max;
    int portion;
    int *list;
};

void Add2Ilist(struct Ilist *ilist, int ind);
struct Ilist *AllocIlistStruct(int portion);
void FreeIlistStruct(struct Ilist *il);
struct Ilist *CopyIlistStruct(struct Ilist *src);
struct Ilist *nCopyIlistStruct(struct Ilist *src, int srcnum);
void DumpIlist(struct Ilist *list);
void DumpIlist2File(struct Ilist *ilist, FILE *fp);

#endif // ILIST_H_INCLUDED
