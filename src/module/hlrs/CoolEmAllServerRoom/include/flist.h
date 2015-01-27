/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FLIST_H_INCLUDED
#define FLIST_H_INCLUDED

struct Flist
{
    int num;
    int max;
    int portion;
    float *list;
};

void Add2Flist(struct Flist *flist, float ind);
struct Flist *AllocFlistStruct(int portion);
void FreeFlistStruct(struct Flist *flist);
struct Flist *CopyFlistStruct(struct Flist *src);
struct Flist *nCopyFlistStruct(struct Flist *src, int srcnum);

void DumpFlist(struct Flist *flist);
void DumpFlist2File(struct Flist *flist, FILE *fp);

#endif // FLIST_H_INCLUDED
