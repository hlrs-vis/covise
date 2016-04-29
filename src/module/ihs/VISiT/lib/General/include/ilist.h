#ifndef  ILIST_H_INCLUDED
#define  ILIST_H_INCLUDED

struct Ilist
{
   int num;
   int max;
   int portion;
   int *list;
};

void Add2Ilist(struct Ilist *ilist, int ind);
struct Ilist * AllocIlistStruct(int portion);
void FreeIlistStruct(struct Ilist *il);
struct Ilist *CopyIlistStruct(struct Ilist *src);
struct Ilist *nCopyIlistStruct(struct Ilist *src, int srcnum);
struct Ilist *nCopynIlistStruct(struct Ilist *src, int isrc, int srcnum);

#ifdef   DEBUG
void DumpIlist(struct Ilist *list);
void DumpIlist2File(struct Ilist *ilist, FILE *fp);
#endif                                            // DEBUG
#endif                                            // ILIST_H_INCLUDED
