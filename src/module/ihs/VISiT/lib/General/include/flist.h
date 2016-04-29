#ifndef  FLIST_H_INCLUDED
#define  FLIST_H_INCLUDED

struct Flist
{
   int num;
   int max;
   int portion;
   float *list;
};

void Add2Flist(struct Flist *flist, float ind);
struct Flist * AllocFlistStruct(int portion);
void FreeFlistStruct(struct Flist *flist);
struct Flist *CopyFlistStruct(struct Flist *src);
struct Flist *nCopyFlistStruct(struct Flist *src, int srcnum);
struct Flist *nCopynFlistStruct(struct Flist *src, int isrc, int srcnum);

#ifdef   DEBUG
void DumpFlist(struct Flist *flist);
void DumpFlist2File(struct Flist *flist, FILE *fp);
#endif                                            // DEBUG
#endif                                            // FLIST_H_INCLUDED
