#ifndef  ELEMENT_H_INCLUDED
#define  ELEMENT_H_INCLUDED

#ifndef NPE
#define NPE 8
#endif

struct Element
{
   int nume;
   int maxe;
   int **e;
};

int *GetElement(struct Element *e, int r[8], int ind);
struct Element * AllocElementStruct(void);
int AddElement(struct Element *p, int elem[8]);
void FreeElementStruct(struct Element *e);
#endif                                            // ELEMENT_H_INCLUDED
