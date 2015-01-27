/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ELEMENT_H_INCLUDED
#define ELEMENT_H_INCLUDED

struct Element
{
    int nume;
    int maxe;
    int **e;
};

int *GetElement(struct Element *e, int r[8], int ind);
struct Element *AllocElementStruct(void);
int AddElement(struct Element *p, int elem[8]);
void FreeElementStruct(struct Element *e);

#endif // ELEMENT_H_INCLUDED
