/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#ifdef _AIX
#include <grp.h>
#endif

#include "coDoSet.h"
#include <covise/covise_appproc.h>

#undef DEBUG

/***********************************************************************\ 
 **                                                                     **
 **   Geometry class                                 Version: 1.0       **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of Geometry for the       **
 **                  renderer                   		               **
 **                                                                     **
 **   Classes      :                                                    **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  12.08.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

/* removed from header

      coDoSet(const char *n, coDistributedObject *elem);
      void add_elements(coDistributedObject **elem);
      coDistributedObject *get_element(const char *n);
      int get_elements(int no, const char **n, coDistributedObject **elem);
      int getNumElements()
      { return elements.get_length(); }

      int getNumElements() { return (int) no_of_elements; }

*/

using namespace covise;

coDistributedObject *coDoSet::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = (coDistributedObject *)new coDoSet(coObjInfo(), arr);
    return ret;
}

int coDoSet::getObjInfo(int no, coDoInfo **il) const
{
    int *eleptr;
    int count, len, i;
    coShmArray *tmparr;
    coDoInfo *tmpinfolist;
    coDoHeader *header;
    const char *tmpcptr;

    if (no == 3)
    {
        count = *(int *)((*il)[0].ptr);
        tmpinfolist = new coDoInfo[count + 2]; // +2 for the first two entries
        tmpinfolist[0].type = (*il)[0].type;
        tmpinfolist[0].type_name = (*il)[0].type_name;
        tmpinfolist[0].description = "Number of Elements";
        tmpinfolist[0].obj_name = (*il)[0].obj_name;
        (*il)[0].obj_name = NULL;
        tmpinfolist[0].ptr = (*il)[0].ptr;
        tmpinfolist[1].type = (*il)[1].type;
        tmpinfolist[1].type_name = (*il)[1].type_name;
        tmpinfolist[1].description = "Maximum Number of Elements";
        tmpinfolist[1].obj_name = (*il)[1].obj_name;
        (*il)[1].obj_name = NULL;
        tmpinfolist[1].ptr = (*il)[1].ptr;
        eleptr = (int *)(*il)[2].ptr;
        delete[] * il;
        *il = tmpinfolist;

        for (i = 0; i < count; i++)
        {
            if (eleptr[2 + 2 * i])
            {
                tmparr = new coShmArray(eleptr[2 + 2 * i], eleptr[2 + 2 * i + 1]);
                header = (coDoHeader *)tmparr->getPtr();
                tmpinfolist[2 + i].type = header->getObjectType();
                tmpinfolist[2 + i].type_name = calcTypeString(tmpinfolist[2 + i].type);
                tmpcptr = header->getName();
                len = (int)(strlen(tmpcptr) + 1);
                tmpinfolist[2 + i].obj_name = new char[len];
                strcpy(tmpinfolist[2 + i].obj_name, tmpcptr);
                delete tmparr;
            }
            else
            {
                tmpinfolist[2 + i].obj_name = NULL;
                tmpinfolist[2 + i].type = 0;
                tmpinfolist[2 + i].type_name = NULL;
            }
        }
        return count + 2; // length + maxlength + setelements
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoSet::coDoSet(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info, "SETELE")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoSet::coDoSet(const coObjInfo &info, int flag)
    : coDistributedObject(info, "SETELE")
{
    if (flag == SET_CREATE)
    {
        elements.set_length(SET_CHUNK);

        covise_data_list dl[] = {
            { INTSHM, &no_of_elements },
            { INTSHM, &max_no_of_elements },
            { SHMPTRARRAY, &elements }
        };

        new_ok = store_shared_dl(3, dl) != 0;
        if (!new_ok)
            return;

        no_of_elements = 0;
        max_no_of_elements = SET_CHUNK;
    }
    else
    {
        print_comment(__LINE__, __FILE__, "wrong flag in new coDoSet");
    }
}

coDoSet::coDoSet(const coObjInfo &info, const coDistributedObject *const *elem)
    : coDistributedObject(info, "SETELE")
{
    int ne = 0, j, max;
    while (elem[ne] != NULL)
        ne++;
    max = (ne / SET_CHUNK + 1) * SET_CHUNK;
    elements.set_length(max);

    covise_data_list dl[] = {
        { INTSHM, &no_of_elements },
        { INTSHM, &max_no_of_elements },
        { SHMPTRARRAY, &elements }
    };

    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;

    no_of_elements = ne;
    max_no_of_elements = max;
    for (j = 0; j < ne; j++)
        elements.set(j, elem[j]);

    // fill rest with NULL pointers
    for (; j < max; j++)
        elements.set(j, NULL);
}

coDoSet::coDoSet(const coObjInfo &info, int numElem, const coDistributedObject *const *elem)
    : coDistributedObject(info, "SETELE")
{
    int j, max;
    max = (numElem / SET_CHUNK + 1) * SET_CHUNK;
    elements.set_length(max);

    covise_data_list dl[] = {
        { INTSHM, &no_of_elements },
        { INTSHM, &max_no_of_elements },
        { SHMPTRARRAY, &elements }
    };

    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;

    no_of_elements = numElem;
    max_no_of_elements = max;
    for (j = 0; j < numElem; j++)
        elements.set(j, elem[j]);

    // fill rest with NULL pointers
    for (; j < max; j++)
        elements.set(j, NULL);
}

coDoSet *coDoSet::cloneObject(const coObjInfo &newinfo) const
{
    const coDistributedObject *const *elems = getAllElements();
    for (size_t i=0; i<getNumElements(); ++i) {
        elems[i]->incRefCount();
    }

    coDoSet *set = new coDoSet(newinfo, getNumElements(), elems);
    delete[] elems;
    return set;
}

int coDoSet::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    covise_data_list dl[] = {
        { INTSHM, &no_of_elements },
        { INTSHM, &max_no_of_elements },
        { SHMPTRARRAY, &elements }
    };

    return restore_shared_dl(3, dl);
}

void coDoSet::addElement(const coDistributedObject *elem)
{
    if (no_of_elements.get() >= max_no_of_elements.get())
    {
        elements.grow(ApplicationProcess::approc, SET_CHUNK);

        max_no_of_elements = max_no_of_elements.get() + SET_CHUNK;

        covise_data_list dl[] = {
            { INTSHM, &no_of_elements },
            { INTSHM, &max_no_of_elements },
            { SHMPTRARRAY, &elements }
        };

        update_shared_dl(3, dl);
    }
    elements.set(no_of_elements, elem);
    no_of_elements = no_of_elements + 1;
}

// Added by Uwe Woessner 26.09

const coDistributedObject *const *coDoSet::getAllElements(int *no) const
{
    /*
    *no=no_of_elements.get();
      if(*no != keepElements_.size())
      {
         int i;
         for(i = 0;i < *no;i++)
         {
            keepElements_.push_back(elements[i]);
         }
      }
      return keepElements_.getArray();
   */
    const coDistributedObject **objs;
    int n = no_of_elements.get();
    if (no)
        *no = n;
    objs = new const coDistributedObject *[n];
    for (int i = 0; i < n; i++)
    {
        objs[i] = elements[i];
    }
    return objs;
}
