/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <shm/covise_shm.h>
#include <covise/covise_process.h>
#include <covise/covise_appproc.h>
#include <covise/covise_msg.h>
#include <do/coDistributedObject.h>

/*
 $Log: covise_shmalloc.C,v $
Revision 1.4  1994/03/23  18:07:06  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.4  93/12/10  13:45:31  zrfg0125
modification for several SDS

Revision 1.3  93/10/08  19:11:10  zrhk0125
adjustment of memory alignment

Revision 1.2  93/09/30  17:08:38  zrhk0125
basic modifications for CRAY

Revision 1.1  93/09/25  20:50:21  zrhk0125
Initial revision

*/

/***********************************************************************\ 
 **                                                                     **
 **   Shared Memory Classes                       Version: 1.0          **
 **                                                                     **
 **                                                                     **
 **   Description  : The classes that deal with the creation and        **
 **                  administration of shared memory.                   **
 **                  is a utility class to organize the used            **
 **		    and unused parts of the shared memory.             **
 **		    ShMaccess allows only the access to the shared     **
 **		    memory areas, not the allocation or return of      **
 **		    allocated regions.                                 **
 **		    coShmAlloc does all the administration of the shared **
 **		    memory regions, using trees with nodes which point **
 **		    to used and free parts. Here all allocation of     **
 **		    regions in the shared memory takes place.          **
 **                                                                     **
 **   Classes      : ShMaccess, coShmAlloc                                **
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
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93        some prints for debugging inserted **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#undef DEBUG

using namespace covise;

const coDistributedObject *coShmPtrArray::operator[](unsigned int i) const
{
    int *iptr = (int *)getDataPtr();
    const coDistributedObject *tmpptr;
    coShmArray *tmparr;

    if ((i >= length) || (iptr[2 * i] == 0)) // aw: identify null obj
        return NULL;
    tmparr = new coShmArray(iptr[2 * i], iptr[2 * i + 1]);
    tmpptr = coDistributedObject::createUnknown(tmparr);
    if (tmpptr && tmpptr->objectOk())
    {
        delete tmparr;
        return tmpptr;
    }
    else
    {
        delete tmparr;
        delete tmpptr;
        return NULL;
    }
}

int coShmPtrArray::holds_object(int i)
{
    int *iptr = (int *)getDataPtr();

    if (iptr[2 * i])
        return 1;
    else
        return 0;
}

void coShmPtrArray::set(int i, const coDistributedObject *elem)
{
    coShmArray *tmparr;

    //    sprintf(tmp_str,"setting element no. %d", i);
    //    print_comment(__LINE__, __FILE__, tmp_str);
    int *iptr = (int *)getDataPtr();

    if (elem)
    {
        tmparr = elem->shmarr;
        iptr[2 * i] = tmparr->shm_seq_no;
        iptr[2 * i + 1] = tmparr->offset;
    }
    else // if a NULL pointer is given in here, we set 0/0
    {
        iptr[2 * i] = 0;
        iptr[2 * i + 1] = 0;
    }
    //    print();
}

int coShmPtrArray::grow(ApplicationProcess *a, unsigned int s)
{
    ShmMessage *shmmsg;
    coShmArray *tmparr;
	int *iptr_new, *iptr_old;
	unsigned int i;

    //    print_comment(__LINE__, __FILE__, "growing coShmPtrArray");
    //    print();
    shmmsg = new ShmMessage(SHMPTRARRAY, length + s);
    a->exch_data_msg(shmmsg, 2, COVISE_MESSAGE_MALLOC_OK, COVISE_MESSAGE_MALLOC_FAILED);
    shm_seq_no = *(int *)&shmmsg->data.data()[0];
    offset = *(int *)&shmmsg->data.data()[sizeof(int)];
    tmparr = new coShmArray(shm_seq_no, offset);
    iptr_new = (int *)tmparr->getPtr();
    iptr_old = (int *)getPtr();
    for (i = 0; i < length; i++)
    {
        iptr_new[2 + 2 * i] = iptr_old[2 + 2 * i];
        iptr_new[2 + 2 * i + 1] = iptr_old[2 + 2 * i + 1];
    }
    for (i = length; i < length + s; i++)
        iptr_new[2 + 2 * i] = iptr_new[2 + 2 * i + 1] = 0;
    ptr = iptr_new;
    length += s;
    //    print();
    return 1;
}

void coShmPtrArray::print()
{
    unsigned int i;

    print_comment(__LINE__, __FILE__, "Printing coShmPtrArray Object ---------");
    int *iptr = (int *)getDataPtr();
    for (i = 0; i < length; i++)
    {
        print_comment(__LINE__, __FILE__, "entry %d: (%d, %d)", i, iptr[2 * i], iptr[2 * i + 1]);
    }
}
