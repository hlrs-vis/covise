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
    a->exch_data_msg(shmmsg, {COVISE_MESSAGE_MALLOC_OK, COVISE_MESSAGE_MALLOC_FAILED});
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
