/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr.h"

using namespace covise;

int DmgrMessage::process_list(DataManagerProcess *dmgr)
{
    coShmPtr *shmptr;
    data_type dt;
    int no = length / (sizeof(data_type) + sizeof(long));
    int i, j, k;
    long size;
    char *chdata = new char[no * 2 * sizeof(int)];

    for (i = 0, j = 0, k = 0; i < no; i++)
    {
        dt = *(data_type *)(&data[j]);
        j += sizeof(data_type);
        size = *(long *)(&data[j]);
        j += sizeof(long);
        shmptr = dmgr->shm_alloc(dt, size);
        *(int *)(&chdata[k]) = shmptr->shm_seq_no;
        k += sizeof(int);
        *(int *)(&chdata[k]) = shmptr->offset;
        k += sizeof(int);
    }
    delete_data();
    data = chdata;
    length = k;
    type = COVISE_MESSAGE_MALLOC_LIST_OK;
    return 1;
}

int DmgrMessage::process_new_object_list(DataManagerProcess *dmgr)
{
    coShmPtr *shmptr;
    data_type dt;
    int no, ok, name_len;
    int i, j, k, otype;
    int start_data;
    long size;
    char *name, *tmp_data;
    char *chdata;

    otype = *(int *)data;
    name_len = int(strlen(data + int(sizeof(int))) + 1);
    name = new char[name_len];
    strcpy(name, &data[sizeof(int)]);
    if (name_len % SIZEOF_ALIGNMENT)
        start_data = sizeof(long) + (name_len / SIZEOF_ALIGNMENT + 1) * SIZEOF_ALIGNMENT;
    else
        start_data = sizeof(long) + name_len;
    tmp_data = data + start_data;
    no = (length - start_data) / (sizeof(data_type) + sizeof(long));
    chdata = new char[no * 2 * sizeof(int)];

    for (i = 0, j = 0, k = 0; i < no; i++)
    {
        dt = *(data_type *)(&tmp_data[j]);
        j += sizeof(data_type);
        size = *(long *)(&tmp_data[j]);
        j += sizeof(long);
        shmptr = dmgr->shm_alloc(dt, size);
        *(int *)(&chdata[k]) = shmptr->shm_seq_no;
        k += sizeof(int);
        *(int *)(&chdata[k]) = shmptr->offset;
        k += sizeof(int);
        delete shmptr;
    }
    delete_data();
    data = chdata;
    length = k;
    type = COVISE_MESSAGE_MALLOC_LIST_OK;
    ok = dmgr->add_object(name, otype, *(int *)data, *(int *)(&data[sizeof(int)]), conn);
    return ok;
}
