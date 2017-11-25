/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "covise_shm.h"
#include <net/message.h>

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
 **		    ShmAccess allows only the access to the shared     **
 **		    memory areas, not the allocation or return of      **
 **		    allocated regions.                                 **
 **		    coShmAlloc does all the administration of the shared **
 **		    memory regions, using trees with nodes which point **
 **		    to used and free parts. Here all allocation of     **
 **		    regions in the shared memory takes place.          **
 **                                                                     **
 **   Classes      : ShmAccess, coShmAlloc                                **
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

void coShmItem::print()
{
    print_comment(__LINE__, __FILE__, "coShmItem shm_seq_no: %d  offset: %d", shm_seq_no, offset);
}

ShmAccess::ShmAccess(int k)
{
    shm = new SharedMemory(k, (shmSizeType)ShmConfig::getMallocSize());
}

ShmAccess::ShmAccess(int *k)
{
    shm = new SharedMemory(k, (shmSizeType)ShmConfig::getMallocSize());
}

ShmAccess::ShmAccess(char *d, int noDelete)
{
#ifdef DEBUG
    char tmp_str[255];
#endif

    char *ptr = (char *)d;
    int num = *((int *)d);
    ptr += sizeof(int);
    for (int i = 0; i < num; i++)
    {
#ifdef DEBUG
        sprintf(tmp_str, "new SharedMemory(%d, %d)", *((int *)ptr), *((shmSizeType *)(ptr+sizeof(int))));
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        shm = new SharedMemory(*((int *)ptr), *((shmSizeType *)(ptr+sizeof(int))), noDelete);
        ptr += sizeof(int)+sizeof(shmSizeType);
    }
}

ShmAccess::~ShmAccess()
{
    delete shm;
}

void ShmAccess::add_new_segment(int k, shmSizeType size)
{
    new SharedMemory(k, size, 1); // 1 means do not delete stored Memory segments
    // only detach
}

char *coStringShmArray::operator[](unsigned int i)
{
    char *tmpch;
    int sn;
    shmSizeType of;
    coCharShmArray *tmparraych;
    char *buf = (char *)(COVISE_POINTER_TYPE)ptr + 2 * sizeof(int);
        // 2* sizeof(int) is seq_nr + key
    
    cerr << "check this, should be only two numbers, not three\n"
         << i << " not in 0.." << length << endl;

    if (i < length)
    {
        //old sn = (int)((COVISE_POINTER_TYPE)ptr + 2 * sizeof(int)) + 3 * i * sizeof(int) + 1;
        //old of = (int)((COVISE_POINTER_TYPE)ptr + 2 * sizeof(int)) + 3 * i * sizeof(int) + 2;
        sn = *(int *)(buf+sizeof(int));
        of = *(shmSizeType *)(buf+sizeof(int)+sizeof(int));
        buf += sizeof(int) + sizeof(int) + sizeof(shmSizeType);
        tmparraych = new coCharShmArray(sn, of);
        tmpch = (char *)tmparraych->getDataPtr();
        delete tmparraych;
        return tmpch;
    }
    //  else
    cerr << "Access error for coStringShmArray\n"
         << i << " not in 0.." << length << endl;
    return NULL;
}

const char *coStringShmArray::operator[](unsigned int i) const
{
    char *tmpch;
    int sn;
    shmSizeType of;
    coCharShmArray *tmparraych;
    char *buf = (char *)(COVISE_POINTER_TYPE)ptr + 2 * sizeof(int);
        // 2* sizeof(int) is seq_nr + key
    
    cerr << "check this, should be only two numbers, not three\n"
         << i << " not in 0.." << length << endl;
    if (i < length)
    {
        //old sn = (int)((COVISE_POINTER_TYPE)ptr + 2 * sizeof(int)) + 3 * i * sizeof(int) + 1;
        //old of = (int)((COVISE_POINTER_TYPE)ptr + 2 * sizeof(int)) + 3 * i * sizeof(int) + 2;
        sn = *(int *)(buf+sizeof(int));
        of = *(shmSizeType *)(buf+sizeof(int)+sizeof(int));
        buf += sizeof(int) + sizeof(int) + sizeof(shmSizeType);
        tmparraych = new coCharShmArray(sn, of);
        tmpch = (char *)tmparraych->getDataPtr();
        delete tmparraych;
        return tmpch;
    }
    //  else
    cerr << "Access error for coStringShmArray\n"
         << i << " not in 0.." << length << endl;
    return NULL;
}
