/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SHM_PTR_ARRAY_H
#define SHM_PTR_ARRAY_H

#include <covise/covise.h>
#include <util/coExport.h>

#if !defined(_WIN32)
#define SHARED_MEMORY
#endif

#ifdef _CRAYT3E
#define HANDLE unsigned int
#endif

#include <util/coTypes.h>

#include <util/covise_list.h>
#include <covise/covise_global.h>
#include <shm/covise_shm.h>
#ifdef _WIN32
#include <windows.h>
#include <windowsx.h>
#endif

namespace covise
{

class ApplicationProcess;

class DOEXPORT coShmPtrArray : public coShmArray
{
public:
    coShmPtrArray()
        : coShmArray(){};
    coShmPtrArray(int no, int o)
        : coShmArray(no, o)
    {
        if (type != SHMPTRARRAY)
        {
            cerr << "wrong type in coShmPtrArray constructor from shared memory\n";
            print_exit(__LINE__, __FILE__, 1);
        }
    };
    void setPtr(int no, int o)
    {
        coShmArray::setPtr(no, o);
        if (type != SHMPTRARRAY)
        {
            cerr << "wrong type in coShmPtrArray constructor from shared memory\n";
            print_exit(__LINE__, __FILE__, 1);
        }
    };
    ~coShmPtrArray(){};
    const coDistributedObject *operator[](unsigned int i) const;
    int holds_object(int);
    int grow(ApplicationProcess *a, unsigned int s); // __alpha
    void set(int i, const coDistributedObject *elem);
    void print();
};
}
#endif
