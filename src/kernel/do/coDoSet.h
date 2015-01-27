/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_SET_H
#define CO_DO_SET_H

/*
 $Log: covise_geometry.h,v $
 * Revision 1.1  1993/09/25  20:44:13  zrhk0125
 * Initial revision
 *
*/

#include "coDistributedObject.h"

/***********************************************************************\ 
 **                                                                     **
 **   Geometry class                                 Version: 1.0       **
 **                                                                     **
 **                                                                     **
 **   Description  : Container class handling multiple objects          **
 **                                                                     **
 **   Classes      : coDoSet                                             **
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

#define SET_CHUNK 20

// leads to duplicate symbols template class DOEXPORT ia<coDistributedObject *>;

namespace covise
{

class DOEXPORT coDoSet : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coIntShm no_of_elements;
    coIntShm max_no_of_elements;
    coShmPtrArray elements;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoSet *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoSet(const coObjInfo &info)
        : coDistributedObject(info, "SETELE")
    {
        if (name)
        {
            if (getShmArray())
            {
                if (rebuildFromShm() == 0)
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };

    /// create set from NULL-terminated list
    coDoSet(const coObjInfo &info, const coDistributedObject *const *elem);
    /// create set from list with given length
    coDoSet(const coObjInfo &info, int numElem, const coDistributedObject *const *elem);
    coDoSet(const coObjInfo &info, coShmArray *arr);
    coDoSet(const coObjInfo &info, int flag);

    const coDistributedObject *const *getAllElements(int *no = NULL) const;

    const coDistributedObject *getElement(int no) const
    {
        return elements[no];
    }

    void addElement(const coDistributedObject *elem);

    int getNumElements() const
    {
        return (int)no_of_elements;
    }
};
}
#endif
