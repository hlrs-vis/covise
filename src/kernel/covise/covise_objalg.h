/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// class coObjectAlgorithms
//
// coObjectAlgorithms is a vase for algorithms for DistributeObject processing
//
// Initial version: 2003-06, sl
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by Vircinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _COVISE_OBJECT_ALGORITHMS_
#define _COVISE_OBJECT_ALGORITHMS_

#include "covise.h"
#include <do/coDistributedObject.h>

namespace covise
{

class COVISEEXPORT coObjectAlgorithms
{
public:
    /** coObjectAlgorithms
       *  @param obj: the object, possibly, but not necessarily a set
       *  @param types: list of types for which we want to test obj
       *  @param strict_case: if true only the listed types pass the test
       *  @return whether this object passes the type test
       */
    template <class T>
    static bool containsType(const coDistributedObject *obj,
                             bool strict_case = true);
};

template <class T>
bool
coObjectAlgorithms::containsType(const coDistributedObject *obj, bool strict_case)
{
    if (!obj)
    {
        return true; // no object is no problem
    }

    if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
    {
        int no_elems;
        const coDistributedObject *const *setList = set->getAllElements(&no_elems);
        int i;
        bool ret = true;
        for (i = 0; i < no_elems; ++i)
        {
            bool thisType = containsType<T>(setList[i], strict_case);
            if (!thisType && strict_case)
            {
                ret = false;
                break;
            }
            else if (!strict_case && thisType)
            {
                break;
            }
        }
        int elem;
        for (elem = 0; elem < no_elems; ++elem)
        {
            delete setList[elem];
        }
        delete[] setList;
        if (!ret) // see !thisType && strict_case
        {
            return false;
        }
        else if (i != no_elems || strict_case) // see !strict_case && thisType
        {
            //   or strict_case and all true
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return dynamic_cast<T>(obj) != NULL;
    }
    return true;
}
}
#endif
