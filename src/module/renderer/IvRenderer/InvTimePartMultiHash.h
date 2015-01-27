/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TIME_PART_MULTI_HASH_
#define _INV_TIME_PART_MULTI_HASH_

// **************************************************************************
//
// Description    : utility class
//
// Class(es)      : multi hash table with double-hashing algorithm
//                  for key data "TimePart"
//
// Author  : Reiner Beller
//
// History :
//
// **************************************************************************

#include <util/coMultiHash.h>
#include "InvTimePart.h"

// maximum of two values

namespace covise
{
extern int Max(int v1, int v2);
template <class KEY, class DATA>
class coMultiHashBase;
}
using namespace covise;

/**
 * Class
 *
 */
template <class DATA>
class UTILEXPORT TimePartMultiHash : public coMultiHash<TimePart, DATA>
{

public:
    // constructor with NULL element
    TimePartMultiHash(const DATA &nullelem)
        : coMultiHash<TimePart, DATA>(nullelem){};

    // constructor without NULL element
    TimePartMultiHash()
        : coMultiHash<TimePart, DATA>(){};

    // maximum time in use
    int getMaxTime() const;

    // maximum part ID in use
    int getMaxPart() const;

private:
    /// 1st Hash function
    virtual unsigned long hash1(const TimePart &key) const
    {
        return ((key[0] + key[1]) % this->size);
    }

    /// 2nd Hash function
    virtual unsigned long hash2(const TimePart &key) const
    {
        return (this->size - 2 - (key[0] + key[1]) % (this->size - 2));
    }

    /// Equal function
    virtual bool equal(const TimePart &key1, const TimePart &key2) const
    {
        return (key1[0] == key2[0] && key1[1] == key2[1]);
    }
};

template <class DATA>
inline int TimePartMultiHash<DATA>::getMaxTime() const
{
    int maxTime;
    int count = 0;
    int first = 0;
    unsigned int i;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
        {
            count++;
            if (count == 1)
                first = this->keys[i].getTime();
        }

    if (count == 0)
        exit(-1); // R.B.: better exception throwing but not yet implemented

    // initialize
    maxTime = first;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
            maxTime = Max(maxTime, this->keys[i].getTime());

    return maxTime;
}

template <class DATA>
inline int TimePartMultiHash<DATA>::getMaxPart() const
{
    int maxPart;
    int count = 0;
    int first = 0;
    unsigned int i;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
        {
            count++;
            if (count == 1)
                first = this->keys[i].getPart();
        }

    if (count == 0)
        exit(-1); // R.B.: better exception throwing but not yet implemented

    // initialize
    maxPart = first;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
            maxPart = Max(maxPart, this->keys[i].getPart());

    return maxPart;
}
#endif // _INV_TIME_PART_MULTI_HASH_
