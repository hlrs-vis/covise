/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_INT_MULTI_HASH_H_
#define _CO_INT_MULTI_HASH_H_

#include "coMultiHash.h"

// 27.02.98

// maximum of two values
namespace covise
{

inline int Max(int v1, int v2)
{
    return (v1 >= v2 ? v1 : v2);
}

template <class KEY, class DATA>
class coMultiHashBase;

/**
 * Class
 *
 */
template <class DATA>
class coIntMultiHash : public coMultiHash<int, DATA>
{

public:
    // constructor with NULL element
    coIntMultiHash(const DATA &nullelem)
        : coMultiHash<int, DATA>(nullelem){};

    // constructor without NULL element
    coIntMultiHash()
        : coMultiHash<int, DATA>(){};

    // maximum key number in use
    int getMaxKey() const;

private:
    /// 1st Hash function
    virtual unsigned long hash1(const int &key) const
    {
        return (key % this->size);
    }

    /// 2nd Hash function
    virtual unsigned long hash2(const int &key) const
    {
        return (this->size - 2 - key % (this->size - 2));
    }

    /// Equal function
    virtual bool equal(const int &key1, const int &key2) const
    {
        return (key1 == key2);
    }
};

template <class DATA>
inline int coIntMultiHash<DATA>::getMaxKey() const
{
    int maxKey;
    int count = 0;
    int first = 0;
    unsigned int i;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
        {
            count++;
            if (count == 1)
                first = this->keys[i];
        }

    if (count == 0)
        exit(-1); // R.B.: better exception throwing but not yet implemented

    // initialize
    maxKey = first;

    for (i = 0L; i < this->size; i++)
        if (this->entryFlags[i] == this->USED)
            maxKey = Max(maxKey, this->keys[i]);

    return maxKey;
}
}
#endif
