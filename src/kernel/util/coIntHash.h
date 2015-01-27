/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_INT_HASH_H_
#define _CO_INT_HASH_H_

#include "coHash.h"

// 24.02.98

/**
 * Class
 *
 */
namespace covise
{

template <class DATA>
class coIntHash : public coHash<int, DATA>
{

public:
    // constructor with NULL element
    coIntHash(const DATA &nullelem)
        : coHash<int, DATA>(nullelem){};

    // constructor without NULL element
    coIntHash()
        : coHash<int, DATA>(){};

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
}
#endif
