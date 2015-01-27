/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_STRING_MULTI_HASH_H_
#define _CO_STRING_MULTI_HASH_H_

#include "coMultiHash.h"

#include <cstring>

// 24.02.01

/**
 * Class
 *
 */
namespace covise
{

template <class DATA>
class coStringMultiHash : public coMultiHash<const char *, DATA>
{

public:
    // constructor with NULL element
    coStringMultiHash(const DATA &nullelem)
        : coMultiHash<const char *, DATA>(nullelem){};

    // constructor without NULL element
    coStringMultiHash()
        : coMultiHash<const char *, DATA>(){};

private:
    // Copy-Constructor: NOT  IMPLEMENTED
    coStringMultiHash(const coStringMultiHash &);

    // Assignment operator: NOT  IMPLEMENTED
    coStringMultiHash &operator=(const coStringMultiHash &x);

    /// 1st Hash function
    unsigned long hash1(const char *const &key) const
    {
        int h;
        const char *t = key;
        for (h = 0; *t; t++)
            h = (64 * h + *t) % this->size;
        return (h);
    }

    /// 2nd Hash function
    unsigned long hash2(const char *const &key) const
    {
        int h;
        const char *t = key;
        for (h = 0; *t; t++)
            h = (64 * h + *t) % (this->size - 2);
        return (this->size - 2 - h);
    }

    /// Equal function
    bool equal(const char *const &key1, const char *const &key2) const
    //{  if((key1)&&(key2)) return( key1==key2) ); else return 0; }
    {
        if (NULL == key1)
            return (false);
        if (NULL == key2)
            return (false);
        return (0 == strcmp(key1, key2));
    }
};
}
#endif
