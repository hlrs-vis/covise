/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_HASH_ITER_H_
#define _CO_HASH_ITER_H_

#include "coMultiHashBase.h"

#include <stdlib.h>

#ifndef INLINE
#define INLINE inline
#endif

namespace covise
{

template <class KEY, class DATA>
class coMultiHash;

template <class KEY, class DATA>
class coHash;

/**
 * Class coHashIter: Iterator class for Hash tables
 * defines sequential access to Hash tables
 *
 */
template <class KEY, class DATA>
class coHashIter
{

    friend class coMultiHash<KEY, DATA>;
    friend class coHash<KEY, DATA>;

public:
    /// Empty HashIter
    coHashIter();

    /// Initialize to first Element of hash
    coHashIter(coMultiHashBase<KEY, DATA> &table);

    /// Initialize to hash key
    coHashIter(coMultiHashBase<KEY, DATA> &table,
               unsigned long hash);

    // Copy-Constructor: use default bitcopy
    // coHashIter(const coHashIter<KEY,DATA> &);

    // Assignment operator: use default bitcopy
    // coHashIter &operator =(const coHashIter<KEY,DATA> &);

    /// Initialize to first Element of hash
    void reset();

    /// Existance: (iter) == true if actual element exists
    operator bool();

    /// Increment
    void operator++();

    /// Access: iter() accesses element
    DATA &operator()();

    /// Access: iter.key() accesses element key
    KEY key();

    /// Access: iter.hash() accesses hash value
    unsigned long hash();

private:
    // pointer to 'my' hashtable
    coMultiHashBase<KEY, DATA> *d_hash;

    // index of the actual element (if running on indices)
    unsigned long d_index;

    // actual hash (if running on hashes) (=index+1)
    unsigned long d_hashIndex;
};

#define HT_EMPTY coMultiHashBase<KEY, DATA>::EMPTY
#define HT_PREVIOUS coMultiHashBase<KEY, DATA>::PREVIOUD
#define HT_USED coMultiHashBase<KEY, DATA>::USED

// Empty constructor
template <class KEY, class DATA>
INLINE coHashIter<KEY, DATA>::coHashIter()
{
    d_hash = NULL;
}

// Initialize for sequential access
template <class KEY, class DATA>
INLINE coHashIter<KEY, DATA>::coHashIter(coMultiHashBase<KEY, DATA> &hash)
{
    d_index = 0;
    d_hashIndex = 0;
    d_hash = &hash;
    while ((d_index < d_hash->size) && (d_hash->entryFlags[d_index] != HT_USED))
        d_index++;
}

// Initialize to first Element of hash series
template <class KEY, class DATA>
INLINE coHashIter<KEY, DATA>::coHashIter(coMultiHashBase<KEY, DATA> &hash,
                                         unsigned long hashIndex)
{
    d_index = hashIndex - 1;
    d_hashIndex = hashIndex;
    d_hash = &hash;
}

// test correctness
template <class KEY, class DATA>
INLINE coHashIter<KEY, DATA>::operator bool()
{
    return ((d_hash)
            && (d_index < d_hash->size)
            && (d_hash->entryFlags[d_index] == HT_USED));
}

template <class KEY, class DATA>
INLINE void coHashIter<KEY, DATA>::operator++()
{
    //static const char flag[3]={'E','P','U'};
    // running on hashes
    if (d_hashIndex)
    {
        //cerr << " call d_hash->nextHash(" << d_hashIndex << ");" << endl;
        //for (unsigned long i=0;i<d_hash->size;i++) {
        //   cerr.width(4);
        //   cerr << i << "  :" << flag[d_hash->entryFlags[i]]
        //        << " : ";
        //   if (d_hash->entryFlags[i]) {
        //      cerr.width(4);
        //      cerr <<  d_hash->keys[i] << " : " << d_hash->data[i];
        //   }
        //   cerr << endl;
        // }
        d_hashIndex = d_hash->nextHash(d_hashIndex);
        d_index = d_hashIndex - 1;
        // cerr << " call d_hash->nextHash() returned " << d_hashIndex << endl;
    }
    // running on indices
    else
    {
        d_index++;
        while ((d_index < d_hash->size) && (d_hash->entryFlags[d_index] != HT_USED))
            d_index++;
    }
}

template <class KEY, class DATA>
INLINE DATA &coHashIter<KEY, DATA>::operator()()
{
    assert((d_index < d_hash->size) && (d_hash->entryFlags[d_index] == HT_USED));
    return d_hash->data[d_index];
}

template <class KEY, class DATA>
INLINE void coHashIter<KEY, DATA>::reset()
{
    if (!d_hash)
        return;
    d_index = 0;
    d_hashIndex = 0;
    while ((d_index < d_hash->size) && (d_hash->entryFlags[d_index] != HT_USED))
        d_index++;
}

template <class KEY, class DATA>
INLINE KEY coHashIter<KEY, DATA>::key()
{
    assert((d_index < d_hash->size) && (d_hash->entryFlags[d_index] == HT_USED));
    return d_hash->keys[d_index];
}

template <class KEY, class DATA>
INLINE unsigned long coHashIter<KEY, DATA>::hash()
{
    return d_index;
    //return d_hashIndex;
}
}
#undef HT_EMPTY
#undef HT_PREVIOUS
#undef HT_USED
#endif
