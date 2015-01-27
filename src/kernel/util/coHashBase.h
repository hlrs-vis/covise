/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CO_HASH_BASE_H
#define __CO_HASH_BASE_H

#include "coExport.h"
#include "coMultiHashBase.h"

/**
 *  Basic class implementing the double-hashing algorithm
 *  SingleMap algorithm
 *
 *  @author		Lars Frenzel, A. Werner
 *  @version		1.1
 *
 */

namespace covise
{

template <class KEY, class DATA>
class coHashBase : public coMultiHashBase<KEY, DATA>
{

public:
    coHashBase(DATA nullelem)
        : coMultiHashBase<KEY, DATA>(nullelem){};

    coHashBase()
        : coMultiHashBase<KEY, DATA>(){};

    /// no identical keys in table
    virtual unsigned long nextHash(unsigned long /* hashIndex */) const
    {
        return 0;
    }

    /// insert: if existing element, replace it
    virtual int insert(const KEY &key, const DATA &inData);

    /// first hash function <em>(pure virtual)</em>
    virtual unsigned long hash1(const KEY &) const = 0;

    /// second hash function <em>(pure virtual)</em>
    virtual unsigned long hash2(const KEY &) const = 0;

    /// KEY1 == KEY2 operation <em>(pure virtual)</em>
    virtual bool equal(const KEY &, const KEY &) const = 0;
};

// +++++++++++ insert an element a hash table

template <class KEY, class DATA>
inline int coHashBase<KEY, DATA>::insert(const KEY &key, const DATA &inData)
{
    unsigned int x, u, hash;

    // find free space in the table
    x = hash1(key);
    u = hash2(key);

    // replace same key
    while ((this->entryFlags[x] == coMultiHashBase<KEY, DATA>::USED)
           && (!equal(key, this->keys[x])))
        x = (x + u) % this->size;

    // copy key and data to respective fields, save Flag info
    if (this->entryFlags[x] != coMultiHashBase<KEY, DATA>::USED)
        this->entries++;
    this->keys[x] = key;
    this->data[x] = inData;
    this->entryFlags[x] = coMultiHashBase<KEY, DATA>::USED;
    hash = x;

    // delete possible other entries with same key
    x = (x + u) % this->size;
    while ((this->entryFlags[x] != coMultiHashBase<KEY, DATA>::EMPTY)
           && (x != hash) && (!equal(key, this->keys[x])))
    {
        x = (x + u) % this->size;
    }
    if (this->entryFlags[x] != coMultiHashBase<KEY, DATA>::EMPTY && this->keys[x] == key)
    {
        this->entryFlags[x] = coMultiHashBase<KEY, DATA>::PREVIOUS;
    }

    // resize table if needed
    if (this->entries > this->maxEntries)
        this->resize();

    // done
    return (1);
}
}
#endif
