/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_MULTI_HASH_H_
#define _CO_MULTI_HASH_H_

// 27.02.98

#include "coMultiHashBase.h"

#include <assert.h>

namespace covise
{

template <class KEY, class DATA>
class coMultiHashBase;

#ifndef INLINE
#define INLINE inline
#endif

/**
 * User-Class for Multi-Hasch tables
 *
 */
template <class KEY, class DATA>
class coMultiHash : protected coMultiHashBase<KEY, DATA>
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coMultiHash(const coMultiHash &)
    {
        assert(0);
    }

    /// Assignment operator: NOT  IMPLEMENTED
    coMultiHash &operator=(const coMultiHash &)
    {
        assert(0);
        return *this;
    }

public:
    /// Default constructor
    coMultiHash()
        : coMultiHashBase<KEY, DATA>()
    {
    }

    /// Default constructor with NULL element
    coMultiHash(DATA nullelem)
        : coMultiHashBase<KEY, DATA>(nullelem)
    {
    }

    /// Destructor
    virtual ~coMultiHash()
    {
    }

    /// insert an entry
    int insert(const KEY &key, const DATA &inData)
    {
        return coMultiHashBase<KEY, DATA>::insert(key, inData);
    }

    /// remove an entry
    //int remove(const coHashIter<KEY,DATA> &iter )
    //  { return coMultiHashBase<KEY,DATA>::remove(iter.d_hashIndex); }
    // R.B. hash index not correct !!!
    int remove(const coHashIter<KEY, DATA> &iter)
    {
        return coMultiHashBase<KEY, DATA>::remove(iter.d_index + 1);
    }

    /// get element
    coHashIter<KEY, DATA> operator[](const KEY &key)
    {
        return coHashIter<KEY, DATA>(*this, getHash(key));
    }

    /// get first to step through
    coHashIter<KEY, DATA> first()
    {
        return coHashIter<KEY, DATA>(*this);
    }

    /// remove all elements
    void clear()
    {
        this->removeAll();
    }

    /// get number of entries currently in hash
    int getNumEntries() const
    {
        return coMultiHashBase<KEY, DATA>::getNumEntries();
    }

    /// get element (only use with preset NULL element!!!)
    const DATA &find(const KEY &key) const;

    /// get hash index, 0 if no element found
    unsigned long getHash(const KEY &key) const;
};

template <class KEY, class DATA>
INLINE const DATA &coMultiHash<KEY, DATA>::find(const KEY &key) const
{
    unsigned long hash = getHash(key);
    if (hash)
        return (*((const coMultiHashBase<KEY, DATA> *)(this)))[hash];
    else
        return this->getNullElem();
}

template <class KEY, class DATA>
INLINE unsigned long coMultiHash<KEY, DATA>::getHash(const KEY &key) const
{
    return coMultiHashBase<KEY, DATA>::getHash(key);
}
}
#endif
