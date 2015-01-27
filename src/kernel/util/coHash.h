/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_HASH_H_
#define _CO_HASH_H_

// 27.02.98

#include "coHashBase.h"
#include "coHashIter.h"

#ifndef INLINE
#define INLINE inline
#endif

namespace covise
{

template <class KEY, class DATA>
class coHashBase;

/**
 * Class
 *
 */
template <class KEY, class DATA>
class coHash : protected coHashBase<KEY, DATA>
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    coHash(const coHash &);

    /// Assignment operator: NOT  IMPLEMENTED
    coHash &operator=(const coHash &);

public:
    /// Default constructor
    coHash()
        : coHashBase<KEY, DATA>(){};

    /// Default constructor with NULL element
    coHash(const DATA &nullelem)
        : coHashBase<KEY, DATA>(nullelem){};

    /// Destructor
    virtual ~coHash(){};

    /// insert an entry
    int insert(const KEY &key, const DATA &inData)
    {
        return coHashBase<KEY, DATA>::insert(key, inData);
    }

    /// remove an entry
    int remove(const coHashIter<KEY, DATA> &iter)
    {
        {
            return coMultiHashBase<KEY, DATA>::remove(iter.d_hashIndex);
        }
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
INLINE const DATA &coHash<KEY, DATA>::find(const KEY &key) const
{
    unsigned long hash = getHash(key);
    if (hash)
        return (*((const coHashBase<KEY, DATA> *)(this)))[hash];
    else
        return this->getNullElem();
}

template <class KEY, class DATA>
INLINE unsigned long coHash<KEY, DATA>::getHash(const KEY &key) const
{
    return coHashBase<KEY, DATA>::getHash(key);
}
}
#endif
