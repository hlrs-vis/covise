/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CO_MULTI_HASH_BASE_H
#define __CO_MULTI_HASH_BASE_H

#include "coExport.h"

#include <assert.h>

namespace covise
{

template <class KEY, class DATA>
class coHashIter;
template <class KEY, class DATA>
class coHashBase;

/**
 *  Virtual base class implementing the double-hashing algorithm.<p>
 *  Requires: <ul>
 *     <li>class <b>KEY</b>  with default constructor + operator=
 *     <li>class <b>DATA</b> with default constructor + operator=
 *  </ul>
 *  Users must derive a class and define: <ul>
 *     <li>virtual unsigned int hash1(const KEY &)  1st level hash fkt.
 *     <li>virtual unsigned int hash2(const KEY &)  2nd level hash fkt.
 *  <ul>
 *  @author		Lars Frenzel, Andreas Werner
 *  @version		1.1
 *
 *  changes: getHash/nextHash/[]     AW
 *  EntryFlags 3 values for remove() AW
 *  flag values, keys, and entry flags now protected: R.B.
 */
template <class KEY, class DATA>
class coMultiHashBase
{

    friend class coHashIter<KEY, DATA>;
    friend class coHashBase<KEY, DATA>;

private:
    // Null element
    DATA d_nullElem;

    // the data associated to the keys
    DATA *data;

    // number of entries in the list
    unsigned int entries;

    // if entries>maxEntries then resize the list
    unsigned int maxEntries;

    // index of prime in the primeList
    int primeIndex;

    // resize the table if necessary
    void resize(void);

    // percentage of filling when to resize the table
    //static const float fillFactor;

    // +++++++++++ number of primes available
    enum
    {
        NUMPRIMES = 28
    };

    // +++++++++++ the primes themselves
    //static const unsigned long primeList[NUMPRIMES];

protected:
    // flag values for entry field
    enum
    {
        EMPTY = 0,
        PREVIOUS,
        USED
    };

    // keys
    KEY *keys;

    // list with flags if the associated entry is used or not
    // values: EMPTY, PREVIOUS(ly used), USED
    unsigned char *entryFlags;

    /// size of the list
    unsigned int size;

    /// the prime currently used as the length of the list
    unsigned int prime;

public:
    /// constructor
    coMultiHashBase();

    /// constructor
    coMultiHashBase(DATA nullelem);

    /// get the NULL element
    const DATA &getNullElem() const
    {
        return d_nullElem;
    };

    /// destructor
    virtual ~coMultiHashBase();

    /// insert an entry (virtual for non-multi hash)
    virtual int insert(const KEY &key, const DATA &inData);

    /// remove an entry by hashIndex
    int remove(unsigned long hashIndex);

    /// remove an entry by hashIndex
    void removeAll();

    /// get hash index, 0 if no element found
    unsigned long getHash(const KEY &key) const;

    /// access element by hash index: assert() correct index !!
    DATA &operator[](unsigned long hashIndex);

    /// access element by hash index: assert() correct index !!
    const DATA &operator[](unsigned long hashIndex) const;

    /// get next hashIndex to given hashIndex
    virtual unsigned long nextHash(unsigned long hashIndex) const;

    /// get number of entries currently in hash
    int getNumEntries() const;

    /// first hash function <em>(pure virtual)</em>
    virtual unsigned long hash1(const KEY &) const = 0;

    /// second hash function <em>(pure virtual)</em>
    virtual unsigned long hash2(const KEY &) const = 0;

    /// KEY1 == KEY2 operation <em>(pure virtual)</em>
    virtual bool equal(const KEY &, const KEY &) const = 0;
};

// +++++++++++ Create a hash table

template <class KEY, class DATA>
inline coMultiHashBase<KEY, DATA>::coMultiHashBase()
{
    static unsigned long primeList[NUMPRIMES] = {
        53ul, 97ul, 193ul, 389ul, 769ul,
        1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
        49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
        1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
        50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
        1610612741ul, 3221225473ul, 4294967291ul
    };

    // create the list
    primeIndex = 0;
    prime = primeList[primeIndex];
    size = prime;
    entries = 0;
    //fillFactor
    maxEntries = (unsigned int)(((float)size) * 0.75);

    // and allocate the key and data storage
    this->keys = new KEY[size];
    this->data = new DATA[size];
    entryFlags = new unsigned char[size];

    unsigned long i;

    for (i = 0; i < size; i++)
        entryFlags[i] = ((unsigned char)EMPTY);

    // done
    return;
}

// +++++++++++ Create a hash table

template <class KEY, class DATA>
inline coMultiHashBase<KEY, DATA>::coMultiHashBase(DATA nullelem)
{
    static unsigned long primeList[NUMPRIMES] = {
        53ul, 97ul, 193ul, 389ul, 769ul,
        1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
        49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
        1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
        50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
        1610612741ul, 3221225473ul, 4294967291ul
    };
    // create the list
    primeIndex = 0;
    prime = primeList[primeIndex];
    size = prime;
    entries = 0;
    d_nullElem = nullelem;
    //fillFactor
    maxEntries = (unsigned int)(((float)size) * 0.75);

    // and allocate the key and data storage
    this->keys = new KEY[size];
    this->data = new DATA[size];
    entryFlags = new unsigned char[size];

    unsigned long i;

    for (i = 0; i < size; i++)
    {
        entryFlags[i] = ((unsigned char)EMPTY);
        this->data[i] = d_nullElem;
    }

    // done
    return;
}

// +++++++++++ remove all elements from

template <class KEY, class DATA>
inline void coMultiHashBase<KEY, DATA>::removeAll()
{
    unsigned long i;

    for (i = 0; i < size; i++)
    {
        entryFlags[i] = ((unsigned char)EMPTY);
        this->data[i] = d_nullElem;
    }

    // done
    return;
}

// +++++++++++ Delete a hash table

template <class KEY, class DATA>
inline coMultiHashBase<KEY, DATA>::~coMultiHashBase()
{
    // clean up
    delete[] this->keys;
    delete[] this->data;
    delete[] entryFlags;
    return;
}

// +++++++++++ resize a hash table
// AW: only called if needed

template <class KEY, class DATA>
inline void coMultiHashBase<KEY, DATA>::resize(void)
{
    // save old table
    KEY *oldKeys = this->keys;
    DATA *oldData = this->data;
    unsigned int oldSize = size;
    unsigned char *oldFlags = entryFlags;

    static unsigned long primeList[NUMPRIMES] = {
        53ul, 97ul, 193ul, 389ul, 769ul,
        1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
        49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
        1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
        50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
        1610612741ul, 3221225473ul, 4294967291ul
    };
    // initialize new table
    primeIndex++;
    prime = primeList[primeIndex];
    size = prime;
    entries = 0L;
    //fillFactor
    maxEntries = (unsigned int)(((float)size) * 0.75);
    this->keys = new KEY[size];
    this->data = new DATA[size];
    entryFlags = new unsigned char[size];
    unsigned int i;
    for (i = 0; i < size; i++)
        entryFlags[i] = ((unsigned char)EMPTY);

    // now convert the old table into the new one
    for (i = 0L; i < oldSize; i++)
        if (oldFlags[i] == USED)
            insert(oldKeys[i], oldData[i]);

    // now clean up
    delete[] oldKeys;
    delete[] oldData;
    delete[] oldFlags;

    // done
    return;
}

// +++++++++++ insert an element a hash table

template <class KEY, class DATA>
inline int coMultiHashBase<KEY, DATA>::insert(const KEY &key, const DATA &inData)
{
    int secondHash = 0;
    unsigned int x = 0, u = 0;

    // find free space in the table
    x = hash1(key);
    while (entryFlags[x] == USED)
    {
        if (!secondHash)
        {
            u = hash2(key);
            secondHash = 1;
        }
        x = (x + u) % size;
    }

    // copy key and data to respective fields, save Flag info
    entries++;
    this->keys[x] = key;
    this->data[x] = inData;
    entryFlags[x] = (unsigned char)USED;

    // resize table if needed
    if (entries > maxEntries)
        resize();

    // done
    return (1);
}

// +++++++++++ remove an element from hash

template <class KEY, class DATA>
inline int coMultiHashBase<KEY, DATA>::remove(unsigned long idx)
{
    int res = 0;
    if ((idx) && (idx <= size))
    {
        idx--;
        if (entryFlags[idx] == USED)
        {
            entryFlags[idx] = PREVIOUS;
            entries--;
            res = 1;
        }
    }
    return res;
}

// +++++++++++ find an element in the hash table, 0 if not found

template <class KEY, class DATA>
inline unsigned long coMultiHashBase<KEY, DATA>::getHash(const KEY &key) const
{
    int secondHash = 0;
    unsigned int idx, u = 0;
    idx = hash1(key);
    while (((entryFlags[idx] == USED) && (!equal(this->keys[idx], key)))
           || (entryFlags[idx] == PREVIOUS))
    {
        if (!secondHash)
        {
            u = hash2(key);
            secondHash = 1;
        }
        idx = (idx + u) % size;
    }

    if (entryFlags[idx] != USED)
        idx = 0;
    else
        idx++; // one more to have 0 as error

    // done
    return idx;
}

// +++++++++++ return value for given hash value
template <class KEY, class DATA>
inline DATA &coMultiHashBase<KEY, DATA>::operator[](unsigned long idx)
{
    // make sure we got legal index
    assert((idx) && (idx <= size));
    return this->data[idx - 1];
}

// +++++++++++ return value for given hash value: const
template <class KEY, class DATA>
inline const DATA &coMultiHashBase<KEY, DATA>::operator[](unsigned long idx) const
{
    // make sure we got legal index
    assert((idx) && (idx <= size));
    return this->data[idx - 1];
}

// +++++++++++ get the next jash value
template <class KEY, class DATA>
inline unsigned long coMultiHashBase<KEY, DATA>::nextHash(unsigned long idx) const
{
    // make sure we got legal index
    assert((idx) && (idx <= size) && (entryFlags[idx - 1] == USED));

    // user idx'es are one too high
    idx--;

    // look out for next one with my key
    const KEY &myKey = this->keys[idx];
    unsigned long u = hash2(myKey);
    //cerr << "Checking further hashes, start with idx=" << idx
    //     << ", key is " << myKey << endl;
    idx = (idx + u) % size;
    //cerr << "+ Check " << idx << endl;
    while (((entryFlags[idx] == USED) && (!equal(this->keys[idx], myKey)))
           || (entryFlags[idx] == PREVIOUS))
    {
        idx = (idx + u) % size;
        //cerr << "+ Check " << idx << endl;
    }

    if (entryFlags[idx] != USED)
        idx = 0;
    else
        idx++;

    return idx;
}

template <class KEY, class DATA>
inline int coMultiHashBase<KEY, DATA>::getNumEntries() const
{
    return entries;
}
}
#endif
