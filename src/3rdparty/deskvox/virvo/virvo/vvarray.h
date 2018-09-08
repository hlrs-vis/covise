// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_ARRAY_H
#define VV_ARRAY_H

#include <limits>
#include <iostream>
#include <string.h>
#include "vvexport.h"
#include "vvmacros.h"
#ifdef max
#undef max
#undef min
#endif

/** Templated dynamic array class.<P>
  Example usage:
  <PRE>
  vvArray<int*> test;
  int i[] = {1,2,3};
  test.append(&i[0]);
  test.append(&i[1]);
  test.insert(1, &i[2]);
  test.remove(2);
  test.clear();
  </PRE>
@author Jurgen P. Schulze
*/

template<class T> class VV_DECL_DEPRECATED vvArray
{
  public:
    VV_DECL_DEPRECATED vvArray();
    VV_DECL_DEPRECATED vvArray(size_t, size_t);
    VV_DECL_DEPRECATED vvArray(const vvArray<T>&);
    VV_DECL_DEPRECATED ~vvArray();

    VV_DECL_DEPRECATED void clear();
    VV_DECL_DEPRECATED void set(size_t, const T&);
    VV_DECL_DEPRECATED T*   get(size_t);
    VV_DECL_DEPRECATED void append(const T&);
    VV_DECL_DEPRECATED void insert(size_t, const T&);
    VV_DECL_DEPRECATED void replace(size_t, const T&);
    VV_DECL_DEPRECATED void remove(size_t);
    VV_DECL_DEPRECATED void removeLast();
    VV_DECL_DEPRECATED bool removeElement(const T&);
    VV_DECL_DEPRECATED void resize(size_t);
    VV_DECL_DEPRECATED void setIncrement(size_t);
    VV_DECL_DEPRECATED void fill(const T&);
    VV_DECL_DEPRECATED T*   first();
    VV_DECL_DEPRECATED T*   last();
    VV_DECL_DEPRECATED size_t find(const T&);
    VV_DECL_DEPRECATED size_t count() const;
    VV_DECL_DEPRECATED void print(char*);
    VV_DECL_DEPRECATED T*   getArrayPtr();
    VV_DECL_DEPRECATED void deleteElementsNormal();
    VV_DECL_DEPRECATED void deleteElementsArray();

    /// Direct array access:
    VV_DECL_DEPRECATED T & operator[] (size_t index)
    {
      if (index>(usedSize-1))
      {
        return nullValue;
      }
      return data[index];
    }

    /// Direct array access:
    VV_DECL_DEPRECATED const T & operator[] (size_t index) const
    {
      if (index>(usedSize-1))
      {
        return nullValue;
      }
      return data[index];
    }

    /// Assign from another vvArray:
    VV_DECL_DEPRECATED vvArray<T> &operator =(const vvArray<T>& v);

  private:
    T*   data;                                    ///< actual data array
    size_t usedSize;                              ///< number of array elements actually used
    size_t allocSize;                             ///< number of array elements allocated in memory
    size_t allocInc;                              ///< number of array elements by which the array grows when increased
    T    nullValue;                               ///< NULL value to use as a return value

    void incSize();
};

//----------------------------------------------------------------------------
/// Default constructor. Array is empty, allocation increment is 10 elements.
template<class T> vvArray<T>::vvArray()
{
  nullValue = (T)0;
  allocSize = 0;
  allocInc  = 10;
  usedSize   = 0;
  data  = 0;
}

//----------------------------------------------------------------------------
/** Constructor for a new array with 'amount' initial elements and an
  array increment of 'inc'.
*/
template<class T> vvArray<T>::vvArray(size_t amount, size_t inc)
{
  nullValue = (T)0;
  allocSize = amount;
  allocInc = inc;
  usedSize = 0;
  data = new T[allocSize];
  for (size_t i=0; i<allocSize; ++i) data[i] = 0;
}

//----------------------------------------------------------------------------
/// Copy constructor.
template<class T> vvArray<T>::vvArray(const vvArray<T>& v)
{
  nullValue = (T)0;
  allocSize = v.allocSize;
  allocInc = v.allocInc;
  usedSize = v.usedSize;
  data = new T[allocSize];
  for (size_t i=0; i<usedSize; ++i) data[i] = v.data[i];
}

//----------------------------------------------------------------------------
/// Destructor: free all memory
template<class T> vvArray<T>::~vvArray()
{
  clear();
}

//----------------------------------------------------------------------------
/// Return pointer to data array.
template<class T> T* vvArray<T>::getArrayPtr()
{
  return data;
}

//----------------------------------------------------------------------------
/// Append element passed directly.
template<class T> void vvArray<T>::append(const T& in_data)
{
  if (usedSize == allocSize) incSize();
  data[usedSize] = in_data;
  ++usedSize;
}

//----------------------------------------------------------------------------
/// Replace element at the given index.
template<class T> inline void vvArray<T>::replace(size_t index, const T& newData)
{
  if (index>(usedSize - 1)) return;
  data[index] = newData;
}

//----------------------------------------------------------------------------
/** Insert element at the given array index. If index is out of bounds,
  nothing will be done.
*/
template<class T> inline void vvArray<T>::insert(size_t index, const T& in_data)
{
  if (usedSize == allocSize)
    incSize();
  if (index >= usedSize) return;

  for (size_t i=usedSize; i>index; --i)
    data[i] = data[i - 1];

  data[index] = in_data;
  usedSize++;
}

//----------------------------------------------------------------------------
/** Remove element from array.
  If the array is a list of pointers, the elements pointed to must be deleted separately!
  @param index index of element to remove (0 for first element, etc.)
*/
template<class T> inline void vvArray<T>::remove(size_t index)
{
  if (index>(usedSize - 1)) return;

  for (size_t i=index; i<usedSize-1; ++i)
    data[i] = data[i + 1];

  --usedSize;
}

//----------------------------------------------------------------------------
/** Remove last element. If array is empty, nothing happens.
  The allocated array size is not changed.
*/
template<class T> inline void vvArray<T>::removeLast()
{
  if (usedSize>0) --usedSize;
}

//----------------------------------------------------------------------------
/** Finds first occurrence of element 'in_data' and delete it.
  @return true if successful, otherwise false
*/
template<class T> inline bool vvArray<T>::removeElement(const T& in_data)
{
  size_t index = find(in_data);
  if (index < std::numeric_limits<size_t>::max())
  {
    remove(index);
    return true;
  }
  else return false;
}

//----------------------------------------------------------------------------
/** Returns the element at 'index'.
  If index is out of bounds, NULL is returned.
  The current index is set to the index of the returned element.
*/
template<class T> inline T* vvArray<T>::get(size_t index)
{
  if (index<0 || index>=usedSize) return &nullValue;

  return &(data[index]);
}

//----------------------------------------------------------------------------
/** Returns the last array element or NULL if array is empty.
  The current index is set to the last element.
*/
template<class T> inline T* vvArray<T>::last()
{
  if (usedSize>0)
  {
    return &(data[usedSize - 1]);
  }
  else return &nullValue;
}

//----------------------------------------------------------------------------
/// Get first element and set current index to 0
template<class T> inline T* vvArray<T>::first()
{
  if (usedSize==0) return NULL;
  return &(data[0]);
}

//----------------------------------------------------------------------------
/** Find the first occurrence of the specified element.
  @element element to find in array
  @return index of the desired element, or max size_t if element was not found
*/
template<class T> inline size_t vvArray<T>::find(const T& element)
{
  if (usedSize==0) return std::numeric_limits<size_t>::max();

  for (size_t i=0; i<usedSize; ++i)
  {
    if (data[i] == element) return i;
  }

  return -1;
}

//----------------------------------------------------------------------------
/** Resize the array.
  @param newSize new array size [elements]
*/
template<class T> void vvArray<T>::resize(size_t newSize)
{
  T* newData;

  newData = new T[allocSize];
  allocSize = newSize;

  // Copy old array:
  memcpy(newData, data, usedSize * sizeof(T));

  delete[] data;
  data = newData;
}

//----------------------------------------------------------------------------
/** Set the array incrementation size [elements].
 */
template<class T> void vvArray<T>::setIncrement(size_t inc)
{
  allocInc = inc;
}

//----------------------------------------------------------------------------
/// Fills the whole array with fillData.
template<class T> void vvArray<T>::fill(const T& fillData)
{
  for (size_t i=0; i<usedSize; ++i)
    data[i] = fillData;
}

//----------------------------------------------------------------------------
/// Clear the array (frees all data).
template<class T> void vvArray<T>::clear()
{
  delete[] data;
  data = 0;
  usedSize = 0;
  allocSize = 0;
}

//----------------------------------------------------------------------------
/** '=' copies the array.
  Only the array entries are copied, but not the elements the pointers
  point to in case of a pointer array!
*/
template<class T> vvArray<T> &vvArray<T>::operator =(const vvArray<T>& v)
{
  if (this != &v)
  {
    clear();
    usedSize = v.usedSize;
    allocSize = v.allocSize;
    data = new T[allocSize];
    memcpy(data, v.data, usedSize * sizeof(T));
  }

  return *this;
}

//----------------------------------------------------------------------------
/// Set array element. Any previously existing element will be overwritten.
template<class T> inline void vvArray<T>::set(size_t index, const T& newData)
{
  if (usedSize == 0 || index < 0 || index > (usedSize - 1)) return;
  data[index] = newData;
}

//----------------------------------------------------------------------------
/// Returns the number of array enries.
template<class T> inline size_t vvArray<T>::count() const
{
  return usedSize;
}

//----------------------------------------------------------------------------
/** Print out the contents of the array.
  @param title some string to print before the array
*/
template<class T> inline void vvArray<T>::print(char* title)
{
  std::cerr << title << std::endl;

  if (usedSize == 0)
  {
      std::cerr << "empty array" << std::endl;
    return;
  }

  for (size_t i=0; i<usedSize; ++i)
    std::cerr << "[" << i << "]: " << data[i] << std::endl;
}

//----------------------------------------------------------------------------
/** Increase allocated size: create a new array and copy all elements from
  the old array.
*/
template<class T> void vvArray<T>::incSize()
{
  allocSize += allocInc;
  T* newData = new T[allocSize];

  // Copy old array:
  memcpy(newData, data, usedSize * sizeof(T));

  delete[] data;
  data = newData;
}

//----------------------------------------------------------------------------
template<class T> void vvArray<T>::deleteElementsNormal()
{
  for (size_t i=0; i<usedSize; ++i)
  {
    delete data[i];
  }
}

//----------------------------------------------------------------------------
template<class T> void vvArray<T>::deleteElementsArray()
{
  for (size_t i=0; i<usedSize; ++i)
  {
    delete[] data[i];
  }
}
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
