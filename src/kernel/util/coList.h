/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef YAC_LIST_H
#define YAC_LIST_H

#include <string.h>
#include <iostream>
#include "coExport.h"

//template<class T>
//class coList;

#ifndef INLINE
#define INLINE inline
#endif

//=========================================================================
// DEFINITION
//=========================================================================

/**
  *   Templated list class.
  *
  *   @author       Franz Maurer
  *   @version      1.0
  *
  */

namespace covise
{

template <class T>
class coList
{

public:
    /// default constructor, list is empty
    coList(void);
    /// create new list with 'amount' elements
    coList(int amount);
    /// coList copy constructor
    coList(const coList<T> &v);
    /// coList destructor
    ~coList(void);

    // list access
    // -----------
    /// direct list access
    T &operator[](int index)
    {
        if (index < 0 || index > (dim - 1))
        {
            // cl_error( coWARNING, "coList", "[]", "Index out of bounds!" );
            return nullValue;
        }
        return (data[index]);
    }

    /// direct list access
    const T &operator[](int index) const
    {
        if (index < 0 || index > (dim - 1))
        {
            // cl_error( coWARNING, "coList", "const []", "Index out of bounds!" );
            return nullValue;
        }
        return (data[index]);
    }

    // assignment
    // ----------
    /// assign from another coList
    coList<T> &operator=(const coList<T> &v);
    /// set all elements of the current list to in_data
    void init(const T &in_data);
    /// clear the list (frees all data)
    void clear(void);
    /// set list element
    void set(int index, const T &in_data);

    // list size manipulation
    // ----------------------
    /// append element
    void append(const T &in_data);
    /// append element
    void append(const T *in_data)
    {
        append(*in_data);
    };
    /// insert element at the given list index
    void insert(int index, const T &in_data);
    /// replace element at the given index
    void replace(int index, const T &in_data);
    /// remove element at position index
    void remove(int index);
    /// remove last element
    T &removeLast()
    {
        if (dim > 0)
            return (data[--dim]);
        else
            return nullValue;
    };
    /// finds first occurence of element 'in_data' and delete it
    bool removeElement(const T &in_data);
    /// resize the list
    void newSize(int nelistw_size);
    /// fills the whole  with in_data
    void fillWith(const T &in_data);
    /// fills the whole  with in_data
    T *getLast()
    {
        return &(data[currentIndex]);
    };
    /// get first element and set current index to 0
    T *first()
    {
        if (dim == 0)
            return NULL;
        currentIndex = 0;
        return &(data[currentIndex]);
    };
    /// get last element and set current index to dim-1
    T *last()
    {
        if (dim == 0)
            return NULL;
        currentIndex = 0;
        return &(data[currentIndex]);
    };
    /// get next element
    T *next()
    {
        if (currentIndex > dim - 1)
            return NULL;
        currentIndex++;
        return &(data[currentIndex]);
    };
    /// get next element
    T *prev()
    {
        if (currentIndex <= 0)
            return NULL;
        dim--;
        return &(data[currentIndex]);
    };

    // list information
    // ----------------
    /// find the first occurrence of the specified element
    int find(const T &in_data) const;
    /// returns the number of list enries
    int entries(void) const
    {
        return (dim);
    };
    /// returns the number of list enries
    int getSize(void) const
    {
        return (dim);
    };
    /// print out the contents of the list
    void print(char *name) const;

private:
    /// increase allocated Size
    void incSize(void);
    T *data;
    int dim;
    int allocsize;
    int allocinc;
    int currentIndex;
    T nullValue;
};

//=========================================================================
// IMPLEMENTATION
//=========================================================================

template <class T>
INLINE coList<T>::coList(void)
{
    // create list
    data = 0L;
    dim = 0;
    allocsize = 0;
    allocinc = 20;
    nullValue = (T)0;
    currentIndex = 0;
}

template <class T>
INLINE coList<T>::coList(int amount)
{

    // create list
    nullValue = (T)0;
    dim = 0;
    allocsize = amount;
    allocinc = amount;
    data = new T[allocsize];
    currentIndex = 0;
    for (int i = 0; i < allocsize; i++)
        data[i] = 0L;
}

template <class T>
INLINE coList<T>::coList(const coList<T> &v)
{

    // copy list
    nullValue = (T)0;
    dim = v.dim;
    allocsize = v.allocsize;
    allocinc = v.allocinc;
    data = new T[allocsize];
    currentIndex = v.currentIndex;
    for (int i = 0; i < dim; i++)
        data[i] = v.data[i];
}

template <class T>
INLINE coList<T>::~coList(void)
{
    // free all memory
    if (data)
        delete[] data;
}

template <class T>
INLINE void coList<T>::append(const T &in_data)
{
    if (dim == allocsize)
        incSize();

    data[dim] = in_data;
    dim++;
}

template <class T>
INLINE void coList<T>::incSize()
{
    allocsize += allocinc;
    T *new_data = new T[allocsize];
    // copy old list
    for (int i = 0; i < dim; i++)
    {
        new_data[i] = data[i];
    }
    delete[] data;
    data = new_data;
}

template <class T>
INLINE void coList<T>::replace(int index, const T &in_data)
{

    if (index < 0 || index > (dim - 1))
        return;

    delete &data[index];
    data[index] = in_data;
}

template <class T>
INLINE void coList<T>::insert(int index, const T &in_data)
{
    int i;
    if (dim == allocsize)
        incSize();

    if (index >= dim || index < 0)
    {
        // cl_error( coWARNING, "coList", "insert", "Index out of bounds!" );
        return;
    }

    for (i = dim; i > index; i++)
        data[i] = data[i - 1];

    data[index] = in_data;

    dim++;
}

template <class T>
INLINE void coList<T>::remove(int index)
{
    int i;

    if (index < 0 || index > (dim - 1))
        return;
    for (i = index; i < dim - 1; i++)
        data[i] = data[i + 1];

    dim--;
    if (currentIndex > dim - 1)
        currentIndex = 0;
}

template <class T>
INLINE bool coList<T>::removeElement(const T &in_data)
{

    int index = find(in_data);
    if (index != -1)
    {
        remove(index);
        return true;
    }
    else
        return false;
}

template <class T>
INLINE int coList<T>::find(const T &in_data) const
{

    if (dim == 0)
    {
        return (-1);
    }

    for (int i = 0; i < dim; i++)
    {
        if (data[i] == in_data)
            return (i);
    }
    return (-1);
}

template <class T>
INLINE void coList<T>::newSize(int new_size)
{
    allocsize = new_size;
    T *new_data = new T[allocsize];
    // copy old list
    for (int i = 0; i < dim; i++)
    {
        new_data[i] = data[i];
    }
    delete[] data;
    data = new_data;
}

template <class T>
INLINE void coList<T>::fillWith(const T &in_data)
{

    for (int i = 0; i < dim; i++)
        data[i] = in_data;
}

template <class T>
INLINE void coList<T>::clear(void)
{

    delete[] data;
    data = 0L;
    dim = 0;
    allocsize = 0;
    currentIndex = 0;
}

template <class T>
INLINE coList<T> &coList<T>::operator=(const coList<T> &v)
{
    if (this != &v)
    {
        delete[] data;
        dim = v.dim;
        allocsize = dim + allocinc;
        currentIndex = v.currentIndex;
        data = new T[allocsize];
        for (int i = 0; i < dim; i++)
            data[i] = v.data[i];
    }
    return (*this);
}

template <class T>
INLINE void coList<T>::init(const T &in_data)
{

    for (int i = 0; i < dim; i++)
        data[i] = in_data;
}

template <class T>
INLINE void coList<T>::set(int index, const T &in_data)
{

    if (dim == 0 || index < 0 || index > (dim - 1))
    {
        // cl_error( coWARNING, "coList", "set", "Index out of bounds!" );
        return;
    }
    else
        data[index] = in_data;
}

template <class T>
INLINE void coList<T>::print(char *name) const
{
    int i = 0;

    std::cout << "*** coList '" << name << "' elements: ***\n";

    if (dim == 0)
    {
        std::cout << "empty list!\n";
        return;
    }

    for (i = 0; i < dim; i++)
    {
        std::cout << "[" << i << "]: " << data[i] << std::endl;
    }
    std::cout << "*** End of coList '" << name << "' ***\n";
}
}
#undef INLINE

#endif // YAC_LIST_H
