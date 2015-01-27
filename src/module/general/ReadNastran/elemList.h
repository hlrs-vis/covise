/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ELEMLIST_H
#define ELEMLIST_H

#include <util/coviseCompat.h>

//=========================================================================
// DEFINITION
//=========================================================================

/**
 * class ElementList
 * Simple templated list class.
 * This is an adapted version of the COVISE/src/util/coList.h
 *
 * @author      Franz Maurer
 * @date        5/15/98
 * @version     0.2
 **/
template <class T>
class ElementList
{

public:
    /// default constructor, with inital size and alloc increment parameters
    ElementList(int num = 0, int inc = 20000);
    /// ElementList copy constructor
    ElementList(const ElementList<T> &v);
    /// ElementList destructor
    ~ElementList();

    /// direct list access
    const T &operator[](int index) const
    {
        if (index < 0 || index > (dim - 1))
        {
#ifdef VERBOSE
            cerr << "WARNING in ElementList::operator[]: Index out of bounds!" << endl;
#endif
            return nullValue;
        }
        return (data[index]);
    }
    /// assign from another ElementList
    ElementList<T> &operator=(const ElementList<T> &v);

    /// add an element
    void add(const T &in_data);
    /// clear the list (frees all data)
    void clear();
    /// find the first occurrence of the specified element
    int find(const T &in_data);
    /// performs binary search on sorted data (only standard data types, like int are supported, and list must be sorted in ascending order)
    int findBinary(const T &in_data);
    /// get last element
    T &get(int index) const
    {
        return (data[index]);
    };
    /// get a pointer to the data
    T *getDataPtr()
    {
        return data;
    };
    /// get a pointer to the data at position index
    T *getDataPtr(int index)
    {
        return &data[index];
    };
    /// init the whole list with the given value
    void init(const T &value);
    /// insert element at the given list index
    void insert(int index, const T &in_data);
    /// resize the list
    void newSize(int new_size);
    /// print out the contents of the list
    void print(const char *str = 0) const;
    /// remove element at position index
    int remove(int index);
    /// finds first occurence of element 'in_data' and delete it
    int removeElement(const T &in_data);
    /// replace element at the given index
    void replace(int index, const T &in_data);
    /// set list element
    void set(int index, const T &in_data);
    /// set all elements of the current list to in_data
    void setAll(const T &in_data);
    /// returns the number of list entries
    int size()
    {
        return dim;
    };

private:
    /// increase allocated size
    void incSize();

    T *data;
    int dim;
    int allocsize;
    int allocinc;
    T nullValue;
};

//=========================================================================
// IMPLEMENTATION
//=========================================================================

template <class T>
ElementList<T>::ElementList(int num, int inc)
{

    dim = num;
    if (dim)
        data = new T[dim];
    else
        data = 0L;
    allocsize = dim;
    allocinc = inc;
    nullValue = (T)0;
}

template <class T>
ElementList<T>::ElementList(const ElementList<T> &v)
{

    // copy list
    nullValue = (T)0;
    dim = v.dim;
    allocsize = v.allocsize;
    allocinc = v.allocinc;
    data = new T[allocsize];
    for (int i = 0; i < dim; i++)
        data[i] = v.data[i];
}

template <class T>
ElementList<T>::~ElementList(void)
{
    // free all memory
    if (data)
        delete[] data;
}

template <class T>
ElementList<T> &ElementList<T>::operator=(const ElementList<T> &v)
{
    if (this != &v)
    {
        delete[] data;
        dim = v.dim;
        allocsize = dim + allocinc;
        data = new T[allocsize];
        for (int i = 0; i < dim; i++)
            data[i] = v.data[i];
    }
    return (*this);
}

template <class T>
void ElementList<T>::add(const T &in_data)
{
    if (dim == allocsize)
        incSize();

    data[dim] = in_data;
    dim++;
}

template <class T>
void ElementList<T>::clear()
{

    delete[] data;
    data = 0L;
    dim = 0;
    allocsize = 0;
}

template <class T>
int ElementList<T>::find(const T &in_data)
{

    if (dim == 0)
        return -1;

    for (int i = 0; i < dim; i++)
    {
        if (data[i] == in_data)
            return i;
    }
    return -1;
}

template <class T>
int ElementList<T>::findBinary(const T &in_data)
{
    int l = 0, r = dim, x;

    while (r >= l)
    {
        x = (l + r) / 2;
        if (in_data < data[x])
            r = x - 1;
        else if (in_data > data[x])
            l = x + 1;
        if (in_data == data[x])
            return x;
    }
    return -1;
}

template <class T>
void ElementList<T>::incSize()
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
void ElementList<T>::init(const T &value)
{
    for (int i = 0; i < dim; i++)
        data[i] = value;
}

template <class T>
void ElementList<T>::insert(int index, const T &in_data)
{

    if (dim == allocsize)
        incSize();

    if (index >= dim || index < 0)
    {
#ifdef VERBOSE
        cerr << "WARNING in ElementList::insert(): Index out of bounds!" << endl;
#endif
        return;
    }

    for (int i = dim; i > index; i--)
        data[i] = data[i - 1];

    data[index] = in_data;
    dim++;
}

template <class T>
void ElementList<T>::newSize(int new_size)
{
    allocsize = new_size;
    T *new_data = new T[allocsize];

    for (int i = 0; i < dim; i++)
    {
        new_data[i] = data[i];
    }
    dim = allocsize;

    delete[] data;
    data = new_data;
}

template <class T>
void ElementList<T>::print(const char *str) const
{

    cout << "---------------------------------------" << endl;
    if (str)
        cout << "ElementList '" << str << "':" << endl;
    else
        cout << "ElementList:" << endl;
    cout << "---------------------------------------" << endl;

    if (dim == 0)
    {
        cout << "empty list!\n";
        cout << "---------------------------------------" << endl;
        return;
    }

    for (int i = 0; i < dim; i++)
    {
        cout << "[" << i << "]: " << data[i] << endl;
    }
    cout << "---------------------------------------" << endl;
}

template <class T>
int ElementList<T>::remove(int index)
{

    if (index >= dim || index < 0)
    {
#ifdef VERBOSE
        cerr << "WARNING in ElementList::remove(): Index out of bounds!" << endl;
#endif
        return false;
    }

    for (int i = index; i < dim - 1; i++)
        data[i] = data[i + 1];

    dim--;
    return true;
}

template <class T>
int ElementList<T>::removeElement(const T &in_data)
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
void ElementList<T>::replace(int index, const T &in_data)
{

    if (index < 0 || index > (dim - 1))
    {
#ifdef VERBOSE
        cerr << "WARNING in ElementList::replace(): Index out of bounds!" << endl;
#endif
        return;
    }

    delete &data[index];
    data[index] = in_data;
}

template <class T>
void ElementList<T>::set(int index, const T &in_data)
{

    if (index >= dim || index < 0)
    {
#ifdef VERBOSE
        cerr << "WARNING in ElementList::set(): Index out of bounds!" << endl;
#endif
        return;
    }
    else
        data[index] = in_data;
}

template <class T>
void ElementList<T>::setAll(const T &in_data)
{

    for (int i = 0; i < dim; i++)
        data[i] = in_data;
}
#endif // ELEMLIST_H
