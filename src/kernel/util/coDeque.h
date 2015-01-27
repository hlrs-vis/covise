/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef YAC_DEQUE_H
#define YAC_DEQUE_H

#include <string.h>
#include <iostream>

//=========================================================================
// DEFINITION
//=========================================================================

/**
  *   Templated double ended queue class.
  *
  *   @author       Franz Maurer
  *   @version      1.0
  *
  */
namespace covise
{

template <class T>
class coDeque
{
public:
    coDeque(void);
    ~coDeque(void);

    /// direct list access
    T &operator[](int index);
    /// direct list access
    const T &operator[](int index) const;

    /// push one element at the beginning of the queue
    void pushFront(const T &in_data);
    /// push one element at the end of the queue
    void pushBack(const T &in_data);
    /// pop one element from the beginning of the queue
    void popFront(void);
    /// pop one element from the end of the queue
    void popBack(void);

    /// get the error id if something goes wrong
    int isBad()
    {
        return error;
    };
    /// returns the size of the queue
    int size(void)
    {
        return dim;
    };
    /// print out the contents of the queue
    void print(void);

private:
    T *data;
    int dim;
    int error;
};

template <class T>
std::ostream &operator<<(std::ostream &str, const coDeque<T> &queue);

//=========================================================================
// IMPLEMENTATION
//=========================================================================

template <class T>
inline coDeque<T>::coDeque(void)
{
    dim = 0;
    data = NULL;
    error = 0;
}

template <class T>
inline coDeque<T>::~coDeque(void)
{

    // free all memory
    if (data)
        delete[] data;
}

//-------------------------------------------------------------------------
// Access functions
//-------------------------------------------------------------------------
template <class T>
inline const T &coDeque<T>::operator[](int index) const
{
    if (index < 0 || index > (dim - 1))
    {
        // out of bounds !
        static T dummy;
        ((coDeque<T> *)this)->error = 1;
        return dummy;
    }
    return (data[index]);
}

template <class T>
inline T &coDeque<T>::operator[](int index)
{
    if (index < 0 || index > (dim - 1))
    {
        // out of bounds !
        static T dummy;
        error = 1;
        return dummy;
    }
    return (data[index]);
}

template <class T>
inline void coDeque<T>::pushBack(const T &in_data)
{

    T *new_data = new T[dim + 1];
    if (dim)
        memcpy(new_data, data, dim * sizeof(T));
    new_data[dim] = in_data;
    delete[] data;
    data = new_data;
    dim++;
}

template <class T>
inline void coDeque<T>::pushFront(const T &in_data)
{

    T *new_data = new T[dim + 1];
    new_data[0] = in_data;
    if (dim)
        memcpy(&new_data[1], data, dim * sizeof(T));
    delete[] data;
    data = new_data;
    dim++;
}

template <class T>
inline void coDeque<T>::popBack(void)
{

    if (dim == 0)
        return;

    if (dim == 1)
    {
        delete[] data;
        data = 0L;
        dim = 0;
        return;
    }

    T *new_data = new T[dim - 1];
    memcpy(new_data, data, (dim - 1) * sizeof(T));
    delete[] data;
    data = new_data;
    dim--;
}

template <class T>
inline void coDeque<T>::popFront(void)
{

    if (dim == 0)
        return;

    if (dim == 1)
    {
        delete[] data;
        data = 0L;
        dim = 0;
        return;
    }

    T *new_data = new T[dim - 1];
    memcpy(new_data, &data[1], (dim - 1) * sizeof(T));
    delete[] data;
    data = new_data;
    dim--;
}

//-------------------------------------------------------------------------
// print out contents
//-------------------------------------------------------------------------
template <class T>
inline void coDeque<T>::print(void)
{
    std::cout << "coDeque contains:" << std::endl;
    if (!dim)
    {
        std::cout << "Nothing!" << std::endl;
    }
    else
        for (int i = 0; i < dim; i++)
            std::cout << "[" << i << "]: " << data[i] << std::endl;
}

template <class T>
inline std::ostream &operator<<(std::ostream &str, const coDeque<T> &queue)
{
    str << "coDeque contains:" << std::endl;
    if (!queue->dim)
    {
        str << "Nothing!" << std::endl;
    }
    else
        for (int i = 0; i < queue->dim; i++)
            str << "[" << i << "]: " << queue->data[i] << std::endl;

    return str;
}
}
#endif // YAC_DEQUE_H
