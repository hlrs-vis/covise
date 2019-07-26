/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_LIST_H
#define EC_LIST_H
//#include <stream.h>

/*
 $Log: covise_list.h,v $
 * Revision 1.2  1993/10/08  19:21:13  zrhk0125
 * destructor now deletes all members
 *
 * Revision 1.1  93/09/25  20:45:40  zrhk0125
 * Initial revision
 *
*/
#ifndef NULL
#define NULL nullptr
#endif
/***********************************************************************\ 
 **                                                                     **
 **   List class                                   Version: 1.01        **
 **                                                                     **
 **                                                                     **
 **   Description  : A templated class to handle lists.                 **
 **                  ListElement is an element of this list             **
 **                                                                     **
 **   Classes      : List, ListElement                                  **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93  Ver 1.01    current() added              **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/
namespace covise
{

template <class T>
class List;

template <class T>
class ListElement // element of the list
{
    friend class List<T>;

private:
    T *data; // pointer to data
    ListElement<T> *next; // pointer to next list element
    ListElement() // initialize with 0
    {
        data = NULL;
        next = NULL;
    };
    ListElement(T *d) // initialize with data
    {
        data = d;
        next = NULL;
    };
    ~ListElement(){};
};

template <class T>
class List // list class
{
private:
    ListElement<T> *list; // root of the list
    ListElement<T> *last; // last element in list
    ListElement<T> *iter; // iterator for next() requests
    int mElemCount; // Counter for number fo elements added

public:
    List()
    {
        list = last = iter = NULL;
        mElemCount = 0;
    };
    ~List();
    void add(T *d); // add new element
    void remove(T *d); // remove element
#if defined(__hpux) || defined(_SX)
    T *next();
#else
    T *next()
    {
        if (iter == NULL)
            return NULL;
        ListElement<T> *tmpptr;
        tmpptr = iter;
        iter = iter->next;
        return tmpptr->data;
    }; // get next element (increments iter)
#endif

    T *current()
    {
        if (iter != NULL)
            return iter->data;
        else
            return 0;
    }; // get current element (keeps iter)

    void reset()
    { // reset iter to begin of the list
        iter = list;
    };

    T *get_last() // get last element in list
    {
        T *res;
        if (last)
            res = last->data;
        else
            res = NULL;
        return res;
    };

    T *get_first() // get first element in list
    {
        T *res;
        if (list)
            res = list->data;
        else
            res = NULL;
        return res;
    };
    void clear();
    void print();
    T *at(int i)
    {
        if (list == NULL)
        {
            return NULL;
        }

        ListElement<T> *ptr = list;
        ListElement<T> *tmp = NULL;

        while (ptr->next != NULL && i > 0)
        {
            i--;
            tmp = ptr->next;
            ptr->next = tmp->next;
        }

        return ptr->data;
    };
    int count()
    {
        return mElemCount;
    };
};

#if defined(__hpux) || defined(_SX)
template <class T>
inline T *List<T>::next()
{
    if (iter == NULL)
        return NULL;

    ListElement<T> *tmpptr;
    tmpptr = iter;
    iter = iter->next;
    return tmpptr->data;
}
#endif

template <class T>
inline void List<T>::add(T *d)
{
    if (list == NULL)
    {
        list = new ListElement<T>(d);
        last = list;
        mElemCount++;
        return;
    }

    if (list->data == NULL)
    {
        list->data = d;
        last = list;
        mElemCount++;
        return;
    }

    ListElement<T> *ptr = new ListElement<T>(d);
    ListElement<T> *tmpptr = list;

    while (tmpptr->next != NULL)
        tmpptr = tmpptr->next;

    tmpptr->next = ptr;
    last = ptr;
    mElemCount++;

    return;
}

template <class T>
inline void List<T>::remove(T *d)
{
    if (list == NULL)
        return;

    if (list->data == d)
    {
        ListElement<T> *tmp = list;
        list = list->next;
        delete tmp;
        if (list == NULL)
            last = NULL;
        mElemCount--;
        return;
    }

    ListElement<T> *ptr = list;

    while (ptr->next != NULL)
    {
        if (ptr->next->data == d)
        {
            if (ptr->next == last)
                last = ptr;
            if (ptr->next == iter)
                iter = ptr;
            ListElement<T> *tmp = ptr->next;
            ptr->next = tmp->next;
            delete tmp;
            if (list == NULL)
                last = NULL;
            mElemCount--;
            return;
        }
        ptr = ptr->next;
    }
    if (list == NULL)
        last = NULL;
    return;
}

template <class T>
inline void List<T>::clear()
{
    ListElement<T> *tmp;
    ListElement<T> *ptr;
    ptr = list;
    while (ptr)
    {
        tmp = ptr->next;
        delete ptr;
        ptr = tmp;
    }
    list = NULL;
    last = NULL;
    iter = NULL;
    mElemCount = 0;
    return;
}

template <class T>
inline void List<T>::print()
{
    ListElement<T> *ptr = list;
    ptr->data->print();
    while ((ptr = ptr->next) != NULL)
        ptr->data->print();
}

template <class T>
List<T>::~List()
{
    ListElement<T> *tmp;
    while (list && list->next)
    {
        tmp = list->next;
        delete list;
        list = tmp;
        mElemCount = 0;
    }
}
}
#endif
