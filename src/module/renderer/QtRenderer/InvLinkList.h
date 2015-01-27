/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef INV_LINK_LIST_H
#define INV_LINK_LIST_H
//#include <stream.h>

#ifndef NULL
#define NULL 0L
#endif

template <class T>
class InvLinkList;

template <class T>
class InvLinkListElement
{ // element of the list
    friend class InvLinkList<T>;

private:
    T *data; // pointer to data
    InvLinkListElement<T> *next; // pointer to next list element
    InvLinkListElement()
    {
        data = 0L;
        next = 0L;
    }; // initialize with 0
    InvLinkListElement(T *d)
    {
        data = d;
        next = 0L;
    }; // initialize with data
    ~InvLinkListElement(){};
};

template <class T>
class InvLinkList
{ // list class
    InvLinkListElement<T> *list; // root of the list
    InvLinkListElement<T> *last; // last element in list
    InvLinkListElement<T> *iter; // iterator for next() requests
public:
    InvLinkList()
    {
        list = last = iter = 0L;
    };
    ~InvLinkList();
    void add(T *d); // add new element
    void remove(T *d); // remove element
#if defined(__hpux) || defined(_SX)
    T *next();
#else
    T *next()
    {
        if (iter == 0L)
            return 0L;

        InvLinkListElement<T> *tmpptr;
        tmpptr = iter;
        iter = iter->next;
        return tmpptr->data;
    }; // get next element (increments iter)
#endif
    T *current()
    {
        if (iter != 0L)
            return iter->data;
        else
            return 0;
    }; // get current element (keeps iter)
    void reset()
    { // reset iter to begin of the list
        iter = list;
    };
    T *get_last()
    { // get last element in list
        T *res;
        if (last)
            res = last->data;
        else
            res = NULL;
        return res;
    };
    T *get_first()
    { // get first element in list
        T *res;
        if (list)
            res = list->data;
        else
            res = NULL;
        return res;
    };
    void clear();
    void print();
};

#if defined(__hpux) || defined(_SX)
template <class T>
T *InvLinkList<T>::next()
{
    if (iter == 0L)
        return 0L;

    InvLinkListElement<T> *tmpptr;
    tmpptr = iter;
    iter = iter->next;
    return tmpptr->data;
}
#endif

template <class T>
void InvLinkList<T>::add(T *d)
{
    if (list == 0L)
    {
        list = new InvLinkListElement<T>(d);
        last = list;
        return;
    }

    if (list->data == 0L)
    {
        list->data = d;
        last = list;
        return;
    }

    InvLinkListElement<T> *ptr = new InvLinkListElement<T>(d);
    InvLinkListElement<T> *tmpptr = list;

    while (tmpptr->next != 0L)
        tmpptr = tmpptr->next;

    tmpptr->next = ptr;
    last = ptr;

    return;
}

template <class T>
void InvLinkList<T>::remove(T *d)
{
    if (list == 0L)
        return;
    if (list->data == d)
    {
        InvLinkListElement<T> *tmp = list;
        list = list->next;
        delete tmp;
        if (list == NULL)
            last = NULL;
        return;
    }

    InvLinkListElement<T> *ptr = list;

    while (ptr->next != 0L)
    {
        if (ptr->next->data == d)
        {
            if (ptr->next == last)
                last = ptr;
            if (ptr->next == iter)
                iter = ptr;
            InvLinkListElement<T> *tmp = ptr->next;
            ptr->next = tmp->next;
            delete tmp;
            if (list == NULL)
                last = NULL;
            return;
        }
        ptr = ptr->next;
    }
    if (list == NULL)
        last = NULL;
    return;
}

template <class T>
void InvLinkList<T>::clear()
{
    InvLinkListElement<T> *tmp;
    InvLinkListElement<T> *ptr;

    ptr = list;
    while (ptr)
    {
        tmp = ptr->next;
        delete ptr;
        ptr = tmp;
    }
    list = 0L;
    last = 0L;
    iter = 0L;
    return;
}

template <class T>
void InvLinkList<T>::print()
{
    InvLinkListElement<T> *ptr = list;

    ptr->data->print();
    while (ptr = ptr->next)
        ptr->data->print();
}

template <class T>
InvLinkList<T>::~InvLinkList()
{
    InvLinkListElement<T> *tmp;

    while (list && list->next)
    {
        tmp = list->next;
        delete list;
        list = tmp;
    }
}

#endif
