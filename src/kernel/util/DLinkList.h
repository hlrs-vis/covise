/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_DLINK_LIST_H
#define VR_DLINK_LIST_H

// **************************************************************************
//
//                            (C) 1996
//              Computer Centre University of Stuttgart
//                         Allmandring 30
//                       D-70550 Stuttgart
//                            Germany
//
//
//
// COVISE Basic VR Environment Library
//
//
//
// Author: D.Rantzau
// Date  : 04.05.96
// Last  :
// **************************************************************************

#ifndef NULL
#define NULL 0L
#endif

namespace covise
{

//==========================================================================
//
//==========================================================================
// struct needed to link items together
template <class T>
struct DLink
{
    T item;
    DLink<T> *prev;
    DLink<T> *next;
    DLink(const T &a)
        : item(a){};
    ~DLink()
    {
        delete item;
    }
};

//==========================================================================
//
//==========================================================================
template <class T>
class DLinkList
{

private:
    // the list, and number of items in list
    DLink<T> *head, *tail, *curr;
    int listItems;

public:
    int noDelete;
    // constructor and destructor
    DLinkList(void);
    virtual ~DLinkList(void);

    //
    // query status of this class
    //

    // number of items in this list
    int num(void)
    {
        return listItems;
    }

    // return TRUE if there is a current item
    int is_current(void)
    {
        return (curr != ((DLink<T> *)0));
    }

    //
    // routines for adding or removing items from the list
    //

    // add new item to end of list
    DLinkList<T> &append(const T &);

    // append new item after the current item
    DLinkList<T> &insert_after(const T &);

    // append new item before the current item
    DLinkList<T> &insert_before(const T &);

    // remove current item from list
    DLinkList<T> &remove(void);

    // remove last item from list
    DLinkList<T> &removeLast(void);

    //
    // routines to access a particular item in the list
    //

    // just return the current item, do not change which item is current
    T current(void);

    // return the Nth item; do not change current item position
    T item(int);

    // return the current item, and move the current item on to the next one
    T get(void)
    {
        T retval = current();
        next();
        return retval;
    }

    //
    // iterator routines:
    //

    // move the current item pointer to the Nth item
    DLinkList<T> &set(int);

    // move the current item pointer to the item which matches the given one
    // return TRUE if found, or FALSE otherwise
    int find(const T &);

    // reset the current item pointer to the beginning of the list
    DLinkList<T> &reset(void)
    {
        if (head)
            curr = head;
        return *this;
    }

    // move the current item pointer to the next item in list
    DLinkList<T> &next(void)
    {
        if (curr)
            curr = curr->next;
        return *this;
    }

    // move the current item pointer to the previous item in list
    DLinkList<T> &prev(void)
    {
        if (curr)
            curr = curr->prev;
        return *this;
    }

    // clear the current item pointer; make it null
    DLinkList<T> &clear(void)
    {
        curr = (DLink<T> *)0;
        return *this;
    }
};

//==========================================================================
//
//==========================================================================
template <class T>
DLinkList<T>::DLinkList(void)
{

    head = tail = curr = NULL;
    listItems = 0;
    noDelete = 0;
}

//==========================================================================
//
//==========================================================================
template <class T>
DLinkList<T>::~DLinkList(void)
{
    if (noDelete)
    {
        head = tail = curr = NULL;
        listItems = 0;
    }
    else
    {
        reset();
        while (curr)
            remove();
    }
}

//==========================================================================
// add new item to end of list
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::append(const T &a)
{
    DLink<T> *newlink = new DLink<T>(a);
    newlink->next = NULL;
    newlink->prev = tail;
    tail = newlink;
    if (newlink->prev)
        (newlink->prev)->next = newlink;
    else
        head = curr = newlink;
    listItems++;
    return *this;
}

//==========================================================================
//  append new item after the current item
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::insert_after(const T &a)
{
    if (curr) // yes, there is a current item
    {
        DLink<T> *newlink = new DLink<T>(a);
        newlink->next = curr->next;
        newlink->prev = curr;
        curr->next = newlink;
        if (newlink->next == NULL)
            tail = newlink;
        else
            (newlink->next)->prev = newlink;
        listItems++;
    }
    else // no current item; just append at end
        append(a);
    return *this;
}

//==========================================================================
//  append new item before the current item
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::insert_before(const T &a)
{
    if (curr) // yes, there is a current item
    {
        DLink<T> *newlink = new DLink<T>(a);
        newlink->next = curr;
        newlink->prev = curr->prev;
        curr->prev = newlink;
        if (newlink->prev == NULL)
            head = newlink;
        else
            (newlink->prev)->next = newlink;
        listItems++;
    }
    else // no current item; just append at end
        append(a);
    return *this;
}

//==========================================================================
//  remove current item from list
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::removeLast(void)
{
    DLink<T> *oldlink = tail;
    if (oldlink)
    {
        if (oldlink->prev)
            (oldlink->prev)->next = oldlink->next;
        if (oldlink->next)
            (oldlink->next)->prev = oldlink->prev;
        if (head == oldlink)
            head = oldlink->next;
        curr = tail = oldlink->prev;
        if (!noDelete)
            delete oldlink;
        listItems--;
    }
    return *this;
}
//==========================================================================
//  remove current item from list
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::remove(void)
{
    DLink<T> *oldlink = curr;
    if (oldlink)
    {
        if (oldlink->prev)
            (oldlink->prev)->next = oldlink->next;
        if (oldlink->next)
            (oldlink->next)->prev = oldlink->prev;
        if (head == oldlink)
            head = oldlink->next;
        if (tail == oldlink)
            tail = oldlink->prev;
        curr = oldlink->next;
        if (!noDelete)
            delete oldlink;
        listItems--;
    }
    return *this;
}

//==========================================================================
// return the Nth item; do not change current item position
//==========================================================================
template <class T>
T DLinkList<T>::item(int n)
{
    DLink<T> *link = head;
    while (link && n > 0)
    {
        link = link->next;
        n--;
    }
    if (link && n == 0)
        return link->item;
    else
        return head->item; // problem if NO items in list
}

//==========================================================================
// just return the current item, do not change which item is current
//==========================================================================
template <class T>
T DLinkList<T>::current(void)
{
    DLink<T> *link = curr;
    if (!link)
    {
        reset();
        return (NULL);
    }
    return link->item;
}

//==========================================================================
// move the current item pointer to the Nth item
//==========================================================================
template <class T>
DLinkList<T> &DLinkList<T>::set(int N)
{
    reset();
    for (int i = 0; i < N; i++)
        next();
    return *this;
}

//==========================================================================
// move the current item pointer to the item which matches the given one
// return TRUE if found, or FALSE otherwise
//==========================================================================
template <class T>
int DLinkList<T>::find(const T &a)
{
    reset();
    while (curr && curr->item != a)
        next();
    return (curr != NULL);
}
}
#endif
