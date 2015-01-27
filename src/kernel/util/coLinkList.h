/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_LINK_LIST_H
#define COVISE_LINK_LIST_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description: Interface classes for application modules to the COVISE   **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1997 RUS                                **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: D. Rantzau                                                     **
 ** Date:  15.08.97  V1.0                                                  **
\**************************************************************************/

#include "coExport.h"
#include <cstdlib>

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
};

//==========================================================================
//
//==========================================================================
template <class T>
class coDLinkList
{

private:
    // the list, and number of items in list
    DLink<T> *head, *tail, *curr;
    int listItems;

public:
    // constructor and destructor

    coDLinkList(void);
    ~coDLinkList(void)
    {
        clear();
    };

    //
    // query status of this class
    //

    // number of items in this list
    int length(void)
    {
        return listItems;
    };

    // size of list
    int size(void)
    {
        return length();
    };

    // return true if there is a current item
    bool is_current(void)
    {
        return (curr != ((DLink<T> *)0));
    };

    // checks if list is empty or not
    int empty(void)
    {
        return head == 0;
    };

    //
    // routines for adding or removing items from the list
    //

    // add new item to end of list
    coDLinkList<T> &append(const T &);

    // append new item after the current item
    coDLinkList<T> &insertAfter(const T &);

    // append new item before the current item
    coDLinkList<T> &insertBefore(const T &);

    // remove current item from list
    coDLinkList<T> &remove(void);

    DLink<T> *first() const
    {
        return head;
    }
    DLink<T> *first_item() const
    {
        return head;
    }
    DLink<T> *last() const
    {
        return tail;
    }
    DLink<T> *last_item() const
    {
        return tail;
    }

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
    };

    T operator[](int i)
    {
        return item(i);
    };

    //
    // iterator routines:
    //

    // move the current item pointer to the Nth item
    coDLinkList<T> &set(int);

    // move the current item pointer to the item which matches the given one
    // return true if found, or false otherwise
    bool search(const T &);

    // reset the current item pointer to the beginning of the list
    coDLinkList<T> &reset(void)
    {
        if (head)
            curr = head;
        return *this;
    };

    // move the current item pointer to the next item in list
    coDLinkList<T> &next(void)
    {
        if (curr)
            curr = curr->next;
        return *this;
    };

    // move the current item pointer to the previous item in list
    coDLinkList<T> &prev(void)
    {
        if (curr)
            curr = curr->prev;
        return *this;
    };

    // clear the current item pointer; make it null
    coDLinkList<T> &clear(void)
    {
        curr = (DLink<T> *)0;
        return *this;
    };
};

//==========================================================================
//
//==========================================================================
template <class T>
coDLinkList<T>::coDLinkList(void)
{

    head = tail = curr = NULL;
    listItems = 0;
}

//==========================================================================
//
//==========================================================================
//template<class T> coDLinkList<T>::~coDLinkList(void)
//{
// reset();
//  while(curr)
//    remove();
//}

//==========================================================================
// add new item to end of list
//==========================================================================
template <class T>
coDLinkList<T> &coDLinkList<T>::append(const T &a)
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
coDLinkList<T> &coDLinkList<T>::insertAfter(const T &a)
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
coDLinkList<T> &coDLinkList<T>::insertBefore(const T &a)
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
coDLinkList<T> &coDLinkList<T>::remove(void)
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
        delete oldlink;
        listItems--;
    }
    return *this;
}

//==========================================================================
// return the Nth item; do not change current item position
//==========================================================================
template <class T>
T coDLinkList<T>::item(int n)
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
T coDLinkList<T>::current(void)
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
coDLinkList<T> &coDLinkList<T>::set(int N)
{
    reset();
    for (int i = 0; i < N; i++)
        next();
    return *this;
}

//==========================================================================
// move the current item pointer to the item which matches the given one
// return true if found, or false otherwise
//==========================================================================
template <class T>
bool coDLinkList<T>::search(const T &a)
{
    reset();
    while (curr && compare(curr->item, a) == false)
        next();
    return (curr != NULL);
}
}
#endif //COVISE_LINK_LIST_H
