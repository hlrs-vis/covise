/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _YAC_DL_LIST_H
#define _YAC_DL_LIST_H

#define LOGWARNING printf
// this is needed for NULL
#include <stddef.h>
#include <list>

//    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  ++                                                                      ++
// ++                                                                        ++
// ++ Description: Double-Linked list class                                  ++
// ++                                                                        ++
// ++   class coDLListElem<T>  Element of type T  + prev/next pointer        ++
// ++   class coDLList<T>      List of Elements of type T                    ++
// ++   class coDLListIter<T>  Iterator working on coDLList                  ++
// ++   class coDLPtrList<T>   List of T elements, which can be deleted on   ++
// ++                          remove: MUST BE POINTERS !!!!!                ++
// ++                                                                        ++
// ++                             (C)1999 RUS                                ++
// ++                Computing Center University of Stuttgart                ++
// ++                            Allmandring 30                              ++
// ++                            70550 Stuttgart                             ++
// ++    Author: A. Werner                                                   ++
//  ++   Date:  15.08.97  V1.0                                              ++
//   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template <class T>
class coDLListIter;

template <class T>
class coDLListSafeIter;

#define INLINE inline

/* struct needed to link items together
 * @author Andreas Werner
 */
template <class T>
struct coDLListElem
{
    // the data
    T item;

    // pointer to previous and next element
    coDLListElem<T> *prev, *next;

    // constructor with given data element
    coDLListElem(const T &data)
    {
        item = data;
        prev = next = NULL;
    }
};

template <class T>
class coDLListCompare
{
public:
    virtual bool equal(const T &op1, const T &op2) const
    {
        return op1 == op2;
    }
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++
// ++++++    class coDLList<T>
// ++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/**
 * Template coDLList : Double linked list of T Elements (not pointers)
 * @author Andreas Werner, 23.03.1999
 * @see coDLList
 * @see coDLListIter
 * @see coDLPtrList
 */
template <class T>
class coDLList
{
    /// friendly iterator
    friend class coDLListIter<T>;
    friend class coDLListSafeIter<T>;

protected:
    // the this->head and this->tail of the list
    coDLListElem<T> *head, *tail;

    // number of items in list
    int listItems;

    // for "safe" iterators
    std::list<coDLListSafeIter<T> *> m_iterators;

    void addIter(coDLListSafeIter<T> *iter)
    {
        m_iterators.push_back(iter);
    }

    void removeIter(coDLListSafeIter<T> *iter)
    {
        if (m_iterators.empty())
            return;
        m_iterators.remove(iter);
    }

    void invalidateIterators(void)
    {
        if (m_iterators.empty())
            return;
        typename std::list<coDLListSafeIter<T> *>::iterator iter;
        LOGWARNING("Invalidating iterators !!! (%zd iterators attached)", m_iterators.size());
        for (iter = m_iterators.begin(); iter != m_iterators.end(); iter++)
        {
            (*iter)->d_actElem = NULL;
        }
    }

    void correctIteratorsFor(coDLListElem<T> *whichElem)
    {
        if (m_iterators.empty())
            return;
        typename std::list<coDLListSafeIter<T> *>::iterator iter;
        for (iter = m_iterators.begin(); iter != m_iterators.end(); iter++)
        {
            if ((*iter)->d_actElem == whichElem)
            {
                if (m_iterators.size() > 1) // no warning if there is only one iterator
                {
                    LOGWARNING("Adjusting iterators in list 0x%p!", this);
                }
                (*iter)->backstep();
            }
        }
    }

public:
    /// constructor: construct empty list
    coDLList(void);

    /// destructor : virtual for all derived classes
    virtual ~coDLList(void);

    // +++++++++++++ status query functions ++++++++++++++++++++++++++++++++++

    /// number of items in this list
    int num(void) const
    {
        return this->listItems;
    }

    /// are there elements in my list ?
    operator bool()
    {
        return (this->listItems != 0);
    }

    // +++++++++++++ adding or removing items from the list ++++++++++++++++++

    /// remove specific item from list : virtual for coDLPtrList
    virtual void remove(coDLListElem<T> *whichElem);

    /// add new item to end of list
    coDLList<T> &append(const T &);

    // +++++++++++++ access list +++++++++++++++++++++++++++++++++++++++++++++

    /// return the Nth item: prefer usage of [] or iterator
    T item(int); // USE  BRACKET OPERATOR !!!!!

    /// return the Nth item
    T &operator[](int);

    /// return the Nth item, const version
    const T &operator[](int) const;

    /// cleanuop everything
    virtual void clean();

    // +++++++++++++ iterator routines ++++++++++++++++++++++++++++++++++++++

    /// get an Iterator placed on the first element
    coDLListIter<T> first();

    /// get an Iterator placed on the last element
    coDLListIter<T> last();

    /// get an Iterator placed to specific element
    coDLListIter<T> findElem(const T &);

    /// get an Iterator placed to specific element with compare-object
    coDLListIter<T> findElem(const T &, const coDLListCompare<T> &comp);

    /// get an Iterator placed to specific element number
    coDLListIter<T> findElem(int i);

    // +++++++++++++ access internal pointers ++++++++++++++++++++++++++++++++

    /// get the head ot the chain: primarily used for defining own iterators
    coDLListElem<T> *getHeadStruct()
    {
        return this->head;
    }

    /// get the tail ot the chain: primarily used for defining own iterators
    coDLListElem<T> *getTailStruct()
    {
        return this->tail;
    }
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++
// ++++++    class coDLPtrList<T> : public coDLList<T*>
// ++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/** List of pointers to class T with possible deleting on remove/deconstruct
 * @author Andreas Werner, 23.03.1999
 * @see coDLList
 * @see coDLListIter
 * @see coDLPtrList
 */
template <class T>
class coDLPtrList : public coDLList<T>
{
private:
    bool d_doDelete;

    // 'current' element -> for old-style list access : only for Pointer Lists
    coDLListElem<T> *curr;

    // NULL element : needed for old-style calls as "empty" return value
    T d_nullElem;

public:
    /** construct empty list, delete by default
       * @param  doDelete  flag whether to delete, default=true
       */
    coDLPtrList(bool doDelete = true)
        : coDLList<T>()
    {
        d_doDelete = doDelete;
        d_nullElem = NULL;
        curr = NULL;
    }

    /// set the return value for 'current' in case of NO-return
    void setNullElem(const T &nullElem)
    {
        d_nullElem = nullElem;
    }

    /** set to no deletion:
       *  @param  noDelete  flag whether to delete, default=false
       */
    void setNoDelete(bool doDelete = false)
    {
        d_doDelete = doDelete;
    }

    /// remove specific item from list: delete if doDelete=true
    virtual void remove(coDLListElem<T> *whichElem);

    /// cleanup everything
    virtual void clean();

    /// destructor: delete elements if doDelete=true
    virtual ~coDLPtrList()
    {
        coDLListElem<T> *elem = this->head;
        while (elem)
        {
            coDLListElem<T> *nextElem;
            nextElem = elem->next;
            if (d_doDelete)
            {
                delete elem->item;
            }
            elem = nextElem;
        }
    };

    /// append an element
    coDLPtrList<T> &append(const T &a);

    // +++++++++++++ obsolete old-time stuff - do not use ++++++++++++++++++++

    /// remove current item from list: <b>obsolete</b>, use iterators
    void remove()
    {
        LOGWARNING("coDLPtrList::remove(void) called in 0x%x ! This method is obsolete !!!", this);
        if (curr)
        {
            coDLListElem<T> *oldCurr = curr;
            curr = curr->next;
            remove(oldCurr);
        }
    }

    /// access 'current' item: <b>obsolete</b>, use iterators
    T current(void);

    /// just return last item, do not change current: <b>obsolete</b>, use iterators
    //T getLast(void);

    /// return the current item and move current to the next<b>obsolete</b>, use iterator
    T get(void)
    {
        T retval = current();
        next();
        return retval;
    }

    /// move the current to the Nth item: <b>obsolete</b>, use iterators
    coDLPtrList<T> &set(int);

    /** move the current item pointer to the item which matches the given one
       * return TRUE if found, or FALSE otherwise: <b>obsolete</b>, use iterators
       */
    int find(const T &);

    /// reset current to the beginning of the list: <b>obsolete</b>, use iterators
    coDLList<T> &reset(void)
    {
        if (this->head)
            curr = this->head;
        else
            curr = NULL;
        return *this;
    }

    /// move the current to next item in list: <b>obsolete</b>, use iterators
    coDLList<T> &next(void)
    {
        if (curr)
            curr = curr->next;
        return *this;
    }

    /***
      /// move current to previous item in list: <b>obsolete</b>, use iterators
      coDLList<T>& prev(void)
      {
         if (curr)
            curr = curr->prev;
         return *this;
      }

      /// clear current pointer; make it null: <b>obsolete</b>, use iterators
      coDLList<T>& clear(void)
      {
         curr = (coDLListElem<T> *)0;
         return *this;
      }

      /// return TRUE if there is a current item: <b>obsolete</b>, use iterators
      int is_current(void)
      {
         return (curr != ((coDLListElem<T> *)0));
      }
      ***/
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++
// ++++++    class coDLListIter<T>
// ++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/** Iterator working on coDLList class: make coDLList work
 *  multiple-call-safe
 * @author Andreas Werner, 23.03.1999
 * @see coDLList
 * @see coDLListIter
 * @see coDLPtrList
 */
template <class T>
class coDLListIter
{
protected:
    // the list we are running on
    coDLList<T> *d_myList;

    // my active element
    coDLListElem<T> *d_actElem;

    // this might be overridden by derived class for selective Iterators

    // advance pointer by one element
    virtual void advance()
    {
        if (d_actElem)
            d_actElem = d_actElem->next;
    }

    // step back one step
    virtual void backstep()
    {
        if (d_actElem)
            d_actElem = d_actElem->prev;
    }

    // forward to next valid element if current is not valid
    virtual void nextValid(){};

    // forward to prev valid element if current is not valid
    virtual void prevValid(){};

public:
    /// create empty iterator
    coDLListIter<T>()
    {
        d_myList = NULL;
        d_actElem = NULL;
    }

    /// create from a given List: set to first element
    coDLListIter<T>(coDLList<T> &list)
    {
        d_myList = &list;
        d_actElem = list.head;
        nextValid();
    }

    /// standard assignment operator
    coDLListIter<T> &operator=(const coDLListIter<T> &old)
    {
        d_myList = old.d_myList;
        d_actElem = old.d_actElem;
        nextValid();
        return *this;
    }

    /// standard copy-Constructor
    coDLListIter<T>(const coDLListIter<T> &old)
    {
        d_myList = old.d_myList;
        d_actElem = old.d_actElem;
        nextValid();
    }

    /// create from a given List + Pointer
    coDLListIter<T>(coDLList<T> &list, coDLListElem<T> *actElem)
    {
        d_myList = &list;
        d_actElem = actElem;
        nextValid();
    }

    /// Destructor
    virtual ~coDLListIter(){};

    /// check wether we point to an existing element:  if (iter) ...
    operator bool()
    {
        return (d_actElem != NULL);
    }

    /// go to next element : ++iter
    void operator++()
    {
        advance();
    }

    /// go to next element : iter++
    void operator++(int)
    {
        advance();
    }

    /// go to previous element : --iter
    void operator--()
    {
        backstep();
    }

    /// go to previous element : iter--
    void operator--(int)
    {
        backstep();
    }

    /// access object with  iter(): if (iter)=false: whatever might be in
    T &operator()()
    {
        T *res;
        if (d_actElem)
            res = &(d_actElem->item);
        else
        {
            static T dummy;
            res = &dummy;
            LOGWARNING("!!!! Accessing dummy-elem with: T &coDLListIter<T>::operator(); !!!!");
        }
        return *res;
    }

    /// access object with (*iter)
    T &operator*()
    {
        T *res;
        if (d_actElem)
            res = &(d_actElem->item);
        else
        {
            static T dummy;
            res = &dummy;
            LOGWARNING("!!!! Accessing dummy-elem with: T &coDLListIter<T>::operator*(); !!!!");
        }
        return *res;
    }

    /// access object with iter->something: only if T is a pointer
    T operator->()
    {
        T *res;
        if (d_actElem)
            res = &(d_actElem->item);
        else
        {
            static T dummy;
            res = &dummy;
            LOGWARNING("!!!! Accessing dummy-elem with: T coDLListIter<T>::operator->(); !!!!");
        }
        return *res;
    }

    /// set to first element
    void setFirst()
    {
        if (d_myList)
        {
            d_actElem = d_myList->head;
            nextValid();
        }
    }

    /// set to last element
    void setLast()
    {
        if (d_myList)
        {
            d_actElem = d_myList->tail;
            prevValid();
        }
    }

    /// add new element after my position
    coDLListIter<T> &insertAfter(const T &newElem);

    /// add new element after my position
    coDLListIter<T> &insertBefore(const T &newElem);

    /// delete my element: iter moves to element behind if existent
    virtual void remove()
    {
        if (d_actElem)
        {
            coDLListElem<T> *newNext = d_actElem->next;
            d_myList->remove(d_actElem);
            d_actElem = newNext;
            nextValid();
        }
    }

    /// set to specific element
    void operator=(const T &elem)
    {
        if (d_myList)
        {
            d_actElem = d_myList->head;
            nextValid();
            while ((d_actElem) && !(d_actElem->item == elem))
                advance();
        }
    }

    /// set to element # if existent
    void operator=(int num)
    {
        if (d_myList)
        {
            d_actElem = d_myList->head;
            nextValid();
            while ((d_actElem) && (num))
            {
                advance();
                num--;
            }
        }
    }
};

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++++
// ++++++    class coDLListSafeIter<T>
// ++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template <class T>
class coDLListSafeIter : public coDLListIter<T>
{
    friend class coDLList<T>;

public:
    /// create empty iterator
    coDLListSafeIter<T>()
        : coDLListIter<T>()
    {
    }

    /// create from a given List
    coDLListSafeIter<T>(coDLList<T> &list)
        : coDLListIter<T>(list)
    {
        this->d_myList->addIter(this);
    }

    /// standard assignment operator
    coDLListSafeIter<T> &operator=(const coDLListSafeIter<T> &old)
    {
        this->d_myList = old.d_myList;
        this->d_actElem = old.d_actElem;
        this->nextValid();
        this->d_myList->addIter(this);
        return *this;
    }

    /// standard copy-Constructor
    coDLListSafeIter<T>(const coDLListSafeIter<T> &old)
        : coDLListIter<T>(old)
    {
        //this->d_myList = old.d_myList;
        //this->d_actElem = old.d_actElem;
        //this->nextValid();
        this->d_myList->addIter(this);
    }

    /// create from a given List + Pointer
    coDLListSafeIter<T>(coDLList<T> &list, coDLListElem<T> *actElem)
    {
        this->d_myList = &list;
        this->d_actElem = actElem;
        this->nextValid();
        this->d_myList->addIter(this);
    }

    /// Destructor
    virtual ~coDLListSafeIter()
    {
        this->d_myList->removeIter(this);
    }

    /// delete my element: iter is moved to next element if it exists by coDLList::remove())
    virtual void remove()
    {
        if (this->d_actElem)
        {
            this->d_myList->remove(this->d_actElem);
            this->nextValid();
        }
    }
};

// -------------- End coDLListSafeIter ------------------------------

//==========================================================================
// INLINE implementation of coDLList member functions
//==========================================================================

template <class T>
INLINE coDLListIter<T> coDLList<T>::first()
{
    return coDLListIter<T>(*this, this->head);
}

template <class T>
INLINE coDLListIter<T> coDLList<T>::last()
{
    return coDLListIter<T>(*this, this->tail);
}

template <class T>
INLINE coDLList<T>::coDLList(void)
{
    this->head = this->tail = NULL;
    this->listItems = 0;
}

template <class T>
INLINE coDLListIter<T> coDLList<T>::findElem(const T &searchItem)
{
    coDLListIter<T> iter(*this, this->head);
    while ((iter) && !((*iter) == searchItem))
    {
        iter++;
    }
    return iter;
}

template <class T>
INLINE coDLListIter<T> coDLList<T>::findElem(const T &searchItem,
                                             const coDLListCompare<T> &comp)
{
    coDLListIter<T> iter(*this, this->head);
    while ((iter) && !comp.equal((*iter), searchItem))
    {
        iter++;
    }
    return iter;
}

template <class T>
INLINE coDLListIter<T> coDLList<T>::findElem(int num)
{
    coDLListIter<T> iter(*this, this->head);
    while ((iter) && (num))
    {
        iter++;
        num--;
    }
    return iter;
}

//==========================================================================
// Destructor: cleanup everything, but NO things pointered on
//==========================================================================
template <class T>
INLINE coDLList<T>::~coDLList(void)
{
    clean();
}

//==========================================================================
// Cleanup: cleanup everything, but NO things pointered on
//==========================================================================
template <class T>
INLINE void coDLList<T>::clean(void)
{
    coDLListElem<T> *curr = this->head;
    while (curr)
    {
        this->head = curr;
        curr = curr->next;
        delete this->head;
    }
    this->head = this->tail = curr = NULL;
    this->listItems = 0;
    invalidateIterators();
}

//==========================================================================
// Cleanup: cleanup everything, dependend on d_doDelete delete things pointered on
//==========================================================================
template <class T>
INLINE void coDLPtrList<T>::clean(void)
{
    coDLListElem<T> *curr = this->head;
    while (curr)
    {
        this->head = curr;
        curr = curr->next;
        if (d_doDelete)
            delete this->head->item;
        delete this->head;
    }
    this->head = this->tail = curr = NULL;
    this->listItems = 0;
    this->invalidateIterators();
}

//==========================================================================
// add new item to end of list
//==========================================================================
template <class T>
INLINE coDLList<T> &coDLList<T>::append(const T &a)
{
    coDLListElem<T> *newlink = new coDLListElem<T>(a);
    newlink->next = NULL;
    newlink->prev = this->tail;
    this->tail = newlink;
    if (newlink->prev)
        (newlink->prev)->next = newlink;
    else
        this->head = newlink;
    this->listItems++;
    return *this;
}

//==========================================================================
// add new item to end of list
//==========================================================================
template <class T>
INLINE coDLPtrList<T> &coDLPtrList<T>::append(const T &a)
{
    coDLListElem<T> *newlink = new coDLListElem<T>(a);
    newlink->next = NULL;
    newlink->prev = this->tail;
    this->tail = newlink;
    if (newlink->prev)
        (newlink->prev)->next = newlink;
    else
        this->head = curr = newlink;
    this->listItems++;
    return *this;
}

//==========================================================================
//  remove specific item from list
//==========================================================================
template <class T>
INLINE void coDLList<T>::remove(coDLListElem<T> *removeElem)
{
    if (removeElem)
    {
        correctIteratorsFor(removeElem);
        if (removeElem->prev)
            (removeElem->prev)->next = removeElem->next;
        if (removeElem->next)
            (removeElem->next)->prev = removeElem->prev;
        if (this->head == removeElem)
            this->head = removeElem->next;
        if (this->tail == removeElem)
            this->tail = removeElem->prev;
        delete removeElem;
        this->listItems--;
    }
}

//==========================================================================
//  remove specific item from list : PTR version
//==========================================================================
template <class T>
INLINE void coDLPtrList<T>::remove(coDLListElem<T> *removeElem)
{
    if (removeElem)
    {
        correctIteratorsFor(removeElem);
        if (d_doDelete)
            delete removeElem->item;
        if (removeElem->prev)
            (removeElem->prev)->next = removeElem->next;
        if (removeElem->next)
            (removeElem->next)->prev = removeElem->prev;
        if (this->head == removeElem)
            this->head = removeElem->next;
        if (this->tail == removeElem)
            this->tail = removeElem->prev;
        delete removeElem;
        this->listItems--;
    }
}

//==========================================================================
// return the Nth item; do not change current item position : CONST
//==========================================================================
template <class T>
INLINE const T &coDLList<T>::operator[](int n) const
{
    coDLListElem<T> *link = this->head;
    while (link && n > 0)
    {
        link = link->next;
        n--;
    }
    if (link && n == 0)
        return link->item;
    else
        return this->head->item; // problem if NO items in list
}

//==========================================================================
// return the Nth item; do not change current item position
//==========================================================================
template <class T>
INLINE T &coDLList<T>::operator[](int n)
{
    coDLListElem<T> *link = this->head;
    while (link && n > 0)
    {
        link = link->next;
        n--;
    }
    if (link && n == 0)
        return link->item;
    else
        return this->head->item; // problem if NO items in list
}

//==========================================================================
// return the Nth item; do not change current item position
//==========================================================================
template <class T>
INLINE T coDLList<T>::item(int n)
{
    coDLListElem<T> *link = this->head;
    while (link && n > 0)
    {
        link = link->next;
        n--;
    }
    if (link && n == 0)
        return link->item;
    else
        return this->head->item; // problem if NO items in list
}

//==========================================================================
// just return the current item, do not change which item is current
//==========================================================================
template <class T>
INLINE T coDLPtrList<T>::current(void)
{
    coDLListElem<T> *link = curr;
    if (!link)
    {
        reset();
        return d_nullElem;
    }
    return link->item;
}

//==========================================================================
// just return the current item, do not change which item is current
//==========================================================================
//template<class T>
//INLINE T coDLList<T>::getLast(void)
//{
//  coDLListElem<T> *link = this->tail;
//  if(!link) {
//    reset();
//    return(NULL);
//  }
//  return link->item;
//}

//==========================================================================
// move the current item pointer to the Nth item
//==========================================================================
template <class T>
INLINE coDLPtrList<T> &coDLPtrList<T>::set(int N)
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
INLINE int coDLPtrList<T>::find(const T &a)
{
    reset();
    while (curr && !(curr->item == a))
        next();
    return (curr != NULL);
}

// +++++++++++++++++ coDLListIter ++++++++++++++++++++++++++++++++++++++++++

//==========================================================================
//  append new item after the current item : Iterator version
//==========================================================================
template <class T>
INLINE coDLListIter<T> &coDLListIter<T>::insertAfter(const T &a)
{
    if (d_actElem)
    { // yes, there is a current item
        coDLListElem<T> *newlink = new coDLListElem<T>(a);
        newlink->next = d_actElem->next;
        newlink->prev = d_actElem;
        d_actElem->next = newlink;
        if (newlink->next == NULL)
            d_myList->tail = newlink;
        else
            newlink->next->prev = newlink;
        d_myList->listItems++;
    }
    else // no current item; just append at end
        d_myList->append(a);

    return *this;
}

//==========================================================================
//  append new item before the current item : Iterator version
//==========================================================================
template <class T>
INLINE coDLListIter<T> &coDLListIter<T>::insertBefore(const T &a)
{
    if (d_actElem)
    { // yes, there is a current item
        coDLListElem<T> *newlink = new coDLListElem<T>(a);
        newlink->next = d_actElem;
        newlink->prev = d_actElem->prev;
        d_actElem->prev = newlink;
        if (newlink->prev == NULL)
            d_myList->head = newlink;
        else
            newlink->prev->next = newlink;
        d_myList->listItems++;
    }
    else // no current item; just append at end
        d_myList->append(a);

    return *this;
}

#endif //YAC_LINK_LIST_H
