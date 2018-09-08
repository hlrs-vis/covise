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

#ifndef VV_SLLIST_H
#define VV_SLLIST_H

#include <iostream>
#include <assert.h>

#include "vvexport.h"

//============================================================================
// Declaration
//============================================================================

//----------------------------------------------------------------------------
/** Node in a singly linked list of pointers.
  @author Juergen P. Schulze
  @see vvSLList
*/
template<class T> class vvSLNode
{
  public:
    enum DeleteType                               /// types for data deletion
    {
      NO_DELETE,                                  ///< don't delete data because it will be deleted by the caller
      NORMAL_DELETE,                              ///< delete data when not used anymore, use normal delete (delete)
      ARRAY_DELETE,                               ///< delete data when not used anymore, use array delete (delete[])
      UNDEFINED                                   ///< used when node doesn't exist etc.
    };
    T         data;                               ///< pointer to this node's list element
    vvSLNode* next;                               ///< pointer to next node
    DeleteType deleteData;                        ///< determines with which delete command the data is deleted in destructor

    /** Constructor
      @param dt initialization of deleteData attribute
    */
    vvSLNode(DeleteType dt)
    {
      data = NULL;
      next = NULL;
      deleteData = dt;
    }
    /// Copy constructor
    vvSLNode(vvSLNode* node)
    {
      data = node->data;
      next = node->next;
      deleteData = node->deleteData;
    }
    /// Destructor
    ~vvSLNode()
    {
      switch(deleteData)
      {
        case NO_DELETE: break;
        case NORMAL_DELETE: delete data; break;
        case ARRAY_DELETE: delete[] data; break;
        default: assert(0); break;
      }
    }
};

//----------------------------------------------------------------------------
/** Singly linked list of pointers.
  A current element is being maintained within the class. Most operations
  refer to the current element in their functionality. For instance, insertAfter()
  inserts a new element after the current element.
  @author Juergen P. Schulze
  @see vvSLNode
*/
template<class T> class vvSLList
{
  private:
    T nullValue;                                  ///< NULL value to return on error

  protected:
    vvSLNode<T>* head;                            ///< head of list. If list is empty, this is NULL
    vvSLNode<T>* cur;                             ///< current element or NULL if list is empty

  public:
    vvSLList();
    vvSLList(vvSLList<T>*);
    ~vvSLList();
    void removeAll();
    int  getIndex();
    T&   getData();
    typename vvSLNode<T>::DeleteType getDeleteType();
    void setData(const T&);
    void setDeleteData(typename vvSLNode<T>::DeleteType);
    void setDeleteDataAll(typename vvSLNode<T>::DeleteType);
    bool next();
    bool previous();
    bool first();
    bool last();
    bool isLast() const;
    bool isEmpty();
    bool remove();
    bool makeCurrent(int);
    bool find(const T&);
    size_t  count() const;
    void append(const T&, typename vvSLNode<T>::DeleteType);
    void insertAfter(const T&, typename vvSLNode<T>::DeleteType);
    void insertBefore(const T&, typename vvSLNode<T>::DeleteType);
    void insertSorted(const T&, typename vvSLNode<T>::DeleteType);
    void merge(vvSLList<T>*);
    void print();
};

//============================================================================
// Implementation
//============================================================================

//----------------------------------------------------------------------------
/// Constructor
template<class T>  vvSLList<T>::vvSLList()
{
  head = cur = NULL;
  nullValue = (T)0;
}

//----------------------------------------------------------------------------
/// Destructor. Removes all list elements.
template<class T>  vvSLList<T>::~vvSLList()
{
  removeAll();
}

//----------------------------------------------------------------------------
/** Copy constructor.
  Creates a copy of a list with new nodes, but the pointers to the
  data remain the same.
  The current element will be the same in the new list as in the copied list.
*/
template<class T> vvSLList<T>::vvSLList(vvSLList<T>* otherList)
{
  vvSLNode<T>* toCopy;                            // list element which is to be copied
  vvSLNode<T>* newNode;                           // new node in copied list
  vvSLNode<T>* prevNode;                          // previous node in copied list

  nullValue = (T)0;
  if (otherList->head==NULL)                      // copy empty list?
  {
    head = NULL;
    cur  = NULL;
  }
  else
  {
    toCopy = otherList->head;
    prevNode = NULL;
    while (toCopy)
    {
      newNode = new vvSLNode<T>(toCopy);
      newNode->next = NULL;
      if (prevNode==NULL)
      {
        head = newNode;
        prevNode = head;
      }
      else
      {
        prevNode->next = newNode;
        prevNode = newNode;
      }
      if (toCopy==otherList->cur) cur = newNode;
      toCopy = toCopy->next;
    }
  }
}

//----------------------------------------------------------------------------
/** Remove all list elements.
 */
template<class T> void vvSLList<T>::removeAll()
{
  vvSLNode<T>* nextElem;
  vvSLNode<T>* thisElem;

  thisElem = head;
  while (thisElem)
  {
    nextElem = thisElem->next;
    delete thisElem;
    thisElem = nextElem;
  }
  head = cur = NULL;
}

//----------------------------------------------------------------------------
/** Return the index of the current element.
  If -1 is returned, the list is empty and there is no current element.
*/
template<class T> int vvSLList<T>::getIndex()
{
  vvSLNode<T>* tmp = head;
  int i = 0;

  if (head==NULL) return -1;                      // empty list

  while (tmp != cur)
  {
    tmp = tmp->next;
    ++i;
  }
  return i;
}

//----------------------------------------------------------------------------
/** Return the pointer to the current element's data.
  If the list is empty, nullValue is returned.
*/
template<class T> T& vvSLList<T>::getData()
{
  if (head) return cur->data;
  else return nullValue;
}

//----------------------------------------------------------------------------
/** Return the pointer to the current element's data.
  If the list is empty, nullValue is returned.
*/
template<class T> typename vvSLNode<T>::DeleteType vvSLList<T>::getDeleteType()
{
  if (head) return cur->deleteData;
  else return vvSLNode<T>::UNDEFINED;
}

//----------------------------------------------------------------------------
/** Set the current element's data value (pointer).
  If the list is empty, nothing happens.
*/
template<class T> void vvSLList<T>::setData(const T& x)
{
  if (head) cur->data = x;
}

//----------------------------------------------------------------------------
/** Set the deleteData attribute of the current list node to the passed value.
  The position of the current list element does not change.
  @param newState new value of deleteData for current node
*/
template<class T> void vvSLList<T>::setDeleteData(typename vvSLNode<T>::DeleteType newState)
{
  if (head) cur->deleteData = newState;
}

//----------------------------------------------------------------------------
/** Set the deleteData attribute of all list nodes to the passed value.
  The position of the current list element does not change.
  @param newState new value of deleteData in all nodes
*/
template<class T> void vvSLList<T>::setDeleteDataAll(typename vvSLNode<T>::DeleteType newState)
{
  vvSLNode<T>* tmp = head;

  while (tmp != NULL)
  {
    tmp->deleteData = newState;
    tmp = tmp->next;
  }
}

//----------------------------------------------------------------------------
/** Move the current element to the next element in the list.
  If the current element is the last element of the list,
  it does not change.
  @return true if a next element was found, false if the current element
          did not change
*/
template<class T> bool vvSLList<T>::next()
{
  if (head==NULL) return false;
  if (cur->next)
  {
    cur = cur->next;
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------
/** Move the current back by one list node.
  If the current element is the first element of the list,
  it is not changed.
  @return true if a previous element was found, false if the current element
          did not change
*/
template<class T> bool vvSLList<T>::previous()
{
  vvSLNode<T>* tmp = head;

  if (head==NULL) return false;                   // empty list
  if (head==cur) return false;                    // current element is first list element
  while (tmp->next != cur)
    tmp = tmp->next;
  cur = tmp;
  return true;
}

//----------------------------------------------------------------------------
/** Set the current element to be the first element in the list.
  @return true if successful, false if list is empty
*/
template<class T> bool vvSLList<T>::first()
{
  if (head==NULL) return false;
  cur = head;
  return true;
}

//----------------------------------------------------------------------------
/** Set the current element to be the last element in the list.
  @return true if successful, false if list is empty
*/
template<class T> bool vvSLList<T>::last()
{
  if (head==NULL) return false;
  cur = head;
  while (cur->next != NULL)
    cur = cur->next;
  return true;
}

//----------------------------------------------------------------------------
/** @return true if current element is the last in the list or the list is empty.
 */
template<class T> bool vvSLList<T>::isLast() const
{
  return ((head == NULL) || (cur->next == NULL));
}

//----------------------------------------------------------------------------
/// Return true if the list is empty, false if it contains at least one element.
template<class T> bool vvSLList<T>::isEmpty()
{
  return (head==NULL);
}

//----------------------------------------------------------------------------
/** Remove current element from list.
  The new current element is the one before the one removed.
  If the head of the list is removed, the previously second element is the
  new current element.
  @return true if successful, false if list is empty
*/
template<class T> bool vvSLList<T>::remove()
{
  vvSLNode<T>* tmp;

  if (head==NULL) return false;                   // list is empty
  if (cur==head)                                  // remove first element
  {
    head = head->next;
    delete cur;
    cur = head;
  }
  else
  {
    tmp = cur;
    previous();
    cur->next = tmp->next;
    delete tmp;
  }
  return true;
}

//----------------------------------------------------------------------------
/** Set the current element to be the one at a specific index. If the
  desired index does not exist, the current element is not changed and
  false is returned.
  @param index index to set the current element to (0=first element)
*/
template<class T> bool vvSLList<T>::makeCurrent(int index)
{
  vvSLNode<T>* tmp = head;
  int i = 0;

  if (head==NULL) return false;
  tmp = head;
  while (tmp && i!=index)
  {
    tmp = tmp->next;
    ++i;
  }
  if (i==index && tmp)
  {
    cur = tmp;
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------
/** Find the first occurrence of an element in the list and make it
  the current element.
  If the element was not found, the current element does not change.
  @return true if element was found, false if it was not found
*/
template<class T> bool vvSLList<T>::find(const T& x)
{
  vvSLNode<T>* tmp = head;

  if (head==NULL) return false;                   // empty list

  while (tmp->data!=x)
  {
    tmp = tmp->next;
    if (tmp==NULL) return false;
  }
  cur = tmp;
  return true;
}

//----------------------------------------------------------------------------
/// Return the number of elements in the list.
template<class T> size_t vvSLList<T>::count() const
{
  size_t numElements = 0;
  vvSLNode<T>* tmp = head;

  while (tmp != NULL)
  {
    numElements++;
    tmp = tmp->next;
  }

  return numElements;
}

//----------------------------------------------------------------------------
/** Append a new element to the end of the list.
  The appended element becomes the current element.
  @param x  element to add
  @param dd true = delete data when list element is removed (defaults to true)
*/
template<class T> void vvSLList<T>::append(const T& x, typename vvSLNode<T>::DeleteType dd)
{
  vvSLNode<T>* newNode = new vvSLNode<T>(dd);
  vvSLNode<T>* tmp;

  newNode->data = x;
  newNode->next = NULL;

  if (head==NULL)
  {
    head = newNode;
  }
  else
  {
    tmp = head;
    while (tmp->next != NULL)
      tmp = tmp->next;
    tmp->next = newNode;
  }
  cur = newNode;
}

//----------------------------------------------------------------------------
/** Insert the new element after the current element.
  The inserted element becomes the current element.
  @param x  element to add
  @param dd true = delete data when list element is removed (defaults to true)
*/
template<class T> void vvSLList<T>::insertAfter(const T& x, typename vvSLNode<T>::DeleteType dt)
{
  vvSLNode<T>* newNode = new vvSLNode<T>(dt);

  newNode->data = x;

  if (head==NULL)
  {
    head = newNode;
    newNode->next = NULL;
  }
  else
  {
    newNode->next = cur->next;
    cur->next = newNode;
  }
  cur = newNode;
}

//----------------------------------------------------------------------------
/** Insert the new element before the current element.
  The inserted element becomes the current element.
  @param x  element to add
  @param dd true = delete data when list element is removed (defaults to true)
*/
template<class T> void vvSLList<T>::insertBefore(const T& x, typename vvSLNode<T>::DeleteType dt)
{
  vvSLNode<T>* newNode = new vvSLNode<T>(dt);
  vvSLNode<T>* tmp;

  newNode->data = x;

  if (head==NULL)
  {
    head = newNode;
    newNode->next = NULL;
  }
  else
  {
    if (cur==head)                                // current element is first list element
    {
      head = newNode;
      head->next = cur;
    }
    else
    {
      tmp = head;
      while (tmp->next != cur)
        tmp = tmp->next;
      tmp->next = newNode;
      newNode->next = cur;
    }
  }
  cur = newNode;
}

//----------------------------------------------------------------------------
/** Insert the new element sorted. Sorting is performed using '<' relation.
  Note that the < operator has to be overloaded for the type to insert.
  The used algorithm is 'insertion sort'.
  @param x element to add
  @param dd true = delete data when list element is removed (defaults to true)
*/
template<class T> void vvSLList<T>::insertSorted(const T& x, typename vvSLNode<T>::DeleteType dt)
{
  T tmp;

  if (count() != 0)
  {
    first();
    while ((tmp = getData()) != 0)
    {
      if (*tmp < *x)
      {
        insertBefore(x, dt);
        break;
      }
      if (!next())
      {
        append(x, dt);
        break;
      }
    }
  }
  else
  {
    append(x, dt);
  }
}

//----------------------------------------------------------------------------
/** Merge two lists. The new list is appended at the end of the current list,
  its nodes are removed. The current element does not change.
  @param newList new list which is appended to this list
*/
template<class T> void vvSLList<T>::merge(vvSLList<T>* newList)
{
  vvSLNode<T>* tmp;

  if (head==NULL)
  {
    head = newList->head;
    cur = head;
  }
  else
  {
    tmp = head;
    while (tmp->next != NULL)
    {
      tmp = tmp->next;
    }
    tmp->next = newList->head;
  }

  // Remove nodes from merged list:
  newList->head = NULL;
  newList->cur = NULL;
}

//----------------------------------------------------------------------------
/** Print the values of all elements. The current element does not change.
 */
template<class T> void vvSLList<T>::print()
{
  int i = 0;
  vvSLNode<T>* tmp;

  tmp = head;
  while (tmp)
  {
    std::cerr << "Node " << i << ": element=" << intptr_t(tmp->data) << ", value=" << int(*(tmp->data)) << std::endl;
    tmp = tmp->next;
    ++i;
  }
}
#endif

/////////////////
// End of File
/////////////////
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
