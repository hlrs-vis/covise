/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <iostream>
using namespace std;

#include <stdio.h>
#include <string.h>

#include "InvObjectList.h"

#ifndef YAC
#include <covise/covise.h>
#include <covise/covise_process.h>
#endif

//
//##########################################################################
// InvObject
//##########################################################################
//

//==========================================================================
//  constructor : use default arguments
//==========================================================================
InvObject::InvObject()
{
    strcpy(name, "UNKNOWN");
    next = NULL;
    objPtr = NULL;
    index = 1;
}

//=========================================================================
// constructor : setup name , type and data pointer
//=========================================================================
InvObject::InvObject(const char *n, SoGroup *ptr)
{
    strcpy(name, n);
    next = NULL;
    index = 1;
    objPtr = ptr;
}

//=========================================================================
// set object name
//=========================================================================
void InvObject::setName(const char *n)
{
    strcpy(name, n);
}

//=========================================================================
// set object pointer
//=========================================================================
void InvObject::setObject(SoGroup *ptr)
{
    objPtr = ptr;
}

//=========================================================================
// get the name of the object
//=========================================================================
char *InvObject::getName()
{
    return name;
}

//=========================================================================
// get the data of the object
//=========================================================================
SoGroup *InvObject::getObject()
{
    return objPtr;
}

//
//#########################################################################
// InvObjectList
//#########################################################################
//

//=========================================================================
// constructor : setup head and tail of list
//=========================================================================
InvObjectList::InvObjectList()
{
    head = new InvObject("HEAD", NULL);
    tail = new InvObject("TAIL", NULL);
    cur = head;
    head->next = tail;
    tail->next = head;
    tail->index = -1;
    head->index = 0;
    length = 0;
}

//=========================================================================
// add object to the list
//=========================================================================
int InvObjectList::add(InvObject *obj)
{
    if (obj != NULL)
    {
        obj->next = head->next;
        head->next = obj;
        length++;
        return 0;
    }
    else
        return -1;
}

//=========================================================================
// remove object from list
//=========================================================================
int InvObjectList::remove(InvObject *obj)
{
    if (obj != NULL)
    {
        InvObject *pos;
        pos = head;
        strcpy(tail->name, obj->name);

        while (strcmp(pos->next->name, obj->name) != 0)
            pos = pos->next;

        if (pos->next != tail)
            pos->next = pos->next->next;
        else
        {
            strcpy(tail->name, "TAIL");
            return -1;
        }

        if (pos->next == tail)
            // last element was deleted
            tail->next = pos;

        // restore tail
        strcpy(tail->name, "TAIL");
        length--;
        return 0;
    }
    else
        return -1;
}

//=========================================================================
// search for object in list
//=========================================================================
InvObject *InvObjectList::search(InvObject *obj)
{
    InvObject *pos;

    if (obj == NULL)
        return NULL;

    strcpy(tail->name, obj->name);
    pos = head;

    do
    {
        pos = pos->next;
    } while (strcmp(pos->name, obj->name) != 0);

    // restore tail
    strcpy(tail->name, "TAIL");

    // look if we return tail
    if (pos->index == -1)
        return NULL;
    else
        return pos;
}

//=========================================================================
// search for object in list
//=========================================================================
InvObject *InvObjectList::search(const char *obj_name)
{
    InvObject *pos;

    if (obj_name == NULL)
        return NULL;

    strcpy(tail->name, obj_name);
    pos = head;

    do
    {
        pos = pos->next;
    } while (strcmp(pos->name, obj_name) != 0);

    // restore tail
    strcpy(tail->name, "TAIL");

    // look if we return tail
    if (pos->index == -1)
        return NULL;
    else
        return pos;
}

//=========================================================================
// get number of list objects
//=========================================================================
int InvObjectList::getLength()
{
    return length;
}

//=========================================================================
// reset list to first object
//=========================================================================
void InvObjectList::resetToFirst()
{
    cur = head;
}

//=========================================================================
// get the next object from the list
//=========================================================================
InvObject *InvObjectList::getNextObject()
{
    if (cur == tail)
        return NULL;
    else
    {
        cur = cur->next;
        if (cur != tail)
            return cur;
        else
            return NULL;
    }
}

//=========================================================================
// print out current object list
//=========================================================================
void InvObjectList::print()
{
    int i = 0;
    InvObject *obj = head;
    obj = obj->next;

    while (obj->index != -1)
    {
        cerr << "NUM: " << i << " NAME: " << obj->name << endl;
        obj = obj->next;
        i++;
    }
}
