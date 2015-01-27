/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: InvObjectList.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  a list of render objects
//
//
// * Class(es)      : InvObjectList, InvObject
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************
//

//
// C stuff
//
#include <covise/covise.h>

#include "InvObjectList.h"

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
const char *InvObject::getName()
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
    if (obj->name == NULL)
    {
        return NULL;
    }
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
InvObject *
InvObjectList::search(const char *objName)
{
    InvObject *pos;
    if (NULL == objName)
    {
        return NULL;
    }
    strcpy(tail->name, objName);
    pos = head;

    do
    {
        pos = pos->next;
    } while (strcmp(pos->name, objName) != 0);

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
InvObject *
InvObjectList::searchX(const char *objName)
{
    if (NULL == objName)
    {
        return NULL;
    }
    InvObject *pos;
    InvObject *posFound = NULL;

    //      strcpy(tail->name,objName);
    pos = head;
    char redName[255];

    do
    {
        strcpy(redName, pos->name);

        // we reduce the objectname here (XXX_Y_OUT_nnn -> XXX_Y_OUT)
        char chch[255];
        char lastCh[255][20];

        strcpy(chch, redName);
        char del[3];
        strcpy(del, "_");
        char *tok;
        tok = strtok(chch, del);
        int cnt = 0;
        while (tok)
        {
            strcpy(lastCh[cnt], tok);
            tok = strtok(NULL, del);
            cnt++;
        }
        int ii;
        strcpy(redName, lastCh[0]);
        strcat(redName, del);
        for (ii = 1; ii < cnt - 1; ++ii)
        {
            strcat(redName, lastCh[ii]);
            if (ii != cnt - 2)
                strcat(redName, del);
        }

        // set posFound if the objName fits either the name of the current obj or
        // its reduced name
        if (!(strcmp(redName, objName) && strcmp(pos->name, objName)))
        {
            posFound = pos;
            break;
        }
        pos = pos->next;

    } while (pos->index > 0);

    return posFound;
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
