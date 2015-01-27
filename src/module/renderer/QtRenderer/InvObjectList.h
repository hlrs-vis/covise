/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECT_LIST_H
#define _INV_OBJECT_LIST_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

//
// Inventor Stuff
//
#include <Inventor/nodes/SoGroup.h>

//
// CLASSES
//
class InvObject;
class InvObjectList;

//================================================================
// InvObject
//================================================================

class InvObject
{
    friend class InvObjectList;

private:
    char name[255];
    InvObject *next;
    SoGroup *objPtr;
    int index;

public:
    InvObject(const char *n, SoGroup *data);

    InvObject();

    void setName(const char *n);

    void setObject(SoGroup *obj);

    char *getName();

    SoGroup *getObject();

    ~InvObject(){};
};

//================================================================
// InvObjectList
//================================================================

class InvObjectList
{
    friend class InvObject;

private:
    InvObject *head;
    InvObject *tail;
    InvObject *cur;
    int length;

public:
    InvObjectList();

    InvObject *search(InvObject *obj);

    InvObject *search(const char *obj_name);

    // returns Inv-obj by name ignoring the trailing digits (i.e. obj with name
    // COLLECT_1_OUT_001 is returned if the name COLLECT_1_OUT is given). In case there are
    // multiple choices the obj. which was found first is returned!
    InvObject *searchX(const char *obj_name);

    void resetToFirst();

    InvObject *getNextObject();

    int add(InvObject *obj);

    int remove(InvObject *obj);

    int getLength();

    void print();

    ~InvObjectList()
    {
        delete head;
        delete tail;
    };
};

#endif
