/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISO_POINT_H_
#define _ISO_POINT_H_

#include <do/coDistributedObject.h>
#include <appl/ApplInterface.h>
using namespace covise;
#include <float.h>

enum Label
{
    TRIUMPH,
    FAILURE
};

struct myPair
{
    Label Tag;
    float Value;
    myPair()
    {
        Tag = FAILURE;
        Value = FLT_MAX;
    }
};

struct Elem
{
    const coDistributedObject *p_Obj;
    Label Tag;
    float Value;

    Elem()
    {
        p_Obj = 0;
        Tag = FAILURE;
        Value = FLT_MAX;
    }
    Elem(const coDistributedObject *ptr, Label outcome, float value)
    {
        p_Obj = ptr;
        Tag = outcome;
        Value = value;
    }
};

struct ListElem
{
    Elem *p_Element;
    ListElem *next;
    ListElem(Elem *p_oneElem)
    {
        p_Element = p_oneElem;
        next = 0;
    }
    ~ListElem()
    {
        delete next;
        delete p_Element;
    }
};

class myList
{
private:
    ListElem *head;

public:
    myList()
    {
        head = 0;
    }

    float getValue(int lookup);

    void clean()
    {
        delete head;
        head = 0;
    }

    ~myList()
    {
        clean();
    }

    void addToList(Elem *p_oneElement);

    Elem *FindInList(coDistributedObject *pdo)
    {
        ListElem *myPtr = head;
        Elem *e_val, *r_val = 0;

        while (myPtr)
        {
            e_val = myPtr->p_Element;
            if (e_val && e_val->p_Obj == pdo)
            {
                r_val = e_val;
                break;
            }
            myPtr = myPtr->next;
        }
        return r_val;
    }
};
#endif
