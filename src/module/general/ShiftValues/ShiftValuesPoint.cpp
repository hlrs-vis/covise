/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ShiftValuesPoint.h"

float myList::getValue(int lookup)
{
    int i = 0;
    ListElem *p_oneElem = head;
    if (head == 0)
    {
        return FLT_MAX;
    }
    while (i < lookup)
    {
        if (p_oneElem->next == 0)
        {
            Covise::sendError("List search out of bounds, the results may be false");
            return FLT_MAX;
        }
        else
            p_oneElem = p_oneElem->next;

        ++i;
    }

    return p_oneElem->p_Element->Value;
}

void myList::addToList(Elem *p_oneElement)
{
    ListElem *myPtr = head;
    ListElem *myNxt;

    if (head == 0)
    {
        head = new ListElem(p_oneElement);
        return;
    }
    else
    {
        while ((myNxt = myPtr->next))
        {
            myPtr = myNxt;
        }
        myPtr->next = new ListElem(p_oneElement);
    }
}
