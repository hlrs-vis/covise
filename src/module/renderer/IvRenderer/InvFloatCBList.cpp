/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvFloatCBList.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include "InvFloatCallbackList.h"

typedef struct SoCallbackStruct
{
    MyFloatCallbackListCB *func;
    void *userData;
} SoCallbackStruct_;

//////////////////////////////////////////////////////////////////////////////
//
//  Constructor
//
MyFloatCallbackList::MyFloatCallbackList()
//
//////////////////////////////////////////////////////////////////////////////
{
}

//////////////////////////////////////////////////////////////////////////////
//
//  Destructor
//
MyFloatCallbackList::~MyFloatCallbackList()
//
//////////////////////////////////////////////////////////////////////////////
{
    int len = list.getLength();

    for (int i = 0; i < len; i++)
    {
        delete ((SoCallbackStruct *)list[i]);
    }
}

//////////////////////////////////////////////////////////////////////////////
//
//  addCallback - adds the function f to the callback list, along with
//  userData. At invocation, f will be passed userData, along with callback
//  specific data.
//
void
MyFloatCallbackList::addCallback(MyFloatCallbackListCB *f, void *userData)
//
//////////////////////////////////////////////////////////////////////////////
{
    if (f == NULL)
        return;

    SoCallbackStruct *cb = new SoCallbackStruct;
    cb->func = f;
    cb->userData = userData;

    list.append(cb);
}

//////////////////////////////////////////////////////////////////////////////
//
//  removeCallback - removes the function f associated with userData from the.
//  callback list.
//
void
MyFloatCallbackList::removeCallback(MyFloatCallbackListCB *f, void *userData)
//
//////////////////////////////////////////////////////////////////////////////
{
    int len = list.getLength();
    SoCallbackStruct *cb;
    int found = 0;

    for (int i = 0; (i < len) && (!found); i++)
    {
        cb = (SoCallbackStruct *)list[i];
        if ((cb->func == f) && (cb->userData == userData))
        {
            list.remove(i);
            delete cb;
            found = 1;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
//
//  invokeCallbacks - invokes each callback func in the list, passing.
//  userData, and callbackData as the parameters.
//
void
MyFloatCallbackList::invokeCallbacks(float callbackData)
//
//////////////////////////////////////////////////////////////////////////////
{
    int len = list.getLength();
    SoCallbackStruct *cb;

    for (int i = 0; i < len; i++)
    {
        cb = (SoCallbackStruct *)list[i];
        (*cb->func)(cb->userData, callbackData);
    }
}
