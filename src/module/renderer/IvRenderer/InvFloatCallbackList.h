/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_FLOAT_CALLBACK_LIST_
#define _INV_FLOAT_CALLBACK_LIST_

/* $Id: InvFloatCallbackList.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvFloatCallbackList.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Inventor/SbPList.h>

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyFloatCallbackList
//
//////////////////////////////////////////////////////////////////////////////

// Callback functions that are registered with this class should
// be cast to this type.
typedef void MyFloatCallbackListCB(void *userData, float callbackData);

// C-api: prefix=SoFCBList
class MyFloatCallbackList
{

public:
    MyFloatCallbackList();
    ~MyFloatCallbackList();

    //
    // Managing callback functions
    // At callback time, f will be called with userData as the first
    // parameter, and callback specific data as the second parameter.
    // e.g. (*f)(userData, callbackData);

    // C-api: name=addCB
    void addCallback(MyFloatCallbackListCB *f, void *userData = NULL);
    // C-api: name=removeCB
    void removeCallback(MyFloatCallbackListCB *f, void *userData = NULL);

    // C-api: name=clearCB
    void clearCallbacks()
    {
        list.truncate(0);
    }
    // C-api: name=getNumCB
    int getNumCallbacks() const
    {
        return list.getLength();
    }

    // C-api: name=invokeCB
    void invokeCallbacks(float callbackData);

private:
    // callbackList holds a list of functions and user data
    SbPList list;
};
#endif /* _INV_FLOAT_CALLBACK_LIST_ */
