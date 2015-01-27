/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <Inventor/SbPList.h>
#include <Inventor/SoPath.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/manips/SoTransformManip.h>

#include "InvManipList.h"

typedef struct InvPathManipStuff
{
    SoPath *selectionPath;
    SoTransformManip *manip;
    SoPath *xfPath;
} InvPathManipStuff;

////////////////////////////////////////////////////////////////////////
//
// Constructor
//
// Use: public
InvManipList::InvManipList()
//
////////////////////////////////////////////////////////////////////////
{
    list = new SbPList;
}

////////////////////////////////////////////////////////////////////////
//
// Destructor
//
// Use: public
InvManipList::~InvManipList()
//
////////////////////////////////////////////////////////////////////////
{
    delete list;
}

////////////////////////////////////////////////////////////////////////
//
// Use: public
int
InvManipList::getLength() const
//
////////////////////////////////////////////////////////////////////////
{
    return list->getLength();
}

////////////////////////////////////////////////////////////////////////
//
// Append adds the selectionPath/manip/xfPath stuff to the list.
// This ref()'s both the paths and the manip.
//
void InvManipList::append(SoPath *selectionP, SoTransformManip *m,
                          SoPath *xfP)
//
////////////////////////////////////////////////////////////////////////
{
    InvPathManipStuff *stuff = new InvPathManipStuff;

    stuff->selectionPath = selectionP;
    stuff->manip = m;
    stuff->xfPath = xfP;
    selectionP->ref();
    m->ref();
    xfP->ref();

    list->append(stuff);
}

////////////////////////////////////////////////////////////////////////
//
// Find locates the first selectionPath/manip/xfPath stuff whose
// selectionPath is p, and returns the index in the list of that stuff.
//
// Use: public
int InvManipList::find(const SoPath *p) const
//
////////////////////////////////////////////////////////////////////////
{
    int which = -1;

    for (int i = 0; (i < list->getLength()) && (which == -1); i++)
    {
        InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[i];
        if (*stuff->selectionPath == *p)
            which = i;
    }

    return which;
}

////////////////////////////////////////////////////////////////////////
//
// Find locates the first selectionPath/manip/xfPath stuff whose manip is m,
// and returns the index in the list of that stuff.
//
// Use: public
int InvManipList::find(const SoTransformManip *m) const
//
////////////////////////////////////////////////////////////////////////
{
    int which = -1;

    for (int i = 0; (i < list->getLength()) && (which == -1); i++)
    {
        InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[i];
        if (stuff->manip == m)
            which = i;
    }

    return which;
}

////////////////////////////////////////////////////////////////////////
//
// findByXfPath locates the first selectionPath/manip/xfPath stuff whose
// xfPath is p, and returns the index in the list of that stuff.
//
// Use: public
int InvManipList::findByXfPath(const SoPath *p) const
//
////////////////////////////////////////////////////////////////////////
{
    int which = -1;

    for (int i = 0; (i < list->getLength()) && (which == -1); i++)
    {
        InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[i];
        if (*stuff->xfPath == *p)
            which = i;
    }

    return which;
}

////////////////////////////////////////////////////////////////////////
//
// Remove removes the selectionPath/manip/xfPath stuff specified by
// which index from the list. This unref()'s both paths and the manip.
//
// Use: public
void InvManipList::remove(int which)
//
////////////////////////////////////////////////////////////////////////
{
    InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[which];

    stuff->selectionPath->unref();
    stuff->manip->unref();
    stuff->xfPath->unref();

    list->remove(which);
}

////////////////////////////////////////////////////////////////////////
//
// This returns the selectionPath in the selectionPath/manip/xfPath stuff
// specified by which index.
//
// Use: public
SoPath *InvManipList::getSelectionPath(int which) const
//
////////////////////////////////////////////////////////////////////////
{
    InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[which];
    return (stuff->selectionPath);
}

////////////////////////////////////////////////////////////////////////
//
// This returns the manip in the selectionPath/manip/xfPath stuff
// specified by which index.
//
// Use: public
SoTransformManip *InvManipList::getManip(int which) const
//
////////////////////////////////////////////////////////////////////////
{
    InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[which];
    return (stuff->manip);
}

////////////////////////////////////////////////////////////////////////
//
// This returns the xfPath of the manip in the
// selectionPath/manip/xfPath stuff specified by which index.
//
// Use: public
SoPath *InvManipList::getXfPath(int which) const
//
////////////////////////////////////////////////////////////////////////
{
    InvPathManipStuff *stuff = (InvPathManipStuff *)(*list)[which];
    return (stuff->xfPath);
}
