/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_MANIP_LIST_
#define _INV_MANIP_LIST_

#include <util/coTypes.h>

class SbPList;
class SoPath;
class SoTransformManip;

// You can add a selectionPath/manip/xfPath triplet to the list.
// Methods let you find the index of this triplet based on any of the three
// things.  You can then use the index to get the selectionPath, manip, or
// xfPath, or remove the triplet from the list.

class InvManipList
{
public:
    InvManipList();
    ~InvManipList();

    int getLength() const;

    // append will ref() the paths and the manip
    void append(SoPath *selectionP,
                SoTransformManip *m, SoPath *xfP);

    // return the index of the triplet.
    // use this index in calls to:
    // remove(), getSelectionPath(), getManip(), getXfPath()
    int find(const SoPath *p) const;
    int find(const SoTransformManip *m) const;
    int findByXfPath(const SoPath *p) const;

    // remove will unref() the paths and the manip
    void remove(int which);

    // these return the paths or the manip.
    SoPath *getSelectionPath(int which) const;
    SoTransformManip *getManip(int which) const;
    SoPath *getXfPath(int which) const;

private:
    SbPList *list;
};
#endif // _Inv_MANIP_LIST_
