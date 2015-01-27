/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGENTRYPTRLIST_H
#define COCONFIGENTRYPTRLIST_H

#include <QLinkedList>
#include <QString>
#include <QStringList>

#include "coConfigConstants.h"
#include <util/coTypes.h>

namespace covise
{

class coConfigEntry;

class CONFIGEXPORT coConfigEntryPtrList : public QLinkedList<coConfigEntry *>
{

public:
    coConfigEntryPtrList();
    ~coConfigEntryPtrList();

    //    int find(const QString & name);
    //    int findNext(const QString & name);
    //
    //    int find(const char * name);
    //    int findNext(const char * name);

protected:
    // Who needs this?
    //   virtual int compareItems(Q3PtrCollection::Item item1,
    //                            Q3PtrCollection::Item item2);
};
}
#include "coConfigEntry.h"
#endif
