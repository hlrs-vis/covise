/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigEntryPtrList.h>

using namespace covise;

coConfigEntryPtrList::coConfigEntryPtrList()
{
}

coConfigEntryPtrList::~coConfigEntryPtrList()
{
    while (!isEmpty())
    {
        delete takeFirst();
    }
}

// Ist this needed?
// int coConfigEntryPtrList::compareItems(Q3PtrCollection::Item item1,
//                                        Q3PtrCollection::Item item2)
// {
//
//    coConfigEntry *first = static_cast<coConfigEntry*>(item1);
//    coConfigEntry *second = static_cast<coConfigEntry*>(item2);
//
//    QString firstPath = first->getPath().section('.', 1);
//    QString secondPath = second->getPath().section('.', 1);
//
//    if (first == second) return 0;
//    return 1;
//
// }

// int coConfigEntryPtrList::find(const QString & name)
// {
//
//    if (isEmpty()) return -1;
//    if (first()->getName() == name) return at();
//    return findNext(name);
//
// }
//
//
// int coConfigEntryPtrList::findNext(const QString & name)
// {
//
//    for (coConfigEntry* entry = next(); entry; entry = next())
//    {
//       if (entry->getName() == name) return at();
//    }
//
//    return -1;
//
// }
//
//
// int coConfigEntryPtrList::find(const char * name)
// {
//
//    if (isEmpty()) return -1;
//    if (strcmp(first()->getCName(), name) == 0) return at();
//    return findNext(name);
//
// }
//
//
// int coConfigEntryPtrList::findNext(const char * name)
// {
//
//    for (coConfigEntry* entry = next(); entry; entry = next())
//    {
//       if (strcmp(entry->getCName(), name) == 0) return at();
//    }
//
//    return -1;
//
// }
