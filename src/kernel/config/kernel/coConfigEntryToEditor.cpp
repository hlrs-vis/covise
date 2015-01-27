/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigEntryToEditor.h>

using namespace covise;

QList<coConfigEntry *> coConfigEntryToEditor::getSubEntries(coConfigEntry *entry)
{
    QList<coConfigEntry *> subEntries;
    subEntries.clear();
    if (entry)
    {
        for (coConfigEntryPtrList::const_iterator item = entry->children.begin();
             item != entry->children.end(); ++item)
        {
            if ((*item) != 0 && (*item)->hasValues())
            {
                subEntries.append((*item));
            }
            if ((*item)->hasChildren())
            {
                subEntries += coConfigEntryToEditor::getSubEntries((*item));
            }
        }
    }
    return subEntries;
}
