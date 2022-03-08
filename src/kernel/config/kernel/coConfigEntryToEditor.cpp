/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/coConfigEntryToEditor.h>

using namespace covise;

std::vector<coConfigEntry *> coConfigEntryToEditor::getSubEntries(coConfigEntry *entry)
{
    std::vector<coConfigEntry *> subEntries;
    if (entry)
    {
        for (coConfigEntryPtrList::const_iterator item = entry->getChildren().begin();
             item != entry->getChildren().end(); ++item)
        {
            if ((*item) != 0 && (*item)->hasValues())
            {
                subEntries.push_back(item->get());
            }
            if ((*item)->hasChildren())
            {
                auto children = coConfigEntryToEditor::getSubEntries(item->get());
                subEntries.insert(subEntries.end(), children.begin(), children.end());
            }
        }
    }
    return subEntries;
}
