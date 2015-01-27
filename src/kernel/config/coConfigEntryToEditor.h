/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIG_ENTRYTOEDITOR_H
#define COCONFIG_ENTRYTOEDITOR_H
#include <config/coConfigEntry.h>
namespace covise
{

class coConfigEntryToEditor
{
public:
    static QList<coConfigEntry *> getSubEntries(coConfigEntry *entry);

protected:
    coConfigEntryToEditor()
    {
    }

private:
    static QStringList groups;
};
}
#endif
