/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGSCHEMAINFOSLIST
#define COCONFIGSCHEMAINFOSLIST

#include <util/coTypes.h>
#include <QList>
#include <QString>
#include <QVector>
#include <QSet>

namespace covise
{

class coConfigSchemaInfos;

class CONFIGEXPORT coConfigSchemaInfosList : public QList<coConfigSchemaInfos *>
{
public:
    coConfigSchemaInfosList(){};
    ~coConfigSchemaInfosList()
    {
        while (!isEmpty())
        {
            delete takeFirst();
        }
    };
};
}
#include <config/coConfigSchemaInfos.h>
#endif
