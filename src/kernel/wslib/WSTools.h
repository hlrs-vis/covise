/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSTOOLS_H
#define WSTOOLS_H

#include <QString>

#include "WSExport.h"

namespace covise
{

class WSParameter;

class WSLIBEXPORT WSTools
{
public:
    static QString fromCovise(const QString &from);
    static QString toCovise(const QString &from);

    static bool setParameterFromString(WSParameter *parameter, const QString &value);

private:
    WSTools()
    {
    }
    ~WSTools()
    {
    }
};
}
#endif
