/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NETHELP_H_
#define NETHELP_H_

#include "export.h"
#include <QString>

namespace covise
{
class QTUTIL_EXPORT NetHelp
{
public:
    QString getLocalIP();
    QString GetNamefromAddr(QString *address);
    QString GetIpAddress(const char *hostname);
};

}
#endif
