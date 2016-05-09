/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGTOOLS_H
#define COCONFIGTOOLS_H

#include <QHash>
#include <QString>

class coConfigTools
{
 public:
    static bool matchingAttributes(QHash<QString, QString *> attributes);
    static bool matchingHost(const QString *host);
    static bool matchingMaster(const QString *master);
    static bool matchingArch(const QString *arch);
    static bool matchingRank(const QString *rank);
};
#endif
