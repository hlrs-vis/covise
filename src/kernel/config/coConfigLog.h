/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGLOG_H
#define COCONFIGLOG_H

#include <util/coTypes.h>
#include <QTextStream>

namespace covise
{

class CONFIGEXPORT coConfigLog
{

public:
    static QTextStream cerr;
    static QTextStream cout;
};
}
#define COCONFIGMSG(message)                          \
    {                                                 \
        covise::coConfigLog::cout << message << endl; \
    }

#define COCONFIGLOG(message)                          \
    {                                                 \
        covise::coConfigLog::cerr << message << endl; \
    }

#define COCONFIGDBG(message)                                                 \
    {                                                                        \
        if (covise::coConfig::getDebugLevel() == covise::coConfig::DebugAll) \
        {                                                                    \
            covise::coConfigLog::cerr << message << endl;                    \
        }                                                                    \
    }
#define COCONFIGDBG_GET_SET(message)                                             \
    {                                                                            \
        if (covise::coConfig::getDebugLevel() >= covise::coConfig::DebugGetSets) \
        {                                                                        \
            covise::coConfigLog::cerr << message << endl;                        \
        }                                                                        \
    }
#define COCONFIGDBG_DEFAULT(message)                      \
    {                                                     \
        if (covise::coConfig::isDebug())                  \
        {                                                 \
            covise::coConfigLog::cerr << message << endl; \
        }                                                 \
    }

#include <config/coConfig.h>

#endif
