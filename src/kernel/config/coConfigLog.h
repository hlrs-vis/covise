/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGLOG_H
#define COCONFIGLOG_H

#include <util/coTypes.h>
#include <iostream>
namespace covise
{
}
#define COCONFIGMSG(message)               \
    {                                      \
        std::cout << message << std::endl; \
    }

#define COCONFIGLOG(message)               \
    {                                      \
        std::cerr << message << std::endl; \
    }

#define COCONFIGDBG(message)                                                 \
    {                                                                        \
        if (covise::coConfig::getDebugLevel() == covise::coConfig::DebugAll) \
        {                                                                    \
            std::cerr << message << std::endl;                               \
        }                                                                    \
    }
#define COCONFIGDBG_GET_SET(message)                                             \
    {                                                                            \
        if (covise::coConfig::getDebugLevel() >= covise::coConfig::DebugGetSets) \
        {                                                                        \
            std::cerr << message << std::endl;                                   \
        }                                                                        \
    }
#define COCONFIGDBG_DEFAULT(message)           \
    {                                          \
        if (covise::coConfig::isDebug())       \
        {                                      \
            std::cerr << message << std::endl; \
        }                                      \
    }

#include <config/coConfig.h>

#endif
