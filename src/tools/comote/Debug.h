/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Debug.h

#ifndef DEBUG_H
#define DEBUG_H

#include <string>
#include <sstream>
#ifndef _MSC_VER
#include <assert.h>
#endif

#define UNUSED(var) ((void)(var))

#ifndef NDEBUG
#ifdef _MSC_VER
#define ASSERT(x) ((void)((x) || (__debugbreak(), 1)))
#else
#define ASSERT(x) assert(x)
#endif
#else
#define ASSERT(x) ((void)(0))
#endif

#ifndef NDEBUG
#define Log() Logger().Get()
#else
#define Log() \
    if (1)    \
    {         \
    }         \
    else      \
    Logger().Get()
#endif

class Logger
{
public:
    Logger();
    ~Logger();

    inline std::ostream &Get()
    {
        return m_stream;
    }

private:
    std::ostringstream m_stream;
};
#endif
