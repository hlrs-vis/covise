// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef _VVLOG_H_
#define _VVLOG_H_

#include "vvexport.h"

#include <iosfwd>
#include <sstream>
#include <string>


namespace virvo
{
namespace logging
{


// Returns the current logging level.
VVFILEIOAPI int getLevel();

// Sets the new logging level.
VVFILEIOAPI int setLevel(int level);

// Returns whether the given logging level is active.
VVFILEIOAPI int isActive(int level);


class Output
{
public:
    virtual ~Output() {}
    virtual void message(int level, std::string const& str) = 0;
};

// Sets the new output function.
// Use setOutput(nullptr) to log to std::clog.
VVFILEIOAPI int setOutput(Output* func);


class Stream
{
    std::ostringstream stream_;
    int level_;

public:
    VVFILEIOAPI Stream(int level, char const* file, int line);
    VVFILEIOAPI ~Stream();

    inline std::ostream& stream() {
        return stream_;
    }
};

class ErrorStream
{
    std::ostringstream stream_;

public:
    VVFILEIOAPI ErrorStream(int level, char const* file, int line);
    VVFILEIOAPI ~ErrorStream();

    inline std::ostream& stream() {
        return stream_;
    }
};

class NullStream
{
public:
    template<typename T>
    inline NullStream& operator <<(T const& /*object*/) {
        return *this;
    }

    inline NullStream& operator <<(std::ostream& (* /*manip*/)(std::ostream&)) {
        return *this;
    }
};

// This class is used to explicitly ignore values in the conditional
// logging macros. This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class Voidify
{
public:
    Voidify()
    {}

    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    template<typename T> void operator &(T&)
    {}
};


} // namespace logging
} // namespace virvo


#define VV_LOGSTREAM(NAME, LEVEL) \
    ::virvo::logging::Voidify() \
        & ::virvo::logging::NAME(LEVEL, __FILE__, __LINE__).stream()


// Log if the given level is active.
#define VV_LOG(LEVEL) \
    !::virvo::logging::isActive(LEVEL) ? void(0) : VV_LOGSTREAM(Stream, LEVEL)


#ifndef NDEBUG

// Log if the given level is active - but only in DEBUG builds!
#define VV_DLOG(LEVEL) \
    VV_LOG(LEVEL)

#else

// Log if the given level is active - but only in DEBUG builds!
#define VV_DLOG(LEVEL) \
    true ? void(0) : ::virvo::logging::Voidify() & ::virvo::logging::NullStream()

#endif


// Log and append a description of the last system error code.
#define VV_LOG_ERROR() \
    false ? void(0) : VV_LOGSTREAM(ErrorStream, -1)

#endif // _VVLOG_H_

