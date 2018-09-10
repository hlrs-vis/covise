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

#include "vvlog.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <iostream>


using virvo::logging::Output;
using virvo::logging::Stream;
using virvo::logging::ErrorStream;


class AutoOutput
{
    Output* func;

public:
    AutoOutput() : func(0) {}
   ~AutoOutput() { delete func; }

    Output* operator ->() const { return func; }

    void reset(Output* newFunc) {
        delete func;
        func = newFunc;
    }

private:
    typedef Output* AutoOutput::* bool_type;
public:
    operator bool_type() const {
        return func ? &AutoOutput::func : 0;
    }
};


static int GetLogLevelFromEnv()
{
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
    if (char* env = std::getenv("VV_DEBUG"))
        return std::atoi(env);

    return 0;
}


// The current logging level
static int LogLevel = GetLogLevelFromEnv();

// The current logging destination
static AutoOutput LogOutput;


static void PrintMessage(int level, std::string const& str)
{
    if (LogOutput)
        LogOutput->message(level, str);
    else
        std::clog << str << std::endl;
}


int virvo::logging::getLevel()
{
    return LogLevel;
}


int virvo::logging::setLevel(int level)
{
    LogLevel = level;
    return 0;
}


int virvo::logging::isActive(int level)
{
    return level <= LogLevel;
}


int virvo::logging::setOutput(Output* func)
{
    LogOutput.reset(func);
    return 0;
}


Stream::Stream(int level, char const* /*file*/, int /*line*/)
    : level_(level)
{
}


Stream::~Stream()
{
    PrintMessage(level_, stream_.str());
}


static std::string FormatLastError()
{
#ifdef _WIN32
    LPSTR buffer = 0;

    // Retrieve the system error message for the last-error code
    FormatMessageA( FORMAT_MESSAGE_ALLOCATE_BUFFER |
                    FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                    NULL,
                    GetLastError(),
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    reinterpret_cast<LPSTR>(&buffer),
                    0,
                    NULL );

    std::string result = buffer;

    LocalFree(buffer);

    return result;
#else
    return std::strerror(errno);
#endif
}


ErrorStream::ErrorStream(int /*level*/, char const* /*file*/, int /*line*/)
{
}


ErrorStream::~ErrorStream()
{
    stream_ << ": " << FormatLastError();

    PrintMessage(-1, stream_.str());
}
