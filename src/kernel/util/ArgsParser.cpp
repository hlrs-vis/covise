/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS ArgsParser
//
// This class @@@
//
// Initial version: 2003-01-27 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "ArgsParser.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <util/unixcompat.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

namespace covise
{

ArgsParser::ArgsParser(int argc, const char *const *argv)
{
    int i;
    d_argc = argc;
    d_firstArg = 1;
    d_argv = new char *[argc];
    for (i = 0; i < argc; i++)
    {
        d_argv[i] = strcpy(new char[1 + strlen(argv[i])], argv[i]);

        /// long-style options @@@ must have no parameter
        if (strstr(d_argv[i], "--") == d_argv[i])
        {
            d_firstArg = i + 1;
        }

        /// short-style options @@@ must have exactly 1 parameterelse
        // das ist falsch, es kann auch ein Switch sein!!
        else if (strstr(d_argv[i], "-") == d_argv[i])
        {
            d_firstArg = i + 2;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ArgsParser::~ArgsParser()
{
    int i;
    for (i = 0; i < d_argc; i++)
        delete[] d_argv[i];
    delete d_argv;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

const char *ArgsParser::getOpt(const char *shortOpt,
                               const char *longOpt,
                               const char *defaultVal)
{
    int i;
    for (i = 0; i < d_argc; i++)
    {
        const char *currentArg = d_argv[i];
        const char *optVal;

        //// long-format options "--longopt=something"
        if (longOpt
            && 0 == strncmp(currentArg, longOpt, strlen(longOpt)))
        {
            optVal = strchr(currentArg, '=');
            if (!optVal)
            {
                std::cerr << "Option: " << currentArg
                          << "requires '=<value>" << std::endl;
                return NULL;
            }
            else
                return optVal + 1; // return value after '=' char
        }

        //// short-format options "-opt something"
        else if (shortOpt
                 && 0 == strcasecmp(currentArg, shortOpt))
        {
            if (i < d_argc - 1)
                return d_argv[i + 1];
            else
            {
                std::cerr << "Option " << currentArg
                          << " requires arguments" << std::endl;
                return NULL;
            }
        }
    }
    return defaultVal;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool ArgsParser::getSwitch(const char *shortOpt, const char *longOpt)
{
    int currentArg;
    for (currentArg = 0; currentArg < d_argc; currentArg++)
    {
        //// long-format options "--longopt=something"
        if (longOpt
            && 0 == strcasecmp(d_argv[currentArg], longOpt))
        {
            return true; // return value after '=' char
        }

        //// short-format options "-opt something"
        else if (shortOpt
                 && 0 == strcasecmp(d_argv[currentArg], shortOpt))
        {
            return true;
        }
    }
    return false;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

const char *ArgsParser::operator[](int idx)
{
    if (idx + d_firstArg < d_argc)
        return d_argv[idx + d_firstArg];
    else
        return NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ArgsParser::numArgs()
{
    return d_argc - d_firstArg;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}
