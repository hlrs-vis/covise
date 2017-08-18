/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <cstring>
#include <cassert>
#include "coCommandLine.h"

namespace opencover
{

int coCommandLine::s_argc = 0;
char **coCommandLine::s_argv = NULL;
coCommandLine *coCommandLine::s_instance = NULL;

coCommandLine::coCommandLine(int argc, char *argv[])
{
    assert(s_instance == NULL);
    s_instance = this;
    s_argc = argc;
    s_argv = argv;
#ifdef __APPLE__
    if (argc >= 2 && !strncmp(argv[1], "-psn_", 5))
        shift(1);
#endif
}

coCommandLine *coCommandLine::instance()
{
    return s_instance;
}

int &coCommandLine::argc()
{
    return s_argc;
}

char **coCommandLine::argv()
{
    return s_argv;
}

char *coCommandLine::argv(int i)
{
    if (i < s_argc)
        return s_argv[i];

    return NULL;
}

void coCommandLine::shift(int amount)
{
    s_argc -= amount;
    for (int i = 1; i < s_argc; ++i)
        s_argv[i] = s_argv[i + amount];
}

std::ostream &operator<<(std::ostream &os, const coCommandLine &cmd)
{
    for (int i = 0; i < cmd.argc(); ++i)
    {
        if (i)
            os << " ";
        os << cmd.argv(i);
    }
    return os;
}
}
