/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <cstring>
#include <cassert>
#include "vvCommandLine.h"

namespace vive
{

int vvCommandLine::s_argc = 0;
char **vvCommandLine::s_argv = NULL;
vvCommandLine *vvCommandLine::s_instance = NULL;

vvCommandLine::vvCommandLine(int argc, char *argv[])
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

vvCommandLine::~vvCommandLine()
{
    s_instance = nullptr;
}

vvCommandLine *vvCommandLine::instance()
{
    return s_instance;
}


void vvCommandLine::destroy()
{
    delete s_instance;
    s_instance = nullptr;
}
int &vvCommandLine::argc()
{
    return s_argc;
}

char **vvCommandLine::argv()
{
    return s_argv;
}

char *vvCommandLine::argv(int i)
{
    if (i < s_argc)
        return s_argv[i];

    return NULL;
}

void vvCommandLine::shift(int amount)
{
    s_argc -= amount;
    for (int i = 1; i < s_argc; ++i)
        s_argv[i] = s_argv[i + amount];
}

std::ostream &operator<<(std::ostream &os, const vvCommandLine &cmd)
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
