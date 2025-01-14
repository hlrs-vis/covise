/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coExport.h>
#include <ostream>

namespace vive
{
class VVCORE_EXPORT vvCommandLine
{
public:
    vvCommandLine(int argc, char *argv[]);
    ~vvCommandLine();
    static vvCommandLine *instance();
    static void destroy();
    static int &argc();
    static char **argv();
    static char *argv(int i);
    static void shift(int amount = 1);

private:
    static int s_argc;
    static char **s_argv;
    static vvCommandLine *s_instance;
};

std::ostream &operator<<(std::ostream &os, const vvCommandLine &cmd);
}
