/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_COMMAND_LINE_H
#define CO_COMMAND_LINE_H

/*! \file
 \brief  store command line parameters

 \author Martin Aumueller <aumueller@uni-koeln.de>
 \author (C) 2007
         ZAIK Center for Applied Informatics,
         Robert-Koch-Str. 10, Geb. 52,
         D-50931 Koeln,
         Germany

 \date   
 */

#include <util/coExport.h>
#include <ostream>

namespace opencover
{
class COVEREXPORT coCommandLine
{
public:
    coCommandLine(int argc, char *argv[]);
    static coCommandLine *instance();
    static int &argc();
    static char **argv();
    static char *argv(int i);
    static void shift(int amount = 1);

private:
    static int s_argc;
    static char **s_argv;
    static coCommandLine *s_instance;
};

std::ostream &operator<<(std::ostream &os, const coCommandLine &cmd);
}
#endif
