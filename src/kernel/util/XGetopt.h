/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// XGetopt.h  Version 1.2
//
// Author:  Hans Dietrich
//          hdietrich2@hotmail.com
//
// This software is released into the public domain.
// You are free to use it in any way you like.
//
// This software is provided "as is" with no expressed
// or implied warranty.  I accept no liability for any
// damage or loss of business that this software may cause.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef XGETOPT_H
#define XGETOPT_H
#include <tchar.h>
#include "coExport.h"

namespace covise
{

extern UTILEXPORT int optind, opterr;
extern UTILEXPORT TCHAR *optarg;

int UTILEXPORT getopt(int argc, TCHAR *argv[], TCHAR *optstring);
}
#endif //XGETOPT_H
