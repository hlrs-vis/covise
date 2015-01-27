/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_LOG_H
#define CO_LOG_H

#include "coExport.h"

/*
 $Log: covise_global.h,v $
 * Revision 1.1  1993/09/25  20:45:10  zrhk0125
 * Initial revision
 *
 */

namespace covise
{

UTILEXPORT void print_exit(int, const char *, int);
UTILEXPORT void print_error(int line, const char *file, const char *fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 3, 4)))
#endif
    ;

UTILEXPORT void print_comment(int line, const char *file, const char *fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 3, 4)))
#endif
    ;
UTILEXPORT void print_time(const char *);
}
#endif
