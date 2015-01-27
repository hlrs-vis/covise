/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_HELPER_H
#define _LIBAPPL_APPL_HELPER_H

#include <appl/appl.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define APPL_LEVEL_NONE 0x0000
#define APPL_LEVEL_NORMAL 0x0001
#define APPL_LEVEL_INFO 0x0002
#define APPL_LEVEL_DEBUG 0x0003

EXTERN int appl_printf(struct _appl_t *appl, int level, const char *format, ...);
EXTERN char *appl_class_name(char *argv0);

#endif
