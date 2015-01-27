/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>

#include "control_define.h"

#ifndef BOOLDEF
#define BOOLDEF
typedef int C_boolean;
#ifdef _WIN32
typedef struct tm tms_strct;
/* typedef struct tms tms_strct; */
#endif
typedef int BOOL;
/*
 typedef enum BOOL { BFALSE, BTRUE };
*/

//typedef enum STATES { S_TRUE , S_INIT , S_READY , S_CONN , S_RUNNING , S_FINISHED, S_OLD, S_NEW, S_OPT };
#define S_TRUE 1
#define S_INIT 2
#define S_READY 3
#define S_CONN 4
#define S_RUNNING 5
#define S_FINISHED 6
#define S_OLD 7
#define S_NEW 8
#define S_OPT 9

#define MODULE_IDLE 0
#define MODULE_RUNNING 1
#define MODULE_START 2
#define MODULE_STOP 3
#endif
