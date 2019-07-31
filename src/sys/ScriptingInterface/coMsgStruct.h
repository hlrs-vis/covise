/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MSG_STRUCT_H
#define CO_MSG_STRUCT_H

#include <net/dataHandle.h>

typedef struct
{
    int type;
    char* data;
    covise::DataHandle dh;
} CoMsg;

#endif
