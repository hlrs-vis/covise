/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_STATE_H
#define _LIBAPPL_APPL_STATE_H

/* States of an application */

#define APPL_CREATED 0x0000
#define APPL_INITED 0x0001
#define APPL_RECONNECTING 0x0002

#define APPL_STARTING 0x0010
#define APPL_STARTED 0x0011

#define APPL_HALTING 0x0020
#define APPL_HALTED 0x0021

#define APPL_ENDING 0x0030
#define APPL_ENDED 0x0031

#define APPL_DISCONNECTING 0x0040
#define APPL_DISCONNECTED 0x0041

/* Working modes of an application */

#define APPL_TRY_RECONNECT 0x1000

#endif
