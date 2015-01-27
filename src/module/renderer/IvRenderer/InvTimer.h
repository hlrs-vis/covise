/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TIMER_H
#define _INV_TIMER_H

//**************************************************************************
//
// * Description    : This is the error output handler for the renderer
//
//
// * Class(es)      : none
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 28.09.94
//
//
//
//**************************************************************************

#ifdef TIMING
extern CoviseTime *covise_time;
char *time_str;

long telePointer_receive_ctr = 0;
long transform_receive_ctr = 0;
long telePointer_send_ctr = 0;
long transform_send_ctr = 0;

ApplicationProcess *ap;
#endif
#endif
