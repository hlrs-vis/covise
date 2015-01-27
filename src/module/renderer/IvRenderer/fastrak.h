/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*****************************************************************************
 *
 * (c) Copyright 1994. National Aeronautics and Space Administration.
 *     Ames Research Center, Moffett Field, CA 94035-1000
 *
 * FILE:     logidrvr.h
 *
 * ABSTRACT: function protos and types for logidrvr.c
 *
 * REVISION HISTORY:
 *
 * $Log: logidrvr.h,v $
 * Revision 1.1  1994/05/18  01:07:08  terry
 * Initial revision
 *
 *
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the NASA Ames Research Center. The name of NASA may not be used
 * to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 ******************************************************************************/

#ifndef INCfastrakh
#define INCfastrakh

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

/* record sizes */
#define DIAGNOSTIC_SIZE 2
#define EULER_RECORD_SIZE 21

/************************* fastrakdrvr data types *******************************/
typedef unsigned char byte;

typedef struct
{
    byte buttons;
    float x;
    float y;
    float z;
    float pitch;
    float yaw;
    float roll;
} MouseRecordType, *MouseRecordPtr;

/************************* function prototypes *******************************/
int fastrak_open(char *port_name);
int fastrak_close(int fd_mouse);

void fastrak_euler_mode(int fd);
void fastrak_get_record(int fd, MouseRecordPtr data);
#endif /* INCfastrakh */
