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

#ifndef INClogidrvrh
#define INClogidrvrh

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

/* record sizes */
#define DIAGNOSTIC_SIZE 2
#define EULER_RECORD_SIZE 16

/* euler record "button" bits */
#define logitech_FLAGBIT 0x80
#define logitech_FRINGEBIT 0x40
#define logitech_OUTOFRANGEBIT 0x20
#define logitech_RESERVED 0x10
#define logitech_SUSPENDBUTTON 0x08
#define logitech_LEFTBUTTON 0x04
#define logitech_MIDDLEBUTTON 0x02
#define logitech_RIGHTBUTTON 0x01

/************************* logidrvr data types *******************************/
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
int logitech_open(char *port_name);
int logitech_close(int fd_mouse);

void cu_incremental_reporting(int fd);
void cu_demand_reporting(int fd);
void cu_euler_mode(int fd);
void cu_headtracker_mode(int fd);
void cu_mouse_mode(int fd);
void cu_request_diagnostics(int fd);
void cu_request_report(int fd);
void cu_reset_control_unit(int fd);

void get_diagnostics(int fd, char data[]);
void get_record(int fd, MouseRecordPtr data);
void reset_control_unit(int fd);
#endif /* INClogidrvrh */
