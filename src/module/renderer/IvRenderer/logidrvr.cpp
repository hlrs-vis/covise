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
 * ABSTRACT: Driver for the Logitech 3D mouse. This driver is based on
 *	    Logitech's IBM-PC driver, "logidrvr.c" (by Jim Barnes, Logitech),
 *	    but does not have full functionality. In particular, only
 *           6D mouse functions in Euler mode are supported (i.e,. currently
 *	    no quaternions or 2D mouse modes).
 *
 * REVISION HISTORY:
 *
 * $Log: logidrvr.c,v $
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

#include <stdio.h> /* perror */
#include <sys/types.h> /* open */
#include <sys/stat.h> /* open */
#include <fcntl.h> /* open */
#include <termios.h> /* tcsetattr */
#include <limits.h> /* sginap */
#include <unistd.h>
#include <sys/ioctl.h>

#include <iostream>
using std::cerr;
using std::endl;
#include "logidrvr.h" /* function prototypes and data types */

#ifdef __hpux
#define NO_TERMIOS
#endif
#ifdef __APPLE__
#define NO_TERMIOS
#endif

/********************** local function prototypes ****************************/
static void euler_to_absolute(byte record[], MouseRecordPtr data);
/*static void print_bin (char a);*/

/* define DEBUG for debugging statements */
/*
#define DEBUG
*/

/******************************************************************************
 *
 * logitech_open - Connect the mouse by opening a serial port
 *		  (19200 baud, 8 data, 1 stop, no parity) and verifying
 *		  diagnostics.
 *
 * INPUTS:
 *   port_name	serial port name (e.g., "/dev/ttyd3")
 *
 * RETURNS:
 *   On success, a file descriptor to serial port or -1 if error opening port.
 *
 ******************************************************************************/
int
logitech_open(char *port_name)
{
#ifndef NO_TERMIOS
    int fd; /* file descriptor */
    struct termios t; /* termio struct */
    char data[DIAGNOSTIC_SIZE]; /* for diagnostics info */

    /* open a serial port, read/write */
    if ((fd = open(port_name, O_RDWR | O_NDELAY)) < 0)
    {
        perror(port_name);
        return (-1);
    }

    /* disable all input mode processing */
    t.c_iflag = 0;

    /* disable all output mode processing */
    t.c_oflag = 0;

    /* hardware control flags: 19200 baud, 8 data bits, 1 stop bits,
      no parity, enable receiver */
    t.c_cflag = B19200 | CS8 | CSTOPB | CREAD;

    /* disable local control processing (canonical, control sigs, etc) */
    t.c_lflag = 0;

    /* set control characters for non-canonical reads: VMIN = 1, VTIME = 0
      i.e., read not satisfied until at least 1 char is read, see termio(7) */
    t.c_cc[VMIN] = 1;
    t.c_cc[VTIME] = 0;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
        return (-1);
    }

    /* do diagnostics, results are in "data" */
    get_diagnostics(fd, data);

#ifdef DEBUG
    printf("diag[0]: %2x=", data[0]);
    print_bin(data[0]);
    printf("\n");
    printf("diag[1]: %2x=", data[1]);
    print_bin(data[1]);
    printf("\n");
#endif

    /* check diagnostic return */
    if ((data[0] != (char)0xbf) || (data[1] != (char)0x3f))
    {
        fprintf(stderr, "Mouse diagnostics failed\n");
        return (-1);
    }
    return (fd);
#else
    return -1;
#endif
}

/******************************************************************************
 *
 * logitech_close - Close the mouse by closing the serial port.
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   0 on success, -1 on failure.
 *
 ******************************************************************************/
int
logitech_close(int fd)
{
    if (close(fd) < 0)
    {
        perror("error closing serial port");
        return (-1);
    }
    else
        return (0);
}

/******************************************************************************
 *
 * cu_incremental_reporting -  Command incremental reporting
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_incremental_reporting(int fd)
{
#ifndef NO_TERMIOS

#ifdef DEBUG
    printf("incremental reporting enabled\n");
#endif
    struct termios t;

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete euler record packet */
    t.c_cc[VMIN] = EULER_RECORD_SIZE;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*I", 2) != 2)
    {
        cerr << "short write1" << endl;
    }
#endif
}

/******************************************************************************
 *
 * cu_demand_reporting -  Command demand reporting
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_demand_reporting(int fd)
{
#ifndef NO_TERMIOS

#ifdef DEBUG
    printf("demand reporting enabled\n");
#endif
    struct termios t;

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete euler record packet */
    t.c_cc[VMIN] = EULER_RECORD_SIZE;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*D", 2) != 2)
    {
        cerr << "short write2" << endl;
    }
#endif
}

/******************************************************************************
 *
 * cu_euler_mode - Command control unit to Euler mode
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_euler_mode(int fd)
{

#ifdef DEBUG
    printf("euler data mode enabled\n");
#endif

    if (write(fd, "*G", 2) != 2)
    {
        cerr << "short write3" << endl;
    }
}

/******************************************************************************
 *
 * cu_headtracker_mode - Command control unit to head tracker mode
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_headtracker_mode(int fd)
{

#ifdef DEBUG
    printf("headtracking mode enabled\n");
#endif

    if (write(fd, "*H", 2) != 2)
    {
        cerr << "short write4" << endl;
    }
}

/******************************************************************************
 *
 * cu_mouse_mode - Command control unit to mouse mode
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_mouse_mode(int fd)
{

#ifdef DEBUG
    printf("mouse mode enabled\n");
#endif

    if (write(fd, "*h", 2) != 2)
    {
        cerr << "short write5" << endl;
    }
}

/******************************************************************************
 *
 * cu_request_diagnostics - Command control unit to perform diagnostics
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_request_diagnostics(int fd)
{

#ifndef NO_TERMIOS
#ifdef DEBUG
    printf("performing diagnostics\n");
#endif
    struct termios t;

    tcgetattr(fd, &t);

    /* set control characters for non-canonical reads: VMIN, VTIME
      i.e., read a complete diagnostics packet */
    t.c_cc[VMIN] = DIAGNOSTIC_SIZE;
    t.c_cc[VTIME] = 1;

    /* control port immediately (TCSANOW) */
    if (tcsetattr(fd, TCSANOW, &t) < 0)
    {
        perror("error controlling serial port");
    }

    if (write(fd, "*\05", 2) != 2)
    {
        cerr << "short write6" << endl;
    }
#endif
}

/******************************************************************************
 *
 * cu_request_report - Demand a single report
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void cu_request_report(int fd)
{

#ifdef DEBUG
    printf("asking for a single report\n");
#endif

    if (write(fd, "*d", 2) != 2)
    {
        cerr << "short write7" << endl;
    }
}

/******************************************************************************
 *
 * cu_reset_control_unit - Command a reset
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
cu_reset_control_unit(int fd)
{

#ifdef DEBUG
    printf("resetting control unit\n");
#endif

    if (write(fd, "*R", 2) != 2)
    {
        cerr << "short write8" << endl;
    }
}

/******************************************************************************
 *
 * get_diagnostics - retrieve diagnostics report
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
get_diagnostics(int fd, char data[])
{
    cu_request_diagnostics(fd); /* command diagnostics */
    /* sginap (100);			/ * wait 1 second */
    if (read(fd, data, DIAGNOSTIC_SIZE) != DIAGNOSTIC_SIZE)
    {
        cerr << "short read1" << endl;
    }
}

/******************************************************************************
 *
 * get_record - Retrieve a single record. This routine will spin until a
 *              valid record (i.e., 16 bytes and bit 7, byte 0 is on) is
 *	       received.
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * OUTPUTS:
 *   data	pointer to MouseRecord storage
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
get_record(int fd, MouseRecordPtr data)
{
#ifndef NO_TERMIOS
    int num_read;
    byte record[EULER_RECORD_SIZE];

    cu_request_report(fd);
    num_read = read(fd, record, EULER_RECORD_SIZE);

    /* if didn't get a complete record or if invalid record, then try
      to get a good one */
    while ((num_read < EULER_RECORD_SIZE) || !(record[0] & logitech_FLAGBIT))
    {

/* printf("get_record: only got %d bytes\n", num_read); */

/* flush the buffer */
#ifdef __linux
        ioctl(fd, CFLUSH, 0);
#else
        ioctl(fd, TCFLSH, 0);
#endif

        cu_request_report(fd);
        num_read = read(fd, record, EULER_RECORD_SIZE);
    }

#ifdef DEBUG
    printf("%d bytes read...", num_read);
#endif

    /* convert the raw euler record to absolute record */
    euler_to_absolute(record, data);
#endif
}

/******************************************************************************
 *
 * reset_control_unit - Set control unit into demand reporting, send
 *		       reset command, and wait for the reset.
 *
 * INPUTS:
 *   fd		file descriptor to serial port
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
void
reset_control_unit(int fd)
{
#ifndef NO_TERMIOS
#if defined(__hpux) || defined(__linux)
    (void)fd;
#else
    cu_demand_reporting(fd); /* make sure control unit is processing */
    sginap((long)10); /* wait 10 clock ticks = 100 ms */
    cu_reset_control_unit(fd); /* command a reset */
    sginap((long)100); /* wait 1 second */
#endif
#endif
}

/******************************************************************************
 *
 * euler_to_absolute - convert from raw Euler data record to absolute data
 *
 * INPUTS:
 *   record	raw logitech record
 *
 * OUTPUTS
 *   data	pointer to MouseRecord storage
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
static void
euler_to_absolute(byte record[], MouseRecordPtr data)
{
    long ax, ay, az, arx, ary, arz;

    data->buttons = (byte)record[0];

    ax = (record[1] & 0x40) ? 0xFFE00000 : 0;
    ax |= (long)(record[1] & 0x7f) << 14;
    ax |= (long)(record[2] & 0x7f) << 7;
    ax |= (record[3] & 0x7f);

    ay = (record[4] & 0x40) ? 0xFFE00000 : 0;
    ay |= (long)(record[4] & 0x7f) << 14;
    ay |= (long)(record[5] & 0x7f) << 7;
    ay |= (record[6] & 0x7f);

    az = (record[7] & 0x40) ? 0xFFE00000 : 0;
    az |= (long)(record[7] & 0x7f) << 14;
    az |= (long)(record[8] & 0x7f) << 7;
    az |= (record[9] & 0x7f);

    data->x = ((float)ax) / 1000.0;
    data->y = ((float)ay) / 1000.0;
    data->z = ((float)az) / 1000.0;

    arx = (record[10] & 0x7f) << 7;
    arx += (record[11] & 0x7f);

    ary = (record[12] & 0x7f) << 7;
    ary += (record[13] & 0x7f);

    arz = (record[14] & 0x7f) << 7;
    arz += (record[15] & 0x7f);

    data->pitch = ((float)arx) / 40.0;
    data->yaw = ((float)ary) / 40.0;
    data->roll = ((float)arz) / 40.0;

#ifdef DEBUG
    printf("raw: %ld %ld %ld %ld %ld %ld\n", ax, ay, az, arx, ary, arz);
    printf("%7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %7.2f\n",
           data->x, data->y, data->z, data->pitch, data->yaw, data->roll);
#endif
}

/******************************************************************************
 *
 * print_bin - print an 8-bit binary string
 *
 * INPUTS:
 *   a		char
 *
 * RETURNS:
 *   none
 *
 ******************************************************************************/
/*static void
print_bin (char a)
{
    
   int i;
    for (i=7; i>=0; i--)
      printf ("%c", (a&(1<<i)) ? '1' : '0');
}
*/
