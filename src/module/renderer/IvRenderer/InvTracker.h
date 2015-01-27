/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INVTRACKER_H
#define _INVTRACKER_H

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  Tracking Extension		                          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  09.09.94  V1.0                                                  **
\**************************************************************************/

typedef unsigned char byte;

#if 0
typedef struct
{
   byte buttons;
   float x;
   float y;
   float z;
   float pitch;
   float yaw;
   float roll;
   float xr;
   float yr;
   float zr;
   float ar;
} MouseRecordType, *MouseRecordPtr;
#endif

#if 0
extern "C"
{

   int  fastrak_open (char *port_name);
   int  fastrak_close (int fd_mouse);

   void fastrak_euler_mode (int fd);
   void fastrak_get_record (int fd, MouseRecordPtr data);
   void fastrak_demand_reporting (int fd);

   int  logitech_open (char *port_name);
   int  logitech_close (int fd_mouse);

   void cu_incremental_reporting (int fd);
   void cu_demand_reporting (int fd);
   void cu_euler_mode (int fd);
   void cu_headtracker_mode (int fd);
   void cu_mouse_mode (int fd);
   void cu_request_diagnostics (int fd);
   void cu_request_report (int fd);
   void cu_reset_control_unit (int fd);

   void get_diagnostics (int fd, char data[]);
   void get_record (int fd, MouseRecordPtr data);
   void reset_control_unit (int fd);

}
#endif
#endif // _INVTRACKER_H
