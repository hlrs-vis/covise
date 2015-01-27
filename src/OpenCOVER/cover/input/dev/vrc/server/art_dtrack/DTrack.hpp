/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* DTrackSDK: C++ header file, A.R.T. GmbH 21.4.05-3.7.07
 *
 * DTrack: functions to receive and process DTrack UDP packets (ASCII protocol)
 * Copyright (C) 2005-2007, Advanced Realtime Tracking GmbH
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *
 * Version v1.3.2
 *
 * Purpose:
 *  - receives DTrack UDP packets (ASCII protocol) and converts them into easier to handle data
 *  - sends DTrack remote commands (UDP)
 *  - DTrack network protocol due to: 'Technical Appendix DTrack v1.24 (December 19, 2006)'
 *  - for DTrack versions v1.16 - v1.24 (and compatible versions)
 *  - tested under Linux (gcc) and MS Windows 2000/XP (MS Visual C++)
 *
 * Usage:
 *  - for Linux, Unix:
 *    - comment '#define OS_WIN', uncomment '#define OS_UNIX' in file 'DTrack.cpp'
 *  - for MS Windows:
 *    - comment '#define OS_UNIX', uncomment '#define OS_WIN' in file 'DTrack.cpp'
 *    - link with 'ws2_32.lib'
 *
 * $Id: DTrack.hpp,v 1.6 2007/07/03 16:19:57 kurt Exp $
 */

#ifndef _ART_DTRACK_H
#define _ART_DTRACK_H

#include <vector>

// --------------------------------------------------------------------------
// Data types

// Standard body data (6DOF):
//  - currently not tracked bodies are getting a quality of -1

typedef struct
{
    int id; // id number (starting with 0)
    float quality; // quality (0 <= qu <= 1, no tracking if -1)

    float loc[3]; // location (in mm)
    float rot[9]; // rotation matrix (column-wise)
} dtrack_body_type;

// A.R.T. Flystick data (6DOF + buttons):
//  - currently not tracked bodies are getting a quality of -1
//  - note the maximum number of buttons and joystick values

#define DTRACK_FLYSTICK_MAX_BUTTON 16 // maximum number of buttons
#define DTRACK_FLYSTICK_MAX_JOYSTICK 8 // maximum number of joystick values

typedef struct
{
    int id; // id number (starting with 0)
    float quality; // quality (0 <= qu <= 1, no tracking if -1)

    int num_button; // number of buttons
    int button[DTRACK_FLYSTICK_MAX_BUTTON]; // button state (1 pressed, 0 not pressed)
    // (0 front, 1..n-1 right to left)
    int num_joystick; // number of joystick values
    float joystick[DTRACK_FLYSTICK_MAX_JOYSTICK]; // joystick value (-1 <= joystick <= 1)
    // (0 horizontal, 1 vertical)

    float loc[3]; // location (in mm)
    float rot[9]; // rotation matrix (column-wise)
} dtrack_flystick_type;

// Measurement tool data (6DOF + buttons):
//  - currently not tracked bodies are getting a quality of -1
//  - note the maximum number of buttons

#define DTRACK_MEATOOL_MAX_BUTTON 1 // maximum number of buttons

typedef struct
{
    int id; // id number (starting with 0)
    float quality; // quality (0 <= qu <= 1, no tracking if -1)

    int num_button; // number of buttons
    int button[DTRACK_MEATOOL_MAX_BUTTON]; // button state (1 pressed, 0 not pressed)

    float loc[3]; // location (in mm)
    float rot[9]; // rotation matrix (column-wise)
} dtrack_meatool_type;

// A.R.T. Fingertracking hand data (6DOF + fingers):
//  - currently not tracked bodies are getting a quality of -1

#define DTRACK_HAND_MAX_FINGER 5 // maximum number of fingers

typedef struct
{
    int id; // id number (starting with 0)
    float quality; // quality (0 <= qu <= 1, no tracking if -1)

    int lr; // left (0) or right (1) hand
    int nfinger; // number of fingers (maximum 5)

    float loc[3]; // back of the hand: location (in mm)
    float rot[9]; // back of the hand: rotation matrix (column-wise)

    struct
    {
        float loc[3]; // finger: location (in mm)
        float rot[9]; // finger: rotation matrix (column-wise)

        float radiustip; // finger: radius of tip
        float lengthphalanx[3]; // finger: length of phalanxes; order: outermost, middle, innermost
        float anglephalanx[2]; // finger: angle between phalanxes
    } finger[DTRACK_HAND_MAX_FINGER]; // order: thumb, index finger, middle finger, ...
} dtrack_hand_type;

// Single marker data (3DOF):

typedef struct
{
    int id; // id number (starting with 1)
    float quality; // quality (0 <= qu <= 1)

    float loc[3]; // location (in mm)
} dtrack_marker_type;

// --------------------------------------------------------------------------
// Library class

class DTrack
{
public:
    // Constructor:
    //
    // udpport (i): UDP port number to receive data from DTrack
    //
    // remote_host (i): DTrack remote control: hostname or IP address of DTrack PC (NULL if not used)
    // remote_port (i): port number of DTrack remote control (0 if not used)
    //
    // udpbufsize (i): size of buffer for UDP packets (in bytes)
    // udptimeout_us (i): UDP timeout (receiving and sending) in us (micro second)

    DTrack(
        int udpport = 5000, const char *remote_host = NULL, int remote_port = 0,
        int udpbufsize = 20000, int udptimeout_us = 1000000);

    // Destructor:

    ~DTrack(void);

    // Check if initialization was successfull:
    //
    // return value (o): boolean

    bool valid(void);

    // Check last receive/send error:
    //
    // return value (o): boolean

    bool timeout(void); // 'timeout'
    bool udperror(void); // 'udp error'
    bool parseerror(void); // 'parse error'

    // Receive and process one DTrack data packet (UDP; ASCII protocol):
    //
    // return value (o): receiving was successfull

    bool receive(void);

    // Get data of last received DTrack data packet:
    //  - currently not tracked bodies are getting a quality of -1

    unsigned int get_framecounter(void); // frame counter
    double get_timestamp(void); // timestamp (-1 if information not available)

    int get_num_body(void); // number of calibrated standard bodies (as far as known)
    dtrack_body_type get_body(int id); // standard body data (id (i): standard body id 0..max-1)

    int get_num_flystick(void); // number of calibrated Flysticks
    dtrack_flystick_type get_flystick(int id); // Flystick data (id (i): Flystick id 0..max-1)

    int get_num_meatool(void); // number of calibrated measurement tools
    dtrack_meatool_type get_meatool(int id); // measurement tool data (id (i): tool id 0..max-1)

    int get_num_hand(void); // number of calibrated Fingertracking hands (as far as known)
    dtrack_hand_type get_hand(int id); // Fingertracking hand data (id (i): hand id 0..max-1)

    int get_num_marker(void); // number of tracked single markers
    dtrack_marker_type get_marker(int index); // single marker data (index (i): index 0..max-1)

    // Send remote control commands (UDP; ASCII protocol) to DTrack:
    //
    // onoff (i): switch function on or off
    //
    // return value (o): sending of remote commands was successfull

    bool cmd_cameras(bool onoff); // control cameras
    bool cmd_tracking(bool onoff); // control tracking calculation (default: on)
    bool cmd_sending_data(bool onoff); // control sending of UDP output data (default: on)

    // frames (i): number of frames

    bool cmd_sending_fixed_data(int frames); // start sending of a fixed number of UDP output frames

private:
    void *d_udpsock; // socket number for UDP
    int d_udptimeout_us; // timeout for receiving and sending UDP data

    int d_udpbufsize; // size of UDP buffer
    char *d_udpbuf; // UDP buffer

    unsigned int d_remote_ip; // DTrack remote command access: IP address
    unsigned short d_remote_port; // DTrack remote command access: port number
    bool d_remote_cameras; // DTrack status: cameras on/off
    bool d_remote_tracking; // DTrack status: tracking on/off
    bool d_remote_sending; // DTrack status: sending of UDP output data on/off

    unsigned int act_framecounter; // frame counter
    double act_timestamp; // timestamp (-1, if information not available)

    int act_num_body; // number of calibrated standard bodies (as far as known)
    std::vector<dtrack_body_type> act_body; // array containing standard body data

    int act_num_flystick; // number of calibrated Flysticks
    std::vector<dtrack_flystick_type> act_flystick; // array containing Flystick data

    int act_num_meatool; // number of calibrated measurement tools
    std::vector<dtrack_meatool_type> act_meatool; // array containing measurement tool data

    int act_num_hand; // number of calibrated Fingertracking hands (as far as known)
    std::vector<dtrack_hand_type> act_hand; // array containing Fingertracking hands data

    int act_num_marker; // number of tracked single markers
    std::vector<dtrack_marker_type> act_marker; // array containing single marker data

    int d_lasterror; // last receive/send error

    void set_noerror(void); // set last receive/send error to 'no error'
    void set_timeout(void); // set last receive/send error to 'timeout'
    void set_udperror(void); // set last receive/send error to 'udp error'
    void set_parseerror(void); // set last receive/send error to 'parse error'

    bool cmd_send(int cmd, int val = 0); // send remote control command
};

// ---------------------------------------------------------------------------------------------------

#endif // _ART_DTRACK_H
