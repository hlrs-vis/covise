/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CLIENT_API_H_
#define __CLIENT_API_H_

/*

   Messages consist of:

   COMMAND PARAMETER PARAMETER PARAMETER ... PARAMETER<CR>[<LF>]

   COMMAND is a keyword from the following list
   PARAMETER is a keyword, a string (delimited by "),
   an integer (decimal or hex) or float number (with decimal point)

   The command line ends with <CR> or <CR><LF>

QUIT		Client disconnects from server

TEST [test number]
Test number:
0	Test sound with 'standard bell' and speaker beep
1	Test with 'standard bell' sound
2	Test with 'information' sound
3	Test with 'warning/exclamation' sound
4	Test with 'critical/stop' sound
5	Test with 'question' sound

PUTDATA <length> <data>
Data transmission with 'length' blocks in total,
ending when last data block arrives

PUTFILE <filename> <length>
Put file 'length' data blocks and store it into 'filename'

GETHANDLE <filename>
Get handle for sound 'filename', Server will send handle
number back to client

SETVOL <volume left> <volume right>
Set wave out devices volume level for both left and right (16 bit values)

PLAYFILE <filename>
Play sound with standard system call, e.g. PlaySound

PLAY <handle>
Play sound assigned to handle
PLAYL <handle>
Play sound assigned to handle with looping

SET_SOUND <handle> POSITION <pos_x> <pos_y> <pos_z>
SET_SOUND <handle> DIRECTION <pos_x> <pos_y> <pos_z>
SET_SOUND <handle> DIRECTION <angle_horiz> <angle_vert>
SET_SOUND <handle> VOLUME <volume>

*/

#define CMD_TEST "TEST"
#define CMD_QUIT "QUIT"

#define CMD_GET_HANDLE "GHDL"
#define CMD_RELEASE_HANDLE "RHDL"
#define CMD_PLAY "PLAY"
#define CMD_STOP "STOP"
#define CMD_SET_VOLUME "SVOL"
#define CMD_SET_EAX "SEAX"
#define CMD_PUT_FILE "PTFI"
#define CMD_PUT_DATA "PTDA"

#define SOUND_POSITION "SSPO"
#define SOUND_DIRECTION "SSDI"
#define SOUND_DIRECTION_VOLUME "SSDV"
#define SOUND_REL_DIRECTION "SSDR"

#define SOUND_REL_DIRECTION_VOLUME "SSRV"

#define SOUND_VOLUME "SSVO"
#define SOUND_VELOCITY "SSVE"
#define SOUND_LOOPING "SSLP"
#define SOUND_PITCH "SSPI"

#define EAX_ENVIRONMENT "SEEN"

#define CMD_SYNC "SYNC"
#endif
