/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

typedef int T_Bool;

bool AVRInit(const char *device, int baud_rate);
bool AVRClose();
bool AVRSendBytes(const char *command, int c_length);
bool AVRReadBytes(unsigned int n, unsigned char *out);
