/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

typedef int T_Bool;

bool Init(char *device, int baud_rate);
bool close_port();
bool send_command(char *command, int c_length);
bool get_answer(unsigned int n, unsigned char *out);
bool getDivisionAnswer(unsigned int n, unsigned char *out);
