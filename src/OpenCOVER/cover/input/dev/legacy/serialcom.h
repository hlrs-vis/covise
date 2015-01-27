/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

typedef int T_Bool;

bool INPUT_LEGACY_EXPORT Init(char *device, int baud_rate);
bool INPUT_LEGACY_EXPORT close_port();
bool INPUT_LEGACY_EXPORT send_command(char *command, int c_length);
bool INPUT_LEGACY_EXPORT get_answer(unsigned int n, unsigned char *out);
bool INPUT_LEGACY_EXPORT getDivisionAnswer(unsigned int n, unsigned char *out);
