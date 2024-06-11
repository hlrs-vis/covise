/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <util/UDP_Sender.h>

using namespace std;
using namespace covise;

int main(int argc, char **argv)
{
    UDP_Sender z8("192.168.1.192", 9099);
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        transform(arg.begin(), arg.end(), arg.begin(), ::toupper);
        if (arg == "3DON")
        {
            //unsigned char buf[] = {0x74, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
            unsigned char buf[] = {0x80, 0x10, 0x01, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
            z8.send(buf, sizeof(buf));
        }
        else if (arg == "3DOFF")
        {
            //unsigned char buf[] = { 0x74, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
            unsigned char buf[] = { 0x80, 0x10, 0x01, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
            z8.send(buf, sizeof(buf));
        }
        if (arg.substr(0, 6) == "PRESET")
        {
            unsigned char buf[] = { 0x07, 0x10, 0x03, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00 };
            int num = 1;
            if (sscanf(argv[i] + 6, "%d", &num) == 1)
            {
                buf[15] = num;
                z8.send(buf, sizeof(buf));
                fprintf(stderr, "setting preset %d\n", num);
            }
        }
        if (arg.substr(0,  10) == "Brightness")
        {
            unsigned char buf[] = { 0x50, 0x10, 0x00, 0x13, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00 };
            unsigned num = 0;
            if (sscanf(argv[i] + 6, "%u", &num) == 1)
            {
                *((unsigned short *)(buf+11)) = num;
                z8.send(buf, sizeof(buf));
                fprintf(stderr, "setting Brightness to %d %d%%\n", num, num/100);
            }
        }
        if (arg.substr(0,  4) == "TEST")
        {
            unsigned char buf[] = { 0x12, 0x10, 0x00, 0x13, 0x00, 0x00, 0x00,     0x00, 0x00,     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00 };
            int num = 1;
            if (sscanf(argv[i] + 4, "%d", &num) == 1)
            {
                buf[17] = num;
                z8.send(buf, sizeof(buf));
                fprintf(stderr, "setting testpattern %d\n", num);
            }
        }
        if (arg.substr(0, 5) == "BLACK")
        {
            unsigned char buf[] = { 0x10, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00,     0x00, 0x00,      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
            z8.send(buf, sizeof(buf));
        }
        if (arg.substr(0,  6) == "WAKEUP")
        {
            unsigned char buf[] = { 0x10, 0x10, 0x00, 0x12, 0x00, 0x00, 0x00,     0x00, 0x00,      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };
            z8.send(buf, sizeof(buf));
        }
    }
 }
