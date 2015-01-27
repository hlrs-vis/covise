/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "ServoStar.h"
#include "CanOpenBus.h"
#include "PcanPci.h"

void printHelp();

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        printHelp();
        exit(1);
    }

    unsigned char opmode;

    if (strcmp(argv[1], "egearing") == 0)
        opmode = 0xf7;
    else if (strcmp(argv[1], "jogging") == 0)
        opmode = 0xf8;
    else if (strcmp(argv[1], "homing") == 0)
        opmode = 0xf9;
    else if (strcmp(argv[1], "trajectory") == 0)
        opmode = 0xfa;
    else if (strcmp(argv[1], "analogcurrent") == 0)
        opmode = 0xfb;
    else if (strcmp(argv[1], "analogspeed") == 0)
        opmode = 0xfc;
    else if (strcmp(argv[1], "digitalcurrent") == 0)
        opmode = 0xfd;
    else if (strcmp(argv[1], "digitalspeed") == 0)
        opmode = 0xfe;
    else if (strcmp(argv[1], "position") == 0)
        opmode = 0xff;
    else if (strcmp(argv[1], "positioningpp") == 0)
        opmode = 0x1;
    else if (strcmp(argv[1], "speedpv") == 0)
        opmode = 0x3;
    else if (strcmp(argv[1], "hominghm") == 0)
        opmode = 0x6;
    else
    {
        printHelp();
        exit(1);
    }

    PcanPci can(1, CAN_BAUD_1M);
    CanOpenBus bus(&can);
    ServoStar drive(&bus, 1);

    drive.setOpMode(opmode);
}

void printHelp()
{
    std::cout << "Tool for setting the operation mode of a ServoStar 600 via a can bus." << std::endl << std::endl;
    std::cout << "Please specify the operation mode as an argument in the command line:" << std::endl;
    std::cout << "\tegearing\n\tjogging\n\thoming\n\ttrajectory\n\tanalogcurrent\n\tanalogspeed\n\tdigitalcurrent\n\tdigitalspeed\n\tposition\n\tpositioningpp\n\tspeedpv\n\thominghm\n";
}
