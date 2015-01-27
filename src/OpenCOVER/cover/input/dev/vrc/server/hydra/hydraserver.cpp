/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#else
#include <sys/socket.h>
#include <strings.h>

#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#endif
#include <signal.h>
#include <UDP_Sender.h>
#include <ArgsParser.h>

#include <stdio.h>

#include <sixense.h>
#include <sixense_math.hpp>

static const char *compatibilityString = ""; // put 0.00 in here for 5.2.3

#if !defined(__MINGW32__)
int strcasecmp(const char *s1, const char *s2)
{
    return stricmp(s1, s2);
}
#endif

static const char *HydraServerVersion = "1.0";

#define MAXSENSORS 4

// signal handler: correctly shut down art
void sigHandler(int signo)
{
    fprintf(stderr, "Shutting down HydraServer for signal %d in Process %d\n", signo, getpid());
    sixenseExit();
    exit(0);
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Split options of form   B1=0 F10=7 ...
//       into targetNo (B1=0, B2=1, ... F1=10, F2=11, ...) and mapping value

bool splitOpt(const char *option, int &targetNo, int &mapNo)
{
    // no string or long string: error
    if (!option || strlen(option) > 6)
        return false;

    // we have 1 char + number=number
    int num1, num2;
    int numRead = sscanf(option, "%d=%d", &num1, &num2);
    switch (numRead)
    {
    case 0:
        return false; // no numbers
    case 2:
        mapNo = num2; // NO BREAK HERE
    case 1:
        targetNo = num1;
        break;
    default:
        return false; // too many numbers - we should never get here!
    }

    // FlyStick or Body?  In either case, correct numbers
    //  if ( *option=='b' || *option=='B' )            // Body:     ID = displayNo - 1
    targetNo += -1;
    // else if ( *option=='f' || *option=='F' )       // FlyStick: ID = displayNo - 1 + 10
    //     targetNo += -1 + 10;
    //  else return false;

    return true;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void displayHelp(const char *progName)
{
    //   fprintf(stderr,"Usage: HydraServer <targetHost> <targetPort> <sendRate> <artPort> <numSens> <bodyId> <vrcId>  <bodyId> <vrcId>\n");
    //   //                     [0]       [1]       [2]     [3]        [4]        [5]       [6]      [7]      [8]      [9]
    //   fprintf(stderr,"Example: HydraServer sgi001 5555 20  5000 2 1 0 2 1\n");
    //   fprintf(stderr, "        body ids start with 0, flysticks ids start with 10\n");

    std::cout << progName << " [options]  dev0[=map0] dev1[=map1] ...\n"
              << "\n"
              << "   devX = Device Numbers of Bodies    (B1, B2, ...)\n"
              << "                         or FlySticks (F1, F2, ...)\n"
              << "   mapX = COVER device numbers (0..31)\n"
              << "\n"
              << "Options:\n"
              << "\n"
              << "   -t <host:port>           set target to send tracking UDP packets\n"
              << "   --target=host:port       (default: localhost:7777)\n"
              << "\n"
              << "   -r <value>               transmission speed in Packets/sec\n"
              << "   --rate=value             (default: 20)\n"
              << "\n"
              << "   -p <portNo>              Port number to receive A.R.T Tracker\n"
              << "   --port=portNo            (default 5000)\n"
              << "\n"
              << "   -s <host:portNo>         Send A.R.T Tracker Start/Stop command to\n"
              << "   --sendStart=host:portNo  (default none, do not use \"receive port\"\n"
              << "\n"
              << "\n"
              << "   -c <value>               Send to second target\n"
              << "   --cyberClassroom=value   (localhost:8888)\n"
              << "\n"
              << "Examples:\n"
              << "\n"
              << "   HydraServer F1 B1          Receive ART information on port 5000, and send \n"
              << "                            it with 20 Hz frequency to localhost Port 7777.\n"
              << "                            send Flystick F1  -->  COVER ID 0\n"
              << "                            send Body 1       -->  COVER ID 1\n"
              << "\n"
              << "   HydraServer F1=2 B1=3      as before, but \n"
              << "                            send Flystick F1  -->  COVER ID 1\n"
              << "                            send Body 1       -->  COVER ID 3\n"
              << "\n"
              << "   HydraServer --sendStart=10.0.0.1:5001 F1=2 B1=3\n"
              << "                            as before, but send signal Tracker PC at address\n"
              << "                            10.0.0.1, Port 5001 to start/stop measurement \n"
              << "                            before/after HydraServer execution\n"
              << std::endl;
}

int main(int argc, char **argv)
{
    ArgsParser arg(argc, argv);

    // get help
    if (argc < 2
        || 0 == strcasecmp(argv[1], "-h")
        || 0 == strcasecmp(argv[1], "--help"))
    {
        displayHelp(argv[0]);
        exit(-1);
    }

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *rateStr = arg.getOpt("-r", "--rate", "20");
    const char *sendRate = arg.getOpt("-s", "--sendrate", "16");

    // get port on which the dtrack sends the data
    float rate = atof(rateStr);

    // verbose max. once per second
    int sendsPerVerbose;
    if (rate < 1)
        sendsPerVerbose = 1;
    else
        sendsPerVerbose = (int)rate;

    // select() delay record
    rate = 1.0 / rate;
    struct timeval delay;
    delay.tv_sec = (int)rate;
    delay.tv_usec = (int)(1e6 * (rate - delay.tv_sec));

    // +++++++++++++++ prepare mapping +++++++++++++++

    int artID[MAXSENSORS], coverID[MAXSENSORS]; // COVER IDs for sensor[i]

    if (arg.numArgs() > MAXSENSORS)
    {
        std::cerr << "Only " << MAXSENSORS << " sensors allowed" << std::endl;
    }

    int i;
    int numSensors = 0;
    for (i = 0; i < arg.numArgs(); i++)
    {
        coverID[numSensors] = i;
        bool ok = splitOpt(arg[i], artID[numSensors], coverID[numSensors]);
        if (ok)
        {
            numSensors++;
        }
        else
        {
            std::cerr << "Illegal device selection: " << arg[i] << std::endl;
            exit(0);
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC HydraServer %-10s    (C) 2005 HLRS  +\n", HydraServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   UDP Target:        %-30s +\n", target);
    printf("  +   Send Rate:         %-3.1f Packets/s                 +\n", 1.0 / rate);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Mapping:                                            +\n");

    for (i = 0; i < numSensors; i++)
    {
        printf("  +   Target %c%-2d  --> COVER ID %-2d                       +\n",
               ((artID[i] >= 10) ? 'F' : 'B'),
               (artID[i] % 10) + 1,
               coverID[i]);
    }
    printf("  +-----------------------------------------------------+\n\n");

    /// ++++++++++++++++++++++++++ All parameters set - start work +++++++++++++++++++

    signal(SIGINT, sigHandler);
#ifndef WIN32
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
#endif
    signal(SIGTERM, sigHandler);

    // create udp socket
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        std::cerr << "Could not start UDP server to "
                  << target << " : "
                  << sender.errorMessage() << std::endl;
        return -1;
    }

    int init = sixenseInit();

    int activebase = sixenseSetActiveBase(0);

    int basecolor = sixenseSetBaseColor(255, 0, 0);

    sixenseAllControllerData acd;

    int reshigh = sixenseSetHighPriorityBindingEnabled(1);
    int autoenable = sixenseAutoEnableHemisphereTracking(1);
    int frame = 0;
    while (1)
    {

        sixenseGetAllNewestData(&acd);
        Sleep((int)*sendRate);

        /*fprintf(stderr,"left pos: x = %f\ty = %f\tz = %f\nright pos: x = %f\ty = %f\t z = %f\n\n",
			acd.controllers[0].pos[0],acd.controllers[0].pos[1],acd.controllers[0].pos[2],
			acd.controllers[1].pos[0],acd.controllers[1].pos[1],acd.controllers[1].pos[2]);	*/

        for (i = 0; i < numSensors; i++)
        {

            char sendbuffer[2048];
            sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ %d %d %s]",
                    coverID[i], acd.controllers[i].buttons, acd.controllers[i].pos[0], acd.controllers[i].pos[1], acd.controllers[i].pos[2],
                    acd.controllers[i].rot_mat[0][0], acd.controllers[i].rot_mat[0][1], acd.controllers[i].rot_mat[0][2],
                    acd.controllers[i].rot_mat[1][0], acd.controllers[i].rot_mat[1][1], acd.controllers[i].rot_mat[1][2],
                    acd.controllers[i].rot_mat[2][0], acd.controllers[i].rot_mat[2][1], acd.controllers[i].rot_mat[2][2],
                    acd.controllers[i].joystick_x, acd.controllers[i].joystick_y, compatibilityString);

            sender.send(sendbuffer);

            if (frame % sendsPerVerbose == 0)
            {
                fprintf(stderr, "%s\n", sendbuffer);
            }
        }
        if (frame % sendsPerVerbose == 0)
        {
            fprintf(stderr, "---\n");
        }
        frame++;
    }

    sixenseExit();
    return 0;
}