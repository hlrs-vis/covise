/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM MIKEserver
//
// MIKE handling server sending UDP packets for MIKE events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//           1.1 Enhanced message window
//           1.2 Signal Handler, terminated when window closes

#include <covise/covise.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>
#include <signal.h>
#if !defined(__hpux) && !defined(_WIN32)
#include <sys/prctl.h>
#endif
#include <util/SerialCom.h>
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

using namespace covise;

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *MikeServerVersion = "1.2";

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);

///////// program starts here

int main(int argc, char *argv[])
{
    ArgsParser arg(argc, argv);

    //at least one stations has to be connected
    if (argc < 2
        || 0 == strcasecmp(argv[1], "-h")
        || 0 == strcasecmp(argv[1], "--help"))
    {
        displayHelp(argv[0]);
        exit(-1);
    }

// ----- create default values for all options
#ifdef __linux__
    static const char *DEF_SERIAL = "/dev/ttyS2";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);
    const char *maskStr = arg.getOpt("-m", "--mask", NULL);
    const char *cyberC = arg.getOpt("-c", "--cyberClassroom", "0");

    // AND the incoming data with this value
    unsigned char chrMask;

    if (maskStr)
    {
        int maskInt = atoi(maskStr);
        chrMask = (unsigned char)((~maskInt) & 0xff);
    }
    else
        chrMask = 0x7f;

    if (arg.numArgs() < 1)
    {
        cerr << "\nCover station ID missing" << endl;
        exit(0);
    }

    //
    int stationID;
    if (sscanf(arg[0], "%d", &stationID) != 1)
    {
        fprintf(stderr, "MIKEserver: sscanf for stationID failed\n");
    }

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + VRC MIKEserver %-10s   (C) 2005 VISENSO GmbH   +\n", MikeServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   Serial Interface:  %-30s +\n", serialPort);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +   Data Mask:         %c%c%c%c%c%c%c%c                       +\n",
            ((chrMask & 128) ? '1' : '0'), ((chrMask & 64) ? '1' : '0'), ((chrMask & 32) ? '1' : '0'), ((chrMask & 16) ? '1' : '0'),
            ((chrMask & 8) ? '1' : '0'), ((chrMask & 4) ? '1' : '0'), ((chrMask & 2) ? '1' : '0'), ((chrMask & 1) ? '1' : '0'));
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

/// establich some signal handlers
#ifndef WIN32
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGHUP, sigHandler);
#endif
    signal(SIGTERM, sigHandler);
    signal(SIGINT, sigHandler);
#ifndef __linux__
#if !defined(__hpux) && !defined(_WIN32)
    prctl(PR_TERMCHILD); // Exit when parent does
#endif
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// open serial port
    SerialCom serial(serialPort, 9600); // "8N1..." should have more parameters
    if (serial.isBad())
    {
        cerr << serialPort << ": "
             << serial.errorMessage() << endl;
        return -1;
    }

    /// Open UDP sender
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : " << serial.errorMessage() << endl;
        return -1;
    }
    /// Open second UDP sender
    UDP_Sender sender2("localhost:8888");
    if (strcmp(cyberC, "0") && sender2.isBad())
    {
        cerr << "Could not start UDP server to second target localhost:8888 : " << sender2.errorMessage() << endl;
        return -1;
    }

    int buttons = 0;
    while (1)
    {
        unsigned char byte = '\0';
        int numBytes = serial.read(&byte, 1);
        if (numBytes)
            buttons = byte & chrMask;

        char sendbuffer[2048];
        sprintf(sendbuffer, "VRC %d %3d [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [0 0]",
                stationID, buttons);
        fprintf(stderr, "%s\n", sendbuffer);
        sender.send(sendbuffer, strlen(sendbuffer) + 1);
        if (strcmp(cyberC, "0"))
            sender2.send(sendbuffer, strlen(sendbuffer) + 1);
    }

    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Help

// show help
void displayHelp(const char *progName)
{
    cout << progName << " [options]  coverID\n"
         << "\n"
         << "   coverID = Station ID for COVER's BUTTON_ADDR config\n"
         << "\n"
         << "Options:\n"
         << "\n"
         << "   -t <host:port>          set target to send tracking UDP packets\n"
         << "   --target=host:port      (default: localhost:7777)\n"
         << "\n"
         << "   -d <serialPort>         Device name of serial port\n"
         << "   --device=serialPort     (default: SGI /dev/ttyd1, Linux /dev/ttyS2)\n"
         << "\n"
         << "   -m <mask>               Bitmask to remove from input\n"
         << "   --mask=maskbits         (default: 128, for some MIKE protocol versions 32)\n"
         << "\n"
         << "   -c <value>               Send to second target\n"
         << "   --cyberClassroom=value   (localhost:8888)\n"
         << "\n"
         << "If experiencing trouble with MIKE devices, probe output with --mask=0\n"
         << "and then mask out the 2nd number of the console output, so when the\n"
         << "console output is \"VRC 0 32 [ ...\" use --mask=32\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3            Read MIKE device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15 -t visenso:6666 4\n"
         << "                           Read MIKE device at /dev/ttyS15 and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15 -m 160 4\n"
         << "                           Read MIKE device at /dev/ttyS15 and send data,\n"
         << "                           remove bits with values 128 and 32\n"
         << endl;
}
