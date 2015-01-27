/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM Vesa
//
// Program for switching Liesegang Projectors to Vesa mode
//
// Initial version: 2003-05-28 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//

#include <covise/covise.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>

#include "SerialCom.h"
#include "ArgsParser.h"

void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *VesaVersion = "1.0";

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
//void showbuffer(unsigned char *bytes, int n,    UDP_Sender &sender, int projectorID);
///////// program starts here

int main(int argc, char *argv[])
{
    ArgsParser arg(argc, argv);

    //at least one projectors has to be connected
    if (argc < 2
        || 0 == strcasecmp(argv[1], "-h")
        || 0 == strcasecmp(argv[1], "--help"))
    {
        displayHelp(argv[0]);
        exit(-1);
    }

// ----- create default values for all options
#ifdef __linux__
    static const char *DEF_SERIAL = "/dev/ttyS1";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + %25s  (C) 2005 VISENSO GmbH   +\n", VesaVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  + \n");
    fprintf(stderr, "  +   Serial Interface:  %-30s +\n", serialPort);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establich some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux__
    prctl(PR_TERMCHILD); // Exit when parent does
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// open serial port
    SerialCom serial(serialPort, 2400, 8, 'N', 2);
    if (serial.isBad())
    {
        cerr << serialPort << ": "
             << serial.errorMessage() << endl;
        return -1;
    }

    char buf[8];
    sprintf(buf, ".DM41!");
    serial.write((void *)buf, 6);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Help

// show help
void displayHelp(const char *progName)
{
    cout << progName << " [options]  ProjectorNumber\n"
         << "\n"
         << "   "
         << "\n"
         << "Options:\n"
         << "\n"
         << "\n"
         << "   -d <serialPort>         Device name of serial port\n"
         << "   --device=serialPort     (default: SGI /dev/ttyd1, Linux /dev/ttyS2)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << "  use default serial and switch to Vesa"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15                    \n"
         << "                           Use Projector device at /dev/ttyS15 and switch to Vesa"
         << endl;
}
