/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM PS2server
//
// PS2 handling server sending UDP packets for PS2 events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//

#include <covise/covise.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include "PsAux.h"
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

using namespace covise;
//All commands to the box are sent as strings without LF/CR
//The response of the box is always a string with LF/CR
//The last character is always a '+' ( OK ) or a '-' ( Error )
//- Initialization:
//   I) "D00" reset to input
//  II) "O00" ( Oh-Zero-zero) setzt die Eingänge auf
//       TTL-Pegel ( die können auch Tristate)
//Reading values
//- Send an "i" to the box
//  You get 5 characters of response the first two
//  of which are response where the last three are "+" ( Or "-" on Error )
//  and LF/CR

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *Ps2ServerVersion = "1.0";

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID);
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

    static const char *DEF_PSAUX = "/dev/psaux";
    const char *cBytesCount = arg.getOpt("-b", "--bytes", "3");
    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *psauxPort = arg.getOpt("-d", "--device", DEF_PSAUX);

    if (arg.numArgs() < 1)
    {
        cerr << "\nCover station ID missing" << endl;
        exit(0);
    }

    //
    int stationID;

    int nitem = sscanf(arg[0], "%d", &stationID);
    cerr << "No. of assigned input items assigned = " << nitem << endl;

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + VRC PS2server %-10s   (C) 2003 VISENSO   GmbH +\n", Ps2ServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   PS2 Interface:  %-30s +\n", psauxPort);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establish some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux
    prctl(PR_TERMCHILD); // Exit when parent does
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// open psaux port
    PsAux psaux(psauxPort);
    if (psaux.isBad())
    {
        cerr << psauxPort << ": "
             << psaux.errorMessage() << endl;
        return -1;
    }
    /// Open UDP sender
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : " << psaux.errorMessage() << endl;
        return -1;
    }

    //    Was used to get the status from
    // the device, but i think it is unnecessary
    //    cerr.setf(ios::hex);
    //    char buf[10];
    //
    //    buf[0]=(char)0xF4;
    //    buf[1]=0;
    //    buf[1]=0;

    //buffer for the bytes read from the ps2 interface
    unsigned char bytes[6];

    bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
    bytes[4] = 0;
    bytes[5] = 0;

    //Some ps2 interfaces send 4 bytes others 3 bytes
    //it is not clear which one sends what
    //so the user has to find/try it out
    //He must specify it by a command line parameter
    // 3 is the default
    int bytesCount = atoi(cBytesCount);

    //Synchronizing
    //each tripel (or quadrupel) begins with a byte where
    // bit 3 is set
    do
    {
        bytes[0] = 0;
        psaux.read(bytes, 1);
    } while ((8 & bytes[0]) == 0);

    //The "starting" byte is found;
    // read the rest

    int i;
    for (i = 0; i < bytesCount - 1; i++)
    {
        psaux.read(bytes + i, 1);
        cerr << "bytes[" << i << "]=" << (int)(char)bytes[i] << endl;
    }
    //Now actual evaluation can begin

    while (true)
    {

        bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
        bytes[4] = 0;
        bytes[5] = 0;
        cerr.setf(ios::hex);
        //For debugging purposes
        cerr << "Bytes: ";

        for (i = 0; i < bytesCount; i++)
        {
            psaux.read(bytes + i, 1);
            if ((7 & bytes[0]) == 3)
            {
                bytes[i] = 4;
            }
            cerr << (int)bytes[i] << " ";
        }
        cerr << endl;
        showbuffer(bytes, sender, stationID);
        //      usleep(100000);
    }
    return 0;
}

// Send the data to the port and display it on the screen
void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID)
{
    //   int button1;
    char sendbuffer[2048];
    bytes[3] = '\0';
    sprintf(sendbuffer, "VRC %d %3d [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [0 0]",
            stationID, 7 & bytes[0]);

    fprintf(stderr, "%s\n", sendbuffer);
    sender.send(sendbuffer, strlen(sendbuffer) + 1);
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
         << "   -d <psauxPort>         Device name of psaux port\n"
         << "   --device=psauxPort     (default: SGI /dev/ttyd1, Linux /dev/ttyS2)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3            Read PS2 device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15 -t visenso:6666 4\n"
         << "                           Read PS2 device at /dev/ttyS15 and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
