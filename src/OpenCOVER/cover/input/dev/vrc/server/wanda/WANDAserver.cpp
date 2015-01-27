/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM WANDAserver
//
// WANDA handling server sending UDP packets for WANDA events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//

#include <covise/covise.h>
#ifndef WIN32
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#if !defined(__hpux)
#include <sys/prctl.h>
#endif

#endif

#include <util/SerialCom.h>
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

using namespace covise;

static int button1, button2, button3;
static int xVal, yVal;

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *WandaServerVersion = "1.0";

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
void showbuffer(unsigned char *bytes, int n, UDP_Sender &sender, int stationID);
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
    static const char *DEF_SERIAL = "/dev/ttyS1";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);

    if (arg.numArgs() < 1)
    {
        cerr << "\nCover station ID missing" << endl;
        exit(0);
    }

    //
    int stationID;
    if (sscanf(arg[0], "%d", &stationID) != 1)
    {
        cerr << "\nFailed to parse Cover station ID" << endl;
        exit(0);
    }

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + VRC WANDAserver %-10s   (C) 2005 VISENSO GmbH +\n", WandaServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   Serial Interface:  %-30s +\n", serialPort);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establish some signal handlers
    signal(SIGINT, sigHandler);
#ifndef WIN32
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGHUP, sigHandler);
#endif
    signal(SIGTERM, sigHandler);
#ifndef __linux__
#ifndef __hpux
#ifndef WIN32
    prctl(PR_TERMCHILD); // Exit when parent does
#endif
#endif
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// open serial port
    SerialCom serial(serialPort, 1200, 7, 'N', 2);
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

    int numRead;
    while (true)
    {
        unsigned char bytes[4];
        bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
        //We wait for the first byte
        //it must have bit 6 set
        numRead = 0;
        int once = true;
        while (!(bytes[0] & 0x40))
        {
            struct timeval timeout = { 0, 100000 };
            numRead = serial.read(bytes, 1, timeout);
            if (!(bytes[0] & 0x40) && once)
            {
                once = false;
                xVal = 0;
                yVal = 0;
                char sendbuffer[2048];
                sprintf(sendbuffer, "VRC %d %3d [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [%d %d]",
                        stationID, button1 | button2 | button3, xVal, yVal);
                fprintf(stderr, "%s\n", sendbuffer);
                sender.send(sendbuffer, strlen(sendbuffer) + 1);
            }
        }
        serial.read(bytes + 1, 1);
        serial.read(bytes + 2, 1);
        //The problem is
        //if the left or right button
        //is pressed, we get three bytes
        //if the middle button is pressed
        //we get four bytes
        //conforming
        //to the 1200 Baud we wait 1 millisecond
        //for the fourth byte to come
        //after that we decide that there are only 3
        //bytes and the middle button was NOT pressed.
        struct timeval timeout = { 0, 1000 };
        numRead = serial.read(bytes + 3, 1, timeout);
        if (numRead == 1)
        {
            showbuffer(bytes, 4, sender, stationID);
        }
        else
        {
            showbuffer(bytes, 3, sender, stationID);
        }
    }
    return 0;
}

// Send the data to the port and display it on the screen
void showbuffer(unsigned char *bytes, int n, UDP_Sender &sender, int stationID)
{
    button1 = (((bytes[0]) & 0x20) >> 3);
    button3 = (((bytes[0]) & 0x10) >> 4);
    button2 = 0;
    if (n == 4) //middle button pressed
    {
        button2 = (((bytes[3]) & 0x20) >> 4);
    }
    xVal = ((3 & bytes[0]) << 6) | (bytes[1] & 0x3F);
    yVal = ((0xC & bytes[0]) << 4) | (bytes[2] & 0x3F);
    if (xVal > 128)
        xVal = -(256 - xVal);
    if (yVal > 128)
        yVal = -(256 - yVal);

    char sendbuffer[2048];
    sprintf(sendbuffer, "VRC %d %3d [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [%d %d]",
            stationID, button1 | button2 | button3, xVal, yVal);
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
         << "   -d <serialPort>         Device name of serial port\n"
         << "   --device=serialPort     (default: SGI /dev/ttyd1, Linux /dev/ttyS2)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3            Read WANDA device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15 -t visenso:6666 4\n"
         << "                           Read WANDA device at /dev/ttyS15 and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
