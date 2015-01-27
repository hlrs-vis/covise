/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM HORNETserver
//
// HORNET handling server sending UDP packets for HORNET events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//

#include <covise/covise.h>
#include <unistd.h>
#include <signal.h>
#ifndef __hpux
#include <sys/prctl.h>
#endif
#include <sys/types.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <util/SerialCom.h>
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

static const char *HornetServerVersion = "1.0";

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
    fprintf(stderr, "  + VRC HORNETserver %-10s   (C) 2005 VISENSO GmbH   +\n", HornetServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   Serial Interface:  %-30s +\n", serialPort);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establish some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux__
#ifndef __hpux
    prctl(PR_TERMCHILD); // Exit when parent does
#endif
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// open serial port
    SerialCom serial(serialPort, 19200, 8, 'N', 1);
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
    cerr << "3333333333" << endl;
    //Initialize the device
    char buf[4];
    sprintf(buf, "d");
    serial.write(buf, 1);
    unsigned char bytes[6];
    bytes[5] = '\0';
    serial.read(bytes, 5);
    cerr << "SSSSSSSSSSSSSStatus: " << bytes << endl;
    sprintf(buf, "D00");
    serial.write(buf, 3);
    sprintf(buf, "O7F");
    serial.write(buf, 3);
    serial.read(bytes, 5);
    cerr << "Status: " << bytes << endl;

    while (true)
    {
        unsigned char bytes[6];
        bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
        bytes[4] = 0;
        bytes[5] = 0;
        sprintf(buf, "i");
        serial.write(buf, 1);
        serial.read(bytes, 5);
        showbuffer(bytes, sender, stationID);

        cerr.setf(ios::hex);
        cerr << (char)bytes[0] << " "
             << (char)bytes[1] << " "
             << (char)bytes[2] << " "
             << (char)bytes[3] << " "
             << (char)bytes[4] << endl;
        usleep(100000);
    }
    return 0;
}

// Send the data to the port and display it on the screen
void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID)
{
    char sendbuffer[2048];
    bytes[3] = '\0';
    sprintf(sendbuffer, "VRC %d %3ld [0.0 0.0 0.0] - [0 0 0 0 0 0 0 0 0] - [0 0]",
            stationID, 0x7fL & ~strtol((const char *)bytes + 1, NULL, 16));
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
         << "   " << progName << " 3            Read HORNET device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -d /dev/ttyS15 -t visenso:6666 4\n"
         << "                           Read HORNET device at /dev/ttyS15 and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
