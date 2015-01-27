/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM CEREALserver
//
// CEREAL handling server sending UDP packets for CEREAL events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version, based on MIKEserver 1.2

#include <covise/covise.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>
#include <signal.h>
#if !defined(WIN32) && !defined(__hpux)
#include <sys/prctl.h>
#include <errno.h>
#endif
#include <util/SerialCom.h>
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

#include <dev/legacy/bgLib.h>

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}
using namespace covise;

static const char *CerealServerVersion = "1.0";

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
    static const char *DEF_SERIAL = "/dev/ttycua2";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *mcastttl = arg.getOpt("-l", "--ttl", "-1");
    const char *mcastif = arg.getOpt("-m", "--mcastif", NULL);
    const char *rateStr = arg.getOpt("-r", "--rate", "20");
    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);

    float rate = atof(rateStr);

    // check rate
    if (rate < 0.1 || rate > 1000)
    {
        fprintf(stderr, "Illegal rate '%f' transmissions per second\n", rate);
        exit(1);
    }

    // verbose max. once per second
    int sendsPerVerbose;
    if (rate < 1)
        sendsPerVerbose = 1;
    else
        sendsPerVerbose = (int)rate;

    // select() delay record
    unsigned long delay_usec = (unsigned long)(1e6 / rate);

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
    fprintf(stderr, "  + VRC CEREALserver %-10s   (C) 2005 VISENSO GmbH   +\n", CerealServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   Serial Interface:  %-30s +\n", serialPort);
    fprintf(stderr, "  +   Multicast TTL:     %-30s +\n", mcastttl);
    fprintf(stderr, "  +   Multicast Interface: %-28s +\n", mcastif ? mcastif : "(null)");
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establich some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);
#if !defined(WIN32) && !defined(__hpux)
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux__
    prctl(PR_TERMCHILD); // Exit when parent does
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
#endif

    /// Open UDP sender
    UDP_Sender sender(target, mcastif, atoi(mcastttl));
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : " << strerror(errno) << endl;
        return -1;
    }

    bglv bgdata;
    memset(&bgdata, 0, sizeof(bgdata));

    // store data values in own field, might be overwritten otherwise
    int outData[3] = { 0, 0, 0 };

    // default values for CerealConfig.     D1  D2  D3
    static const char *defaultCereal[] = { "IN", "IN", "OUT 0x1" };
    static const int inFlags[] = { DIC1, DIC2, DIC3 };
    static const int outFlags[] = { DOC1, DOC2, DOC3 };

    // some buffers...
    char varname[32];
    char buffer[256];

    // loop: names IO1,IO2,IO3, but fields 0..2
    int i;
    bgdata.dig_in = 0;
    bgdata.dig_out = 0;
    for (i = 0; i < 3; i++)
    {
#ifndef STANDALONE
        sprintf(varname, "CerealConfig.IO%d", i + 1);
        const char *config = coCoviseConfig::getEntry(varname);
#else
        const char *config = NULL;
#endif
        if (!config)
            config = defaultCereal[i];

        // OUT flag: read value
        if (strstr(config, "OUT") || strstr(config, "out"))
        {
            bgdata.dig_out |= outFlags[i];
            if (sscanf(config, "%s %s", varname, buffer) != 2)
            {
                cerr << "failed to parse 'out' config" << endl;
            }
            // allow HEX input
            outData[i] = strtol(buffer, NULL, 0);
        }

        // IN flag: no values
        else
        {
            bgdata.dig_in |= inFlags[i];
        }
    }

    /// Read all analog inputs
    bgdata.analog_in = AIC1 | AIC2 | AIC3 | AIC4 | AIC5 | AIC6 | AIC7 | AIC8;

    /// Do not use analog output
    bgdata.analog_out = 0;
    bgdata.dout[0] = 0x0;
    bgdata.dout[1] = 0x0;
    bgdata.dout[2] = 0x0;

    /*
    *  Set the baud rate
    */
    bgdata.baud = BAUD192;

    int st = open_lv(&bgdata, (char *)serialPort, FB_NOBLOCK);
    if (st < 0)
    {
        fprintf(stderr, "error connecting to CerealBox on %s !!\n", serialPort);
        exit(0);
    }

    st = init_lv(&bgdata);
    if (st < 0)
    {
        fprintf(stderr, "error initializing CerealBox on %s !!\n", serialPort);
        exit(0);
    }

    unsigned long frame = 0;
    while (1)
    {
        for (i = 0; i < 3; i++)
        {
            bgdata.dout[i] = outData[i];
        }
        send_outputs(&bgdata); // necessary?
        // make sure that sampling is not done more often than 20 times/second
        usleep(delay_usec);

        check_inputs(&bgdata);
        int buttons = bgdata.din[0];

        char sendbuffer[2048];
        sprintf(sendbuffer, "VRC %d %3d [0.0 0.0 0.0] - [",
                stationID, buttons);
        for (i = 0; i < 8; i++)
        {
            sprintf(sendbuffer + strlen(sendbuffer), "%5.2f ", bgdata.ain[i]);
        }
        sprintf(sendbuffer + strlen(sendbuffer), "0] - [0 0]");
        sender.send(sendbuffer, strlen(sendbuffer) + 1);

        if (frame % sendsPerVerbose == 0)
        {
            fprintf(stderr, "%s\n", sendbuffer);
        }

        frame++;
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
         << "   -m <ifaddr>             set multicast interface address\n"
         << "   --mcastif=ifaddr        (default: INADDR_ANY)\n"
         << "\n"
         << "   -l <ttl>                set multicast packet time to live/maximum hop count\n"
         << "   --mcastttl=ttl          (default: 1, only on local subnet)\n"
         << "\n"
         << "   -r <value>              transmission speed in Packets/sec\n"
         << "   --rate=value            (default: 20)\n"
         << "\n"
         << "   -d <serialPort>         Device name of serial port\n"
         << "   --device=serialPort     (default: SGI /dev/ttyd1, Linux /dev/ttycua2)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3            Read CEREAL device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -d /dev/ttycua15 -t visenso:6666 4\n"
         << "                           Read CEREAL device at /dev/ttycua15 and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
