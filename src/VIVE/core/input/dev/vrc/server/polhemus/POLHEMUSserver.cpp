/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// fobserver
// input device daemon for Ascension Flock of Birds
// reads data from serial port
// and writes data to a udp socket (host/port)
// (C) 2002-2003 VirCinity GmbH
// authors: we

#include <covise/covise.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#endif
#include <device/polhemusdrvr.h>

#include <util/UDP_Sender.h>

#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>

#include <util/ArgsParser.h>

using namespace covise;

static const char *PolhemusServerVersion = "1.0";
//static int MAX_POLHEMUS_RECV=4;

// global variable for signal handler
fastrak *tracker;

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    (void)signo;

    if (tracker)
    {
        delete tracker;
        tracker = NULL;
    }
    exit(0);
}

#ifdef __linux__
#define facos acos
#endif

void displayHelp(const char *progName)
{
    cout << progName << " [options]  dev0[=map0] dev1[=map1] ...\n"
         << "\n"
         << "   devX = Device Numbers of Receivers (1..14)\n"
         << "   mapX = COVER device numbers (0..31)\n"
         << "\n"
         << "   Device Dev0 must be Interaction Device !\n"
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
         << "   --device=serialPort     (default: SGI /dev/ttyd1, Linux /dev/ttyS2)\n"
         << "\n"
         << "   -b <baudrate>           Speed of serial interface\n"
         << "   --baudrate=<baudrate>   (default 19200 Baud)\n"
         << "\n"
         << "   -H <hemisphere>         Hemisphere selection: FRONT REAR LEFT RIGHT UPPER LOWER\n"
         << "   --Hemisphere=Hemisphere (default: FRONT)\n"
         << "\n"
         << "   -i <inputdevice>        Used input device: STYLUS or WAND \n"
         << "   --inputdevice=<device>  (default: STYLUS)\n"
         << "\n"
         << "   -n <numStations>        Number of Stations attached:\n"
         << "   --numStations=number    (default: number of given divices)\n"
         << "\n"
         << "   -h, --help              Show this help\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   POLHEMUSserver -d /dev/ttyS2 1 2\n"
         << "\n"
         << "   Start a Server to localhost:7777, reading with 19200 Baud on /dev/ttyS2\n"
         << "   from a Flock with 2 stations. Use FRONT Hemisphere, transmit stations 1 \n"
         << "   and 2 to COVER IDs 1 and 2.\n"
         << "\n"
         << "   POLHEMUSserver -d /dev/ttyS2 1=0 2=4\n"
         << "\n"
         << "   Same, but send station 1 results with ID=0 and station 2 with ID=4\n"
         << "\n"
         << "   POLHEMUSserver --target=devil:7234 --device=/dev/ttyS2 --baudrate=38400 \\\n"
         << "                  --hemisphere=LOWER -i STYLUS --numStations=4  4=0 3=1\n"
         << "\n"
         << "   Start a Server to host \"devil\" Port 7234, reading from a Polhemus with\n"
         << "   4 receivers using LOWER hemisphere and a WAND device. Send Station 4\n"
         << "   results with ID=0 and station 3 results with ID=1\n"
         << endl;
}

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

    signal(SIGINT, sigHandler);
#ifndef WIN32
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
#endif
    signal(SIGTERM, sigHandler);

    static const int MAX_POLHEMUS_RECV = 16;
    float x, y, z;

    int hardwareID[MAX_POLHEMUS_RECV]; //
    int vrcIds[MAX_POLHEMUS_RECV]; //
    int i, j;

// ----- create default values for all options
#ifdef __linux__
    static const char *DEF_SERIAL = "/dev/ttyS2";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    /// parse arguments
    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *mcastttl = arg.getOpt("-l", "--ttl", "-1");
    const char *mcastif = arg.getOpt("-m", "--mcastif", NULL);
    const char *rateStr = arg.getOpt("-r", "--rate", "20");
    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);
    const char *baudrateStr = arg.getOpt("-b", "--baudrate", "19200");
    const char *hem = arg.getOpt("-H", "--Hemisphere", "FRONT");
    const char *inputDevStr = arg.getOpt("-i", "--inputdevice", "STYLUS");
    const char *numStationsStr = arg.getOpt("-n", "--numStations", "-1");

    int baudrate = atoi(baudrateStr);
    int numStations = atoi(numStationsStr);
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
    rate = 1.0 / rate;
    struct timeval delay;
    delay.tv_sec = (int)rate;
    delay.tv_usec = (int)(1e6 * (rate - delay.tv_sec));

    // --------- hemisphere ---------------------------------------------
    float hx, hy, hz;

    if (strncasecmp(hem, "FRONT", 5) == 0)
    {
        hx = 0.0;
        hy = 1.0;
        hz = 0.0;
    }
    else if (strncasecmp(hem, "REAR", 4) == 0)
    {
        hx = 0.0;
        hy = -1.0;
        hz = 0.0;
    }
    else if (strncasecmp(hem, "UPPER", 5) == 0)
    {
        hx = 0.0;
        hy = 0.0;
        hz = 1.0;
    }
    else if (strncasecmp(hem, "LOWER", 5) == 0)
    {
        hx = 0.0;
        hy = 0.0;
        hz = -1.0;
    }
    else if (strncasecmp(hem, "LEFT", 4) == 0)
    {
        hx = -1.0;
        hy = 0.0;
        hz = 0.0;
    }
    else if (strncasecmp(hem, "RIGHT", 5) == 0)
    {
        hx = 1.0;
        hy = 0.0;
        hz = 0.0;
    }
    else
    {
        cerr << "Invalid hemisphere:" << hem << endl;
        ;
        exit(0);
    }

    // --------- get input device ---------------------------------------
    int inputDev;
    if (strncasecmp(inputDevStr, "stylus", 6) == 0)
        inputDev = fastrak::BUTTONDEVICE_STYLUS;
    else if (strncasecmp(inputDevStr, "wand", 4) == 0)
        inputDev = fastrak::BUTTONDEVICE_WAND;
    else
    {
        cerr << "Illegal Input device: " << inputDevStr << endl;
        exit(0);
    }

    // ------ count given IDs -----------------------------------------------
    int numVrcIds = arg.numArgs();

    // ------ number of birds: if not given, count --------------------------
    if (numStations <= 0)
        numStations = numVrcIds;

    if (numStations > MAX_POLHEMUS_RECV)
    {
        fprintf(stderr, "more than %d receivers are not supported\n", MAX_POLHEMUS_RECV);
        exit(-1);
    }

    // ------ Fill ID arrays --------------------------------------

    for (i = 0; i < MAX_POLHEMUS_RECV; i++)
    {
        hardwareID[i] = -1;
        vrcIds[i] = -1;
    }

    for (i = 0; i < numVrcIds; i++)
    {
        // We either have a single number, or number=number
        int birdId, vrcId;
        int numRead = sscanf(arg[i], "%d=%d", &birdId, &vrcId);
        if (birdId >= MAX_POLHEMUS_RECV)
        {
            cerr << "Maximum Bird ID = " << MAX_POLHEMUS_RECV << endl;
            exit(0);
        }
        if (numRead < 2)
            vrcId = birdId;

        for (j = 0; j < MAX_POLHEMUS_RECV; j++)
        {
            if (birdId == (j + 1))
            {
                hardwareID[j] = birdId;
                vrcIds[j] = vrcId;
            }
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC FOBserver %-10s    (C) 2005 VISENSO GmbH   +\n", PolhemusServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   UDP Target:        %-30s +\n", target);
    printf("  +   Multicast TTL:     %-30s +\n", mcastttl);
    printf("  +   Multicast Interface: %-28s +\n", mcastif ? mcastif : "(null)");
    printf("  +   Send Rate:         %-3.1f Packets/s                 +\n", 1.0 / rate);
    printf("  +   Serial Interface:  %-30s +\n", serialPort);
    printf("  +   Baudrate:          %-30d +\n", baudrate);
    printf("  +   Hemisphere:        %-8s ( %2.0f %2.0f %2.0f )          +\n", hem, hx, hy, hz);
    printf("  +   Device:            %-30s +\n", inputDevStr);
    printf("  +   # of Stations:     %-30d +\n", numStations);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Mapping:                                            +\n");

    for (i = 0; i < MAX_POLHEMUS_RECV; i++)
    {
        if (hardwareID[i] != -1)
            printf("  +   Receiver %-2d  --> COVER: ID %-2d                     +\n",
                   hardwareID[i], vrcIds[i]);
    }
    printf("  +-----------------------------------------------------+\n");

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Above was only parameter getting, now for the real work

    // create udp socket
    UDP_Sender sender(target, mcastif, atoi(mcastttl));
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : " << sender.errorMessage() << endl;
        return -1;
    }

    // Set up Tracker
    printf("  - Initialize Tracker class\n");
    tracker = new fastrak(serialPort, baudrate, numStations, inputDev);

    // Check Connection
    printf("  - Check Connection\n");
    if (!tracker->testConnection())
    {
        cerr << "flock tracker not connected" << endl;
        exit(0);
    }

    // Set Stations
    for (i = 0; i < numStations; i++)
    {
        printf("  - Setup Station %d\n", i + 1);
        tracker->setStation(i + 1);
    }

    // Setup 1st Device in Line as Hand device
    printf("  - Init Hand Device: %d\n", hardwareID[0]);
    tracker->setHemisphere(hardwareID[0], hx, hy, hz);
    tracker->setStation(hardwareID[0]);
    tracker->setStylusMouseMode();

    // set up all others
    for (i = 1; i < numVrcIds; i++)
    {
        printf("  - Init Recv Device: %d\n", hardwareID[i]);
        tracker->setHemisphere(hardwareID[i], hx, hy, hz);
        tracker->setStation(hardwareID[i]);
    }

    // fork server process
    printf("  - Fork slave process\n");
    tracker->start();
    int frame = 0;

    while (1)
    {
        struct timeval delayT = delay; // select does not guarantee holding the value
        select(0, NULL, NULL, NULL, &delayT);

        for (i = 0; i < MAX_POLHEMUS_RECV; i++)
        {
            if (vrcIds[i] != -1)
            {
                int station = hardwareID[i];

                /// Position / orientation Using Performer here - change it!
                float w, q1, q2, q3;
                float v1, v2, v3;
                float mat[4][4];
                tracker->getQuaternions(station, &w, &q1, &q2, &q3);
                float phi = 2 * acos(w);
                v1 = q1 / sin(phi / 2);
                v2 = q2 / sin(phi / 2);
                v3 = q3 / sin(phi / 2);
                float sina = sin(phi);
                float cosa = cos(phi);
                mat[0][0] = cosa + v1 * v1 * (1 - cosa);
                mat[0][1] = v1 * v2 * (1 - cosa) - v3 * sina;
                mat[0][2] = v1 * v3 * (1 - cosa) + v2 * sina;
                mat[1][0] = v2 * v2 * (1 - cosa) + v3 * sina;
                mat[1][1] = cosa + v2 * v2 * (1 - cosa);
                mat[1][2] = v2 * v3 * (1 - cosa) - v1 * sina;
                mat[2][0] = v3 * v1 * (1 - cosa) - v2 * sina;
                mat[2][1] = v3 * v2 * (1 - cosa) + v1 * sina;
                mat[2][2] = cosa + v3 * v3 * (1 - cosa);
                //////mat.makeRot(phi*180/M_PI, n[0], n[1], n[2]);
                tracker->getAbsPositions(station, &x, &y, &z);

                unsigned int but;
                tracker->getStylusSwitchStatus(station, &but);
                char sendbuffer[2048];
                sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [%6.3f %6.3f ]",
                        vrcIds[i], but, x, y, z,
                        mat[0][0], mat[0][1], mat[0][2],
                        mat[1][0], mat[1][1], mat[1][2],
                        mat[2][0], mat[2][1], mat[2][2],
                        0.0, 0.0);

                sender.send(sendbuffer, strlen(sendbuffer) + 1);

                if (frame % sendsPerVerbose == 0)
                {
                    fprintf(stderr, "%s\n", sendbuffer);
                }
            }
        }
        if (frame % sendsPerVerbose == 0)
        {
            fprintf(stderr, "---\n");
        }
        frame++;
    }
}
