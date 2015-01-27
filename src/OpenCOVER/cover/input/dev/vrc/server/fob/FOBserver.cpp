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
#include <stdlib.h>
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include "fob.h"

#include <util/UDP_Sender.h>

#ifndef WIN32
#include <sys/ioctl.h>
#include <sys/time.h>
#endif
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>

#include <util/ArgsParser.h>

using namespace covise;

static const char *FobServerVersion = "2.0";

// global variable for signal handler
fob *flock = NULL;

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    if (flock)
    {
        fprintf(stderr, "Shutting down flock for signal %d\n", signo);
#ifdef VISENSO
#ifndef WIN32
        delete flock;
        exit(0);
#endif
#endif
        flock->setStopping();
    }
}

void displayHelp(const char *progName)
{
    cout << progName << " [options]  dev0[=map0] dev1[=map1] ...\n"
         << "\n"
         << "   devX = Device Numbers of Receivers (1..14)\n"
         << "   mapX = COVER device numbers (0..31)\n"
         << "\n"
         << "Options:\n"
         << "\n"
         << "   -t <host:port>          set target to send tracking UDP packets\n"
         << "   --target=host:port      (default: localhost:7777)\n"
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
         << "   -m <selection>          Flock mode selection: STREAM or POINT\n"
         << "   --mode=<selection>      (default: STREAM)\n"
         << "\n"
         << "   -n <numBirds>           Number of Flocks attached: ERC counts as 1 bird!\n"
         << "   --numbirds=number       (default: number of given stations)\n"
         << "\n"
         << "   -p <protocol>           Protocol number: either 5.2 or 5.3\n"
         << "   --protocol=proto        (default: 5.3)\n"
         << "\n"
         << "   -mr <rate>           Measurement Rate\n"
         << "   --measurementrate=rate        (default: 103.3)\n"
         << "\n"
         << "   -h, --help              Show this help\n"
         << "\n"
         << "   -c <value>               Send to second target\n"
         << "   --cyberClassroom=value   (localhost:8888)\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   FOBserver -d /dev/ttyS2 1 2\n"
         << "\n"
         << "   Start a Server to localhost:7777, reading with 19200 Baud on /dev/ttyS2\n"
         << "   from a Flock with 2 stations. Use FRONT Hemisphere and STREAM mode,\n"
         << "   transmit stations 1 and 2 to COVER.\n"
         << "\n"
         << "   FOBserver -d /dev/ttyS2 1=0 2=4\n"
         << "\n"
         << "   Same as before, but send station 1 results with ID=0 and station 2 with ID=4\n"
         << "\n"
         << "   FOBserver --target=devil:7234 --device=/dev/ttyS2 --baudrate=38400 \\\n"
         << "             --hemisphere=LOWER --mode=POINT --numbirds=4  4=0 3=1\n"
         << "\n"
         << "   Start a Server to host \"devil\" Port 7234, reading from a Flock with Extended\n"
         << "   Range Transmitter and 3 receivers using LOWER hemisphere and POINT mode. Send\n"
         << "   Station 4 results with ID=0 and station 3 results with ID=1\n"
         << endl;
}

int main(int argc, char *argv[])
{
    ArgsParser arg(argc, argv);

    //at least one stations has to be connected
    if (argc < 2
        || 0 == strcmp(argv[1], "-h")
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

    static const int MAXBIRDS = 16;
    float x, y, z;

    int mode;
    int birdIds[MAXBIRDS]; //
    int vrcIds[MAXBIRDS]; //
    int i, j;
    float mat[3][3];

// ----- create default values for all options
#ifdef LINUX
    static const char *DEF_SERIAL = "/dev/ttyS2";
#else
    static const char *DEF_SERIAL = "/dev/ttyd1";
#endif

    const char *target = arg.getOpt("-t", "--target", "localhost:7777");
    const char *rateStr = arg.getOpt("-r", "--rate", "20");
    const char *serialPort = arg.getOpt("-d", "--device", DEF_SERIAL);
    const char *baudrateStr = arg.getOpt("-b", "--baudrate", "19200");
    const char *hem = arg.getOpt("-H", "--Hemisphere", "FRONT");
    const char *fobmode = arg.getOpt("-m", "--mode", "STREAM");
    const char *numBirdsStr = arg.getOpt("-n", "--numBirds", "-1");
    const char *protoStr = arg.getOpt("-p", "--protocol", "5.3");
    const char *mRateStr = arg.getOpt("-mr", "--measurementrate", "103.3");
    std::string cyberC = arg.getOpt("-c", "--cyberClassroom", "0");

    int baudrate = atoi(baudrateStr ? baudrateStr : "19200");
    float rate = atoi(rateStr ? rateStr : "20");
    int numBirds = atoi(numBirdsStr ? numBirdsStr : "-1");

    // Protocol: arr 3rd 0.0 or not
    const char *compatStr;
    if (strstr(protoStr, "5.2"))
        compatStr = "0.0 ";
    else
        compatStr = "";

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
    fob::hemisphere hemisphere = fob::FRONT_HEMISPHERE;

    hemisphere = fob::FRONT_HEMISPHERE;
    if (strncasecmp(hem, "FRONT", 5) == 0)
    {
        hemisphere = fob::FRONT_HEMISPHERE;
    }
    else if (strncasecmp(hem, "REAR", 4) == 0)
    {
        hemisphere = fob::REAR_HEMISPHERE;
    }
    else if (strncasecmp(hem, "UPPER", 5) == 0)
    {
        hemisphere = fob::UPPER_HEMISPHERE;
    }
    else if (strncasecmp(hem, "LOWER", 5) == 0)
    {
        hemisphere = fob::LOWER_HEMISPHERE;
    }
    else if (strncasecmp(hem, "LEFT", 4) == 0)
    {
        hemisphere = fob::LEFT_HEMISPHERE;
    }
    else if (strncasecmp(hem, "RIGHT", 5) == 0)
    {
        hemisphere = fob::RIGHT_HEMISPHERE;
    }
    else
    {
        fprintf(stderr, "ERROR in %s: %s is an invalid hemisphere\n", argv[0], hem);
        fprintf(stderr, "Setting hemisphere to FRONT\n");
    }

    // ------ point or stream mode ------------------------------------------
    mode = (0 == strcasecmp(fobmode, "STREAM"));
    // ------ count given IDs -----------------------------------------------
    int numVrcIds = arg.numArgs();

    // ------ number of birds: if not given, count --------------------------
    if (numBirds <= 0)
        numBirds = numVrcIds;

    if (numBirds > MAXBIRDS)
    {
        fprintf(stderr, "more than 16 birds are not supported\n");
        exit(-1);
    }

    // ------ Fill ID arrays --------------------------------------

    for (i = 0; i < MAXBIRDS; i++)
    {
        birdIds[i] = -1;
        vrcIds[i] = -1;
    }

    for (i = 0; i < numVrcIds; i++)
    {
        // We either have a single number, or number=number
        int birdId, vrcId;
        int numRead = sscanf(arg[i], "%d=%d", &birdId, &vrcId);
        if (birdId >= MAXBIRDS)
        {
            cerr << "Maximum Bird ID = " << MAXBIRDS << endl;
            exit(0);
        }
        if (numRead < 2)
            vrcId = birdId;

        for (j = 0; j < MAXBIRDS; j++)
        {
            if (birdId == (j + 1))
            {
                birdIds[j] = birdId;
                vrcIds[j] = vrcId;
            }
        }
    }

    float mRate = -1.;
    if (strcmp(mRateStr, "103.3"))
    {
        int nitem = sscanf(mRateStr, "%f", &mRate);
        cerr << "No of input itemd = " << nitem << endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC FOBserver %-10s    (C) 2005 VISENSO GmbH   +\n", FobServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   UDP Target:        %-30s +\n", target);
    printf("  +   Send Rate:         %-3.1f Packets/s                 +\n", 1.0 / rate);
    printf("  +   Serial Interface:  %-30s +\n", serialPort);
    printf("  +   Baudrate:          %-8d                       +\n", baudrate);
    printf("  +   FOB Hemisphere:    %-30s +\n", hem);
    printf("  +       Mode:          %-30s +\n", fobmode);
    printf("  +       Birds:         %-2d (including ERT if exist.)   +\n", numBirds);
    printf("  +   VRC Protocol:      %-30s +\n", ((compatStr[0]) ? "5.2 or earlier" : "5.3 or later"));
    printf("  +-----------------------------------------------------+\n");
    printf("  + Mapping:                                            +\n");

    for (i = 0; i < MAXBIRDS; i++)
    {
        if (birdIds[i] != -1)
            printf("  +   Bird %-2d  --> COVER: ID %-2d                         +\n",
                   birdIds[i], vrcIds[i]);
    }
    printf("  +-----------------------------------------------------+\n");

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Above was only parameter getting, now for the real work

    // create udp socket
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : " << sender.errorMessage() << endl;
        return -1;
    }

    // create second udp socket
    UDP_Sender sender2("localhost:8888");
    if (cyberC != "0" && sender2.isBad())
    {
        cerr << "Could not start UDP server to " << target << " : " << sender2.errorMessage() << endl;
        return -1;
    }

    flock = new fob(serialPort, baudrate, numBirds, mode);

    if (!flock->testConnection())
    {
        fprintf(stderr, "flock tracker not connected\n");
    }

    flock->autoconfig();
    if (mRate != -1.)
    {
        flock->setMeasurementRate(mRate);
    }
    flock->printSystemStatus();
    fprintf(stderr, "found %d ERCs\n", flock->getNumERCs());
    fprintf(stderr, "found %d receivers\n", flock->getNumReceivers());

    // send to all bird
    for (i = flock->getNumERCs() + 1; i < numBirds + 1; i++)
    {
        flock->setDataFormat(i, fob::FLOCK_POSITIONMATRIX);
        flock->setHemisphere(i, hemisphere);
        //flock->reportRate(i);
        if (flock->getNumERCs() == 0)
        {
            flock->changeRange(i, 1); //fullrange
        }
    }

    if (mode)
        flock->enableStreamMode();
    else
        //POINT mode - initially sends the group mode command only
        flock->sendGroupMode();

#ifdef VISENSO
#ifndef _WIN32
    fprintf(stderr, "Starting Server Process ...\n");
    flock->startServerProcess();
#endif
#endif

    int frame = 1;
    while (true)
    {
#ifndef VISENSO
        flock->processSerialStream();
#else
#ifdef _WIN32

        flock->processSerialStream();
#endif
#endif
        struct timeval delayT = delay; // select does not guarantee holding the value
        select(0, NULL, NULL, NULL, &delayT);

        for (i = 0; i < MAXBIRDS; i++)
        {
            if (vrcIds[i] != -1)
            {
                int birdId = birdIds[i];

                flock->getPositionMatrix(birdId, &x, &y, &z, &(mat[0][0]), &(mat[0][1]), &(mat[0][2]), &(mat[1][0]), &(mat[1][1]), &(mat[1][2]), &(mat[2][0]), &(mat[2][1]), &(mat[2][2]));
                x /= 10.0;
                y /= 10.0;
                z /= 10.0;

                if (x == 0.0 && y == 0.0 && z == 0.0)
                {
                    x = 0.00001;
                    y = 0.00001;
                    z = 0.00001;
                }

                unsigned short int buttons;
                flock->getButtons(birdId, &buttons);

                if (buttons == 112)
                    buttons = 4;
                if (buttons == 48)
                    buttons = 2;
                if (buttons == 16)
                    buttons = 1;

                char sendbuffer[2048];
                sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ 0.0 0.0 %s]",
                        vrcIds[i], buttons, x, y, z,
                        mat[0][0], mat[0][1], mat[0][2],
                        mat[1][0], mat[1][1], mat[1][2],
                        mat[2][0], mat[2][1], mat[2][2],
                        compatStr);

                sender.send(sendbuffer, strlen(sendbuffer) + 1);
                if (cyberC != "0")
                    sender2.send(sendbuffer, strlen(sendbuffer) + 1);

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
