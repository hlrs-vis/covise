/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MotionStar Server
// input device daemon for MotionStar
// reads data from MotionStar udp port
// and writes data to a udp socket (host/port)
// (C) 2003 VirCinity GmbH
// authors: we

#include <covise/covise.h>

#include <sys/socket.h>
#include <strings.h>

#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>

#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>
#include <device/VRTracker.h>
#include <device/birdTracker.h>

using namespace covise;

#ifdef MAXSENSORS
#undef MAXSENSORS
#endif

static const int MAXSENSORS = 128;
static const char *MotionStarServerVersion = "1.0";

// put " 0.0" in here for older VRC compatibility
static const char *compatibilityString = "";

static unsigned long getIP(const char *hostname)
{
    // try dot notation
    unsigned int binAddr;
    binAddr = inet_addr(hostname);
    if (binAddr != INADDR_NONE)
        return binAddr;

    // try nameserver
    struct hostent *hostRecord = gethostbyname(hostname);
    if (hostRecord == NULL)
        return INADDR_NONE;

    // analyse 1st result
    if (hostRecord->h_addrtype == AF_INET)
    {
        char *cPtr = (char *)&binAddr;
        cPtr[0] = *hostRecord->h_addr_list[0];
        cPtr[1] = *(hostRecord->h_addr_list[0] + 1);
        cPtr[2] = *(hostRecord->h_addr_list[0] + 2);
        cPtr[3] = *(hostRecord->h_addr_list[0] + 3);
        return binAddr;
    }
    else
        return INADDR_NONE;
}

// signal handler: correctly shut down Server
void sigHandler(int signo)
{
    (void)signo;

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

    return true;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void displayHelp(const char *progName)
{
    //   fprintf(stderr,"Usage: artserver <targetHost> <targetPort> <sendRate> <artPort> <numSens> <bodyId> <vrcId>  <bodyId> <vrcId>\n");
    //   //                     [0]       [1]       [2]     [3]        [4]        [5]       [6]      [7]      [8]      [9]
    //   fprintf(stderr,"Example: artserver sgi001 5555 20  5000 2 1 0 2 1\n");
    //   fprintf(stderr, "        body ids start with 0, flysticks ids start with 10\n");

    cout << progName << " [options] <adress> dev0[=map0]  [ dev1[=map1] ... ]\n"
         << "\n"
         << "   address = network addres of MotionStar\n"
         << "   devX    = Device Numbers of Bodies    (1, 2, ...)\n"
         << "   mapX    = COVER device numbers (0..31)\n"
         << "\n"
         << "Options:\n"
         << "\n"
         << "   -t <host:port>           set target to send tracking UDP packets\n"
         << "   --target=host:port       (default: localhost:7777)\n"
         << "\n"
         << "   -r <value>               transmission speed in Packets/sec\n"
         << "   --rate=value             (default: 20)\n"
         << "\n"
         << "   -H <hemisphere>          Hemisphere selection: FRONT REAR LEFT RIGHT UPPER LOWER\n"
         << "   --Hemisphere=Hemisphere  (default: FRONT)\n"
         << "\n"
         << "   -d ON                    Dual transmitter configuration\n"
         << "   --dual=ON                (default: OFF)\n"
         << "\n"
         << "   -a <angleMode>           Mode setup: Euler / Matrix\n"
         << "   --angle=mode             (default: Matrix)\n"
         << "\n"
         << "   -s <samplingRage>        Sampling Rate\n"
         << "   --sample=rate            (default: 80)\n"
         << "\n"
         << "   -o <numReceivers>        Use old BIOS with number receivers\n"
         << "   --oldBiosNumrec=<num>    (default: OFF)"
         << "\n"
         << "   -n <numRecv>             Number of Receivers attached\n"
         << "   --numbirds=number        (default: Number of devices in command line)\n"
         << "\n"
         << "   -b <buttonsystem>        Button system used to feed through: NONE MIKE\n"
         << "   --buttons=system         (default: NONE)"
         << "\n"
         << "\n"
         << "Examples:\n"
         << "\n"
         << "   MOTIONSTARserver myStar 1 2\n"
         << "                 Connect MotionStar named \"myStar\", send data \n"
         << "                 with 20 Hz frequency to localhost Port 7777.\n"
         << "                 Device 1   -->  COVER ID 1\n"
         << "                 Device 2   -->  COVER ID 2\n"
         << "\n"
         << "   MOTIONSTARserver 192.168.0.17 6=1 4=2 \n"
         << "                 use MotionStar at IP adress 192.168.0.17\n"
         << "                 Device 6   -->  COVER ID 1\n"
         << "                 Device 4   -->  COVER ID 2\n"
         << "\n"
         << "   MOTIONSTARserver --dual=ON -s 40 -b MIKE 192.168.0.17  3=0 1\n"
         << "                 Use a dual transmitter and sample with 40 Hz\n"
         << "                 Device 3   -->  COVER ID 0  with MIKE buttons\n"
         << "                 Device 1   -->  COVER ID 1\n"
         << endl;
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
    const char *hemisphereStr = arg.getOpt("-H", "--Hemisphere", "FRONT");
    const char *angleStr = arg.getOpt("-a", "--angle", "Matrix");
    const char *sampleRateStr = arg.getOpt("-s", "--sample", "80");
    const char *oldBiosStr = arg.getOpt("-o", "--oldBios", NULL);
    const char *dualStr = arg.getOpt("-d", "--dual", "OFF");
    const char *buttonStr = arg.getOpt("-b", "--buttons", "NONE");

    // Process BIOS : either nothing or numRecv
    bool oldBios = (oldBiosStr != NULL);
    int numRecv = 0;
    if (oldBios)
        numRecv = atoi(oldBiosStr);

    // Dual
    bool dual = (0 == strcasecmp(dualStr, "on"));

    // Sampling Rate
    int samplingRate = atoi(sampleRateStr);

    // Button system
    int buttonSystem;
    if (0 == strcasecmp(buttonStr, "NONE"))
        buttonSystem = B_NONE;
    else if (0 == strcasecmp(buttonStr, "MIKE"))
        buttonSystem = B_MIKE;
    else
    {
        cerr << "Button system " << buttonStr << " not yet supported" << endl;
        exit(0);
    }

    // --- Hemisphere
    enum birdTracker::hemisphere hemisphere;

    if (strncasecmp(hemisphereStr, "FRONT", 5) == 0)
        hemisphere = birdTracker::FRONT_HEMISPHERE;
    else if (strncasecmp(hemisphereStr, "REAR", 4) == 0)
        hemisphere = birdTracker::REAR_HEMISPHERE;
    else if (strncasecmp(hemisphereStr, "UPPER", 5) == 0)
        hemisphere = birdTracker::UPPER_HEMISPHERE;
    else if (strncasecmp(hemisphereStr, "LOWER", 5) == 0)
        hemisphere = birdTracker::LOWER_HEMISPHERE;
    else if (strncasecmp(hemisphereStr, "LEFT", 4) == 0)
        hemisphere = birdTracker::LEFT_HEMISPHERE;
    else if (strncasecmp(hemisphereStr, "RIGHT", 5) == 0)
        hemisphere = birdTracker::RIGHT_HEMISPHERE;
    else
    {
        cerr << "Illegal setting \"" << hemisphereStr
             << "\" for hemisphere" << endl;
        exit(0);
    }

    // --- Angle format
    birdTracker::dataformat angleMode;
    if (strncasecmp(angleStr, "Euler", 5) == 0)
        angleMode = birdTracker::FLOCK_POSITIONANGLES;
    else if (strncasecmp(angleStr, "Matrix", 6) == 0)
        angleMode = birdTracker::FLOCK_POSITIONMATRIX;
    else
    {
        cerr << "Illegal setting \"" << hemisphereStr
             << "\" for angle mode" << endl;
        exit(0);
    }

    // ---

    // --- verbose max. once per second
    float rate = atof(rateStr);
    int sendsPerVerbose;
    if (rate < 1)
        sendsPerVerbose = 1;
    else
        sendsPerVerbose = (int)rate;

    // --- select() delay record
    rate = 1.0 / rate;
    struct timeval delay;
    delay.tv_sec = (int)rate;
    delay.tv_usec = (int)(1e6 * (rate - delay.tv_sec));

    // +++++++++++++++ retrieve IP number +++++++++++++++

    unsigned long ip = getIP(arg[0]);
    struct in_addr addr = { ip };

    // +++++++++++++++ prepare mapping +++++++++++++++

    // COVER IDs for sensor[i]
    int hardwareID[MAXSENSORS], coverID[MAXSENSORS];

    if (arg.numArgs() > MAXSENSORS)
    {
        cerr << "Only " << MAXSENSORS << " sensors allowed" << endl;
    }

    int i;
    int numSensors = 0;
    for (i = 1; i < arg.numArgs(); i++)
    {
        coverID[numSensors] = i;
        bool ok = splitOpt(arg[i], hardwareID[numSensors], coverID[numSensors]);
        if (ok)
        {
            numSensors++;
        }
        else
        {
            cerr << "Illegal device selection: " << arg[i] << endl;
            exit(0);
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC MOTIONSTARserver %-5s  (C) 2005 VISENSO GmbH   +\n", MotionStarServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   UDP Target:             %-25s +\n", target);
    printf("  +   Send Rate:              %-3.1f Packets/s            +\n", 1.0 / rate);
    printf("  +   Motion Star Address:    %-25s +\n", inet_ntoa(addr));
    printf("  +   Hemisphere:             %-25s +\n", hemisphereStr);
    printf("  +   Angle Format:           %-25s +\n", angleStr);
    printf("  +   Old Bios:               %-25s +\n", ((oldBios) ? "Yes" : "No"));
    if (oldBios)
        printf("  +   Number of receivers:    %-25d +\n", numRecv);
    printf("  +   Dual Transmitter:       %-25s +\n", ((dual) ? "Yes" : "No"));
    if (buttonSystem == B_MIKE)
        printf("  +   MIKE Buttons at device: %-25d +\n", hardwareID[0]);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Mapping:                                            +\n");

    for (i = 0; i < numSensors; i++)
    {
        printf("  +   Target %-2d  --> COVER ID %-2d                        +\n",
               hardwareID[i], coverID[i]);
    }
    printf("  +-----------------------------------------------------+\n\n");

    /// ++++++++++++++++++++++++++ All parameters set - set up +++++++++++++++++++

    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);

    // create udp socket
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : "
             << sender.errorMessage() << endl;
        return -1;
    }
    printf("  - created UDP server\n");

    // open connection to MotionStar
    birdTracker tracker(inet_ntoa(addr), hardwareID[0],
                        oldBiosStr, ((oldBios) ? "OLD" : "NEW"),
                        false, false);

    if (!tracker.isConnected())
        exit(0);
    printf("  - created Bird Tracker Object\n");

    if (tracker.init() == -1)
        exit(0);
    printf("  - initialized\n");

    if (dual)
        tracker.DualTransmitter(1);

    if (tracker.setup(hemisphere, angleMode, samplingRate) < 0)
    {
        cerr << "hemisphere/angleMode/samplingRate setup failed" << endl;
        exit(0);
    }
    printf("  - set up\n");

    int num = tracker.getNumReceivers();

    printf("  - number of receivers: %d\n", num);

    // setup filters and buttons
    int d;
    for (d = 0; d < num; d++)
    {
        int i = tracker.hasButtons(d);
        if (!oldBios)
        {

            if (i)
                tracker.setFilter(d, 1, 1, 1, 1);
            else
                tracker.setFilter(d, 1, 1, 1, 1);
        }
        // #endif
    }
    printf("  - activated filtering\n");

    // forking client
    tracker.runContinuous();
    printf("  - forked server\n\n");

    /// ++++++++++++++++++++++++++ Star loop +++++++++++++++++++

    int frame = 0;
    while (1)
    {

        struct timeval delayT = delay; // select does not guarantee holding the value
        select(0, NULL, NULL, NULL, &delayT);

        for (i = 0; i < numSensors; i++)
        {
            float x, y, z;
            float mat[3][3];

            tracker.getPositionMatrix(hardwareID[i], &x, &y, &z, &(mat[0][0]), &(mat[0][1]), &(mat[0][2]), &(mat[1][0]), &(mat[1][1]), &(mat[1][2]), &(mat[2][0]), &(mat[2][1]), &(mat[2][2]));
            x /= 10.0;
            y /= 10.0;
            z /= 10.0;

            int b;
            unsigned int hwButton;
            tracker.getButtons(hardwareID[i], &hwButton);
            b = hwButton;

            char sendbuffer[2048];
            sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ 0 0 %s]",
                    coverID[i], b, x, y, z,
                    mat[0][0], mat[0][1], mat[0][2],
                    mat[1][0], mat[1][1], mat[1][2],
                    mat[2][0], mat[2][1], mat[2][2],
                    compatibilityString);

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
}
