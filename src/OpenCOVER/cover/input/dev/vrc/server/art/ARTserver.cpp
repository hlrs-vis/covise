/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// artserver
// input device daemon for A.R.T
// reads data from A.R.T udp port
// and writes data to a udp socket (host/port)
// (C) 2003 VirCinity GmbH
// authors: we, dr

#include <covise/covise.h>
#include <cover/input/dev/legacy/DTrack.h>
#ifdef WIN32
#include <windows.h>
#include <process.h>
#else
#include <sys/socket.h>
#include <strings.h>

#include <sys/ioctl.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#endif
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

using namespace covise;

// global variables for signal handler
DTrack *art = NULL;
UDP_Sender *startStop = NULL;

static const char *ArtServerVersion = "1.2";

static const char *ART_CameraOnCalc = "dtrack 10 3";
static const char *ART_CameraOff = "dtrack 10 0";

static const char *ART_StartSend = "dtrack 31";
static const char *ART_StopSend = "dtrack 32";

static const char *compatibilityString = ""; // put 0.00 in here for 5.2.3

// signal handler: correctly shut down art
void sigHandler(int signo)
{
    if (art)
    {
        fprintf(stderr, "Shutting down ARTserver for signal %d in Process %d\n", signo, getpid());
#ifndef WIN32
        if (art->getSlavePID() > 0)
        {
            fprintf(stderr, "Killing slave\n");
            kill(art->getSlavePID(), SIGKILL);
        }
#endif
        delete art;
        art = NULL;
        if (startStop) // we use receive port
        {
            startStop->send(ART_StopSend);
            sleep(1);
            startStop->send(ART_CameraOff);
            delete startStop;
        }
        exit(0);
    }
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
    int numRead = sscanf(option + 1, "%d=%d", &num1, &num2);
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
    if (*option == 'b' || *option == 'B') // Body:     ID = displayNo - 1
        targetNo += -1;
    else if (*option == 'f' || *option == 'F') // FlyStick: ID = displayNo - 1 + 10
        targetNo += -1 + 10;
    else
        return false;

    return true;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void displayHelp(const char *progName)
{
    //   fprintf(stderr,"Usage: artserver <targetHost> <targetPort> <sendRate> <artPort> <numSens> <bodyId> <vrcId>  <bodyId> <vrcId>\n");
    //   //                     [0]       [1]       [2]     [3]        [4]        [5]       [6]      [7]      [8]      [9]
    //   fprintf(stderr,"Example: artserver sgi001 5555 20  5000 2 1 0 2 1\n");
    //   fprintf(stderr, "        body ids start with 0, flysticks ids start with 10\n");

    cout << progName << " [options]  dev0[=map0] dev1[=map1] ...\n"
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
         << "   ARTserver F1 B1          Receive ART information on port 5000, and send \n"
         << "                            it with 20 Hz frequency to localhost Port 7777.\n"
         << "                            send Flystick F1  -->  COVER ID 0\n"
         << "                            send Body 1       -->  COVER ID 1\n"
         << "\n"
         << "   ARTserver F1=2 B1=3      as before, but \n"
         << "                            send Flystick F1  -->  COVER ID 1\n"
         << "                            send Body 1       -->  COVER ID 3\n"
         << "\n"
         << "   ARTserver --sendStart=10.0.0.1:5001 F1=2 B1=3\n"
         << "                            as before, but send signal Tracker PC at address\n"
         << "                            10.0.0.1, Port 5001 to start/stop measurement \n"
         << "                            before/after ARTserver execution\n"
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
    const char *portStr = arg.getOpt("-p", "--port", "5000");
    const char *sendStr = arg.getOpt("-s", "--sendStart", NULL);
    const char *cyberC = arg.getOpt("-c", "--cyberClassroom", "0");

    // get port on which the dtrack sends the data
    int artPort = atoi(portStr);
    float rate = (float)atof(rateStr);

    // verbose max. once per second
    int sendsPerVerbose;
    if (rate < 1)
        sendsPerVerbose = 1;
    else
        sendsPerVerbose = (int)rate;

    // select() delay record
    rate = 1.0f / rate;
    struct timeval delay;
    delay.tv_sec = (int)rate;
    delay.tv_usec = (int)(1e6 * (rate - delay.tv_sec));

    // +++++++++++++++ prepare mapping +++++++++++++++

    int artID[MAXSENSORS], coverID[MAXSENSORS]; // COVER IDs for sensor[i]

    if (arg.numArgs() > MAXSENSORS)
    {
        cerr << "Only " << MAXSENSORS << " sensors allowed" << endl;
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
            cerr << "Illegal device selection: " << arg[i] << endl;
            exit(0);
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC ARTserver %-10s    (C) 2005 VISENSO GmbH +\n", ArtServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   ART port:          %-5d                          +\n", artPort);
    printf("  +   ART startup:       %-30s +\n", ((sendStr) ? sendStr : "(Not used)"));
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
        cerr << "Could not start UDP server to "
             << target << " : "
             << sender.errorMessage() << endl;
        return -1;
    }
    // create second udp socket
    UDP_Sender sender2("localhost:8888");
    if (strcmp(cyberC, "0") && sender2.isBad())
    {
        cerr << "Could not start UDP server to second target localhost:8888 : " << sender2.errorMessage() << endl;
        return -1;
    }

    // we use the ReceivePort
    if (sendStr)
    {
        startStop = new UDP_Sender(sendStr);
        startStop->send(ART_CameraOnCalc);
        cerr << "Sent \"" << ART_CameraOnCalc << "\" to " << sendStr << endl;
        sleep(1);
        startStop->send(ART_StartSend);
        cerr << "Sent \"" << ART_StartSend << "\" to " << sendStr << endl;
    }

    // fork separate process which reads data from udp port and puts it into shm
    art = new DTrack(artPort, NULL);

    int frame = 1;
    sleep(2); // wait for server to come up

    while (1)
    {

        struct timeval delayT = delay; // select does not guarantee holding the value
        select(0, NULL, NULL, NULL, &delayT);

        for (i = 0; i < numSensors; i++)
        {
            float x, y, z;
            float mat[3][3];

            art->getPositionMatrix(artID[i], &x, &y, &z, &(mat[0][0]), &(mat[0][1]), &(mat[0][2]), &(mat[1][0]), &(mat[1][1]), &(mat[1][2]), &(mat[2][0]), &(mat[2][1]), &(mat[2][2]));
            x /= 10.0;
            y /= 10.0;
            z /= 10.0;

            if (x == 0.0 && y == 0.0 && z == 0.0)
            {
                x = 0.001f;
                y = 0.001f;
                z = 0.001f;
            }

            unsigned int b;
            art->getButtons(artID[i], &b);

            char sendbuffer[2048];
            sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ 0 0 %s]",
                    coverID[i], b, x, y, z,
                    mat[0][0], mat[0][1], mat[0][2],
                    mat[1][0], mat[1][1], mat[1][2],
                    mat[2][0], mat[2][1], mat[2][2],
                    compatibilityString);

            sender.send(sendbuffer);
            if (strcmp(cyberC, "0"))
                sender2.send(sendbuffer);

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

    return 0;
}
