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
//#include <cover/DTrack.h>
#include "DTrack.hpp" // use SDK from A.R.T.
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>
// #include "UDP_Sender.h"
// #include "ArgsParser.h"

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

using namespace covise;

#define MAXSENSORS 20
#define MAXBODYS 10
#define MAXFLYSTICKS 10
#define MAXBYTES 4000

using namespace covise;

// global variables for signal handler
DTrack *art = NULL;
bool useReceivePort = false;

// global variables for mapping of body/flystick id to corresponding COVER id
//int bodyID2coverID[MAXSENSORS];
std::map<int, int> bodyID2coverID;
//int flystickID2coverID [MAXSENSORS];
std::map<int, int> flystickID2coverID;
int numBodies = 0;
int numFlysticks = 0;

static const char *ArtServerVersion = "2.0";

static const char *compatibilityString = ""; // put 0.00 in here for 5.2.3

// signal handler: correctly shut down art
void sigHandler(int signo)
{
    (void)signo;

    if (art)
    {
        if (useReceivePort)
        {
            if (!(art->cmd_sending_data(false)))
            {
                std::cerr << "Sending command to switch off sending of data failed!" << std::endl;
            }

            if (!(art->cmd_cameras(false)))
            {
                std::cerr << "Sending command to switch off cameras to failed!" << std::endl;
            }
        }

        delete art;
        art = NULL;

        exit(0);
    }
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Split options of form   B1=0 F10=7 ...
//       into targetNo (B1=0, B2=1, ... F1=10, F2=11, ...) and mapping value

bool splitOpt(const char *option) //, int &targetNo, int &mapNo)
{
    // no string or long string: error
    if (!option || strlen(option) > 6)
        return false;

    // we have 1 char + number=number
    int num1, num2;
    int numRead = sscanf(option + 1, "%d=%d", &num1, &num2);

    // FlyStick or Body?  In either case, correct numbers
    if (*option == 'b' || *option == 'B')
    {
        if (numRead == 2)
        {
            bodyID2coverID[num1] = num2;
        }
        else
        {
            bodyID2coverID[num1] = numBodies + numFlysticks;
        }
        numBodies++;
    }
    else if (*option == 'f' || *option == 'F')
    {
        if (numRead == 2)
        {
            flystickID2coverID[num1] = num2;
        }
        else
        {
            flystickID2coverID[num1] = numBodies + numFlysticks;
        }
        numFlysticks++;
    }
    else
    {
        return false;
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
         << "   ART_DTRACKserver F1 B1   Receive ART information on port 5000, and send \n"
         << "                            it with 20 Hz frequency to localhost Port 7777.\n"
         << "                            send first Flystick F1  -->  COVER ID 0\n"
         << "                            send first Body B1       -->  COVER ID 1\n"
         << "\n"
         << "   ART_DTRACKserver F1=2 B1=3\n"
         << "                            as before, but \n"
         << "                            send first Flystick F1  -->  COVER ID 2\n"
         << "                            send first Body B1       -->  COVER ID 3\n"
         << "\n"
         << "   ART_DTRACKserver --sendStart=10.0.0.1:5001 F1=2 B1=3\n"
         << "                            as before, but send signal Tracker PC at address\n"
         << "                            10.0.0.1, Port 5001 to start/stop measurement \n"
         << "                            before/after ARTserver execution\n"
         << "\n"
         << "Body and Flystick enumeration start with 1, i.e. to use F0 or B0 is an error.\n"
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
    std::string cyberC = arg.getOpt("-c", "--cyberClassroom", "0");

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

    if (arg.numArgs() > MAXSENSORS)
    {
        cerr << "Only " << MAXSENSORS << " sensors allowed" << endl;
    }

    int i, j;
    for (i = 0; i < arg.numArgs(); i++)
    {
        bool ok = splitOpt(arg[i]); //, artID[numSensors],coverID[numSensors]);
        if (!ok)
        {
            cerr << "Illegal device selection: " << arg[i] << endl;
            exit(0);
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Show what we will do

    printf("\n");
    printf("  +-----------------------------------------------------+\n");
    printf("  + VRC ARTserver %-10s     (C) 2005 VISENSO GmbH +\n", ArtServerVersion);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Settings:                                           +\n");
    printf("  +   ART port:          %-5d                          +\n", artPort);
    printf("  +   ART startup:       %-30s +\n", ((sendStr) ? sendStr : "(Not used)"));
    printf("  +   UDP Target:        %-30s +\n", target);
    printf("  +   Send Rate:         %-3.1f Packets/s                 +\n", 1.0 / rate);
    printf("  +-----------------------------------------------------+\n");
    printf("  + Mapping:                                            +\n");
    //for (i = 0; i < numFlysticks; i++)
    for (std::map<int, int>::const_iterator iter = flystickID2coverID.begin(); iter != flystickID2coverID.end(); iter++)
    {
        printf("  +   Target F%-2d  --> COVER ID %-2d                       +\n",
               (*iter).first /*i+1*/,
               (*iter).second /*flystickID2coverID[i]*/);
    }
    //for (i = 0; i < numBodies; i++)
    for (std::map<int, int>::const_iterator iter = bodyID2coverID.begin(); iter != bodyID2coverID.end(); iter++)
    {
        printf("  +   Target B%-2d  --> COVER ID %-2d                       +\n",
               (*iter).first /*i+1*/,
               (*iter).second /*bodyID2coverID[i]*/);
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
    if (cyberC != "0" && sender2.isBad())
    {
        cerr << "Could not start UDP server to "
             << target << " : "
             << sender.errorMessage() << endl;
        return -1;
    }

    // we use the ReceivePort
    if (sendStr)
    {
        useReceivePort = true;

        char remote_host[1024];
        remote_host[1023] = '\0';
        strncpy(remote_host, sendStr, 1023);

        // grep out port part
        char *dpoint = strchr(remote_host, ':');
        if (!dpoint)
        {
            std::cerr << "Target must be hostname:port" << std::endl;
            return -1;
        }

        // convert to int
        int remote_port;
        int retval;
        retval = sscanf(dpoint + 1, "%d", &remote_port);
        if (retval != 1)
        {
            std::cerr << "UDP_Sender::UDP_Sender: sscanf failed" << std::endl;
            return -1;
        }

        // mask away port from copied string - result is host name
        *dpoint = '\0';

        art = new DTrack(artPort, remote_host, remote_port);
    }
    else
    {
        art = new DTrack(artPort);
    }

    if (!art->valid())
    {
        std::cout << "Warning: DTrack init error" << std::endl;
    }

    if (useReceivePort)
    {
        if (!(art->cmd_cameras(true)))
        {
            std::cerr << "Sending command to switch on cameras to " << sendStr << " failed!" << std::endl;
        }

        if (!(art->cmd_sending_data(true)))
        {
            std::cerr << "Sending command to switch on sending of data to " << sendStr << " failed!" << std::endl;
        }
    }

    dtrack_flystick_type flystick;
    dtrack_body_type body;
    int frame = 1;

    while (1)
    {

        struct timeval delayT = delay;
        select(0, NULL, NULL, NULL, &delayT);

        bool ok = art->receive();

        if (!ok)
        {
            if (art->timeout())
            {
                printf("--- timeout while waiting for udp data\n");
                continue;
            }

            if (art->udperror())
            {
                printf("--- error while receiving udp data\n");
                continue;
            }

            if (art->parseerror())
            {
                printf("--- error while parsing udp data\n");
                continue;
            }
            printf("--- unknown error");
            continue;
        }

        for (i = 0; i < art->get_num_flystick(); i++)
        {
            //if (i >= numFlysticks) {
            if (flystickID2coverID.find(i + 1) == flystickID2coverID.end())
            {
                continue;
            }

            flystick = art->get_flystick(i);

            if (flystick.quality < 0)
            {
                if (frame % sendsPerVerbose == 0)
                {
                    printf("Flystick F%d not visible\n", flystick.id + 1);
                }
                continue;
            }

            // convert from n array-int -> 1 binary-state int
            int button = 0;
            int k = 1;
            for (j = 0; j < flystick.num_button; j++)
            {
                if (flystick.button[j])
                {
                    button += k;
                }
                k *= 2;
            }

            float joystick[2];
            joystick[0] = joystick[1] = 0.0f;
            for (j = 0; (j < flystick.num_joystick) || (j < 2); j++)
            {
                joystick[j] = flystick.joystick[j];
            }

            // mm -> cm
            flystick.loc[0] /= 10.0;
            flystick.loc[1] /= 10.0;
            flystick.loc[2] /= 10.0;

            char sendbuffer[2048];
            sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ %6.3f %6.3f %s]",
                    flystickID2coverID[i + 1] /*flystickID2coverID[i]*/,
                    button,
                    flystick.loc[0], flystick.loc[1], flystick.loc[2],
                    flystick.rot[0], flystick.rot[1], flystick.rot[2],
                    flystick.rot[3], flystick.rot[4], flystick.rot[5],
                    flystick.rot[6], flystick.rot[7], flystick.rot[8],
                    joystick[0], joystick[1],
                    compatibilityString);

            sender.send(sendbuffer);
            if (cyberC != "0")
                sender2.send(sendbuffer);

            if (frame % sendsPerVerbose == 0)
            {
                fprintf(stderr, "%s\n", sendbuffer);
            }
        }

        for (i = 0; i < art->get_num_body(); i++)
        {
            //if (i >= numBodies) {
            if (bodyID2coverID.find(i + 1) == bodyID2coverID.end())
            {
                continue;
            }

            body = art->get_body(i);

            if (body.quality < 0)
            {
                if (frame % sendsPerVerbose == 0)
                {
                    printf("Body B%d not visible\n", body.id + 1);
                }
                continue;
            }

            // mm -> cm
            body.loc[0] /= 10.0;
            body.loc[1] /= 10.0;
            body.loc[2] /= 10.0;

            if (body.loc[0] == 0.0 && body.loc[1] == 0.0 && body.loc[2] == 0.0)
            {
                body.loc[0] = 0.001f;
                body.loc[1] = 0.001f;
                body.loc[2] = 0.001f;
            }

            char sendbuffer[2048];
            sprintf(sendbuffer, "VRC %d %3d [%5.1f %5.1f %5.1f] - [%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f] - [ 0 0 %s]",
                    bodyID2coverID[i + 1] /*bodyID2coverID[i]*/,
                    0 /* button */,
                    body.loc[0], body.loc[1], body.loc[2],
                    body.rot[0], body.rot[1], body.rot[2],
                    body.rot[3], body.rot[4], body.rot[5],
                    body.rot[6], body.rot[7], body.rot[8],

                    compatibilityString);

            sender.send(sendbuffer);
            if (cyberC != "0")
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
