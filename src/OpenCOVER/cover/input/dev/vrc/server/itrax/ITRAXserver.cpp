/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Device Server template
// (C) 2002-2003 VirCinity GmbH
// authors: we

#include <util/UDP_Sender.h>
#include <stdio.h>
#include <covise/covise.h> // either iostream or iostream.h
#include <unistd.h> // sleep
#include <math.h> // asin
#include <sys/time.h> // struct timeval

#include "istracker.h"

#define HOSTNAME "127.0.0.1"
#define UDPPORT "5500"
#define TRACKERCOMPORT 2
#define MOUSEID 1
#define TRACKERID 2

int main(int argc, char *argv[])
{
    using namespace std;

    string Target(HOSTNAME);
    string Port = (UDPPORT);
    int TrackerPortNr = TRACKERCOMPORT;
    int MouseID = MOUSEID;
    int TrackerID = TRACKERID;
    bool badArguments = false;

    cout << "Tracker daemon for COVER" << endl;
    // check arguments!
    if (argc == 1)
    {
        cout << "Using default values:" << endl;
    }
    else if (argc > 2)
    {
        int i = 1;
        for (; i < argc - 1; i += 2)
        {
            if (strcmp(argv[i], "-h") == 0)
                Target = argv[i + 1];
            else if (strcmp(argv[i], "-p") == 0)
                Port = argv[i + 1];
            else if (strcmp(argv[i], "-t") == 0)
                TrackerPortNr = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-MID") == 0)
                MouseID = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-TID") == 0)
                TrackerID = atoi(argv[i + 1]);
            else
            {
                badArguments = true;
            }
        }
        // arguments were not even
        if (i != argc)
            badArguments = true;
    }
    else
    {
        badArguments = true;
    }

    if (badArguments)
    {
        cout << "Wrong parameters" << endl;
        cout << "Usage: " << argv[0] << " -h hostname -p udp-port -t tracker-port -MID HandAddr -TID HeadAddr" << endl;
        cout << "-h   : hostname, where COVER is running (default " << HOSTNAME << ")" << endl;
        cout << "-p   : udp-port, where COVER is running (default " << UDPPORT << ")" << endl;
        cout << "-t   : number of comport for HMD tracker (e.g. 2 for com2 using /dev/ttyS1, default " << TRACKERCOMPORT << ")" << endl;
        cout << "-MID : ID of HAND device defined in Covise.config (Default " << MOUSEID << ")" << endl;
        cout << "-TID : ID of HEAD device defined in Covise.config (Default " << TRACKERID << ")" << endl;
        return -1;
    }

    // write connetion information
    cout << "hostname " << Target << endl;
    cout << "udp-port " << Port << endl;
    cout << "tracker-port com" << TrackerPortNr << endl;
    cout << "Handaddr " << MouseID << endl;
    cout << "Headaddr " << TrackerID << endl;
    cout << endl;

    // Open tracker device on serial port 2
    ISTracker Tracker(TrackerPortNr); // open Tracker connection at serialPort
    if (!Tracker.DeviceFound())
    {
        cerr << "no InterTrax Tracker found at commport " << TrackerPortNr << endl;
    }

    // variables for Covise
    float mat[3][3];
    float x, y, z;
    float rotX, rotY, rotZ;

    ///float Deg2Rad = PI/180.0;
    float Deg2Rad = M_PI / 180.0;

    // Process program arguments
    // ...

    x = 0.0;
    y = 0.0; //Distance Eye->Helmet in [mm]
    z = 0.0;

    // Create UDP server: IP address	+ Port number from VRCConfig
    Target += ":" + Port;
    UDP_Sender sender(Target.c_str());
    if (sender.isBad())
    {
        cerr << "Could not start UDP server to "
             << Target << " : " << sender.errorMessage() << endl;
        return -1;
    }
    else
    {
        cout << "Tracker is running" << endl;
        cout << "UDP server on: " << Target << endl;
    }

    bool DaemonRunning = true;
    char sendbuffer[256];
    while (DaemonRunning)
    {

        //... read tracker device
        if (Tracker.DeviceFound())
        {
            Tracker.getData(rotZ, rotX, rotY);

            fprintf(stderr, " degree from tracker: rotX=%f rotY=%f rotZ=%f\n", rotX, rotY, rotZ);

            //... create matrix

            //rotX *= Deg2Rad; dr: das liefert falsches VZ
            rotX *= -Deg2Rad;
            rotY *= Deg2Rad;
            rotZ *= Deg2Rad;

            mat[0][0] = cos(rotX) * cos(rotY); // cosx*cosz
            mat[0][1] = -cos(rotY) * sin(rotZ); // cosx*sinz
            mat[0][2] = -sin(rotY); // -sinz

            mat[1][0] = (-sin(rotX) * sin(rotY) * cos(rotZ)) + (cos(rotX) * sin(rotZ));
            mat[1][1] = (sin(rotX) * sin(rotY) * sin(rotZ)) + (cos(rotX) * cos(rotZ));
            mat[1][2] = -sin(rotX) * cos(rotY);

            mat[2][0] = (cos(rotX) * sin(rotY) * cos(rotZ)) + (sin(rotX) * sin(rotZ));
            mat[2][1] = (-cos(rotX) * sin(rotY) * sin(rotZ)) + (sin(rotX) * cos(rotZ));
            mat[2][2] = cos(rotX) * cos(rotY);

            sprintf(sendbuffer, "VRC %d %d [%5.0f %5.0f %5.0f] - [%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f] - [%5.2f %5.2f %5.2f]",
                    TrackerID, 0, 0.0, 10.0, 0.0,
                    mat[0][0], mat[0][1], mat[0][2],
                    mat[1][0], mat[1][1], mat[1][2],
                    mat[2][0], mat[2][1], mat[2][2],
                    0.0, 0.0, 0.0);

            sender.send(sendbuffer);
        }

#ifdef _DEBUG_LONG
        cout << "TraX = " << x << " TraY = " << y << " TraZ = " << z;
        cout << " RotX = " << rotX << " RotY = " << rotY << " RotZ = " << rotZ << "\r";
#endif

        struct timeval delayT;
        float rate = 1.0 / 10;
        delayT.tv_sec = (int)rate;
        delayT.tv_usec = (int)(1e6 * (rate - delayT.tv_sec));

        select(0, NULL, NULL, NULL, &delayT);
    }
}
