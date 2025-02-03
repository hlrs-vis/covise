/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM MAGELLANserver
//
// MAGELLAN handling server sending UDP packets for MAGELLAN events
//
// Initial version: 2003-01-22 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:  1.0 Initial version
//
//
#include <covise/covise.h>
//#include <iostream.h>
#define ROT MagellanRotationMatrix(matr, -A, -C, -B)
#include <unistd.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/time.h>

#include <sys/ioctl.h>

#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

#include "xdrvlib.h"

#define MAXSENS 16000.0
#define STARTSENS 1000.0
#define MINSENS 400.0

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}

static const char *MagellanServerVersion = "1.0";

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
    const char *target = arg.getOpt("-t", "--target", "localhost:7777");

    if (arg.numArgs() < 1)
    {
        cerr << "\nCover station ID missing" << endl;
        exit(0);
    }

    //
    int stationID;
    sscanf(arg[0], "%d", &stationID);

    ////// Echo it
    fprintf(stderr, "\n");
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + VRC MAGELLANserver %-10s (C) 2005 VISENSO GmbH   +\n", MagellanServerVersion);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "  + Settings:                                           +\n");
    fprintf(stderr, "  +   UDP Target:        %-30s +\n", target);
    fprintf(stderr, "  +   COVER station ID:  %-2d                             +\n", stationID);
    fprintf(stderr, "  +-----------------------------------------------------+\n");
    fprintf(stderr, "\n");

    /// establich some signal handlers
    signal(SIGINT, sigHandler);
    signal(SIGPIPE, sigHandler);
    signal(SIGCHLD, sigHandler);
    signal(SIGTERM, sigHandler);
    signal(SIGHUP, sigHandler);
#ifndef __linux__
    prctl(PR_TERMCHILD); // Exit when parent does
#else
    prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

    /// Open UDP sender
    UDP_Sender sender(target);
    if (sender.isBad())
    {
        cerr << "Could not start UDP server "
             << target << endl;
        return -1;
    }

    Display *display;
    Window root, window;

    int screennumber, width, height;
    XSizeHints *sizehints;
    XWMHints *wmhints;
    XClassHint *classhints;

    char *WinName = "Magellan 3D Controller";
    XTextProperty WindowName;
    GC wingc;
    XGCValues xgcvalues;

    XEvent report;
    MagellanFloatEvent MagellanEvent;

    XComposeStatus compose;
    KeySym keysym;

    int MagellanDemoEnd = FALSE;
    char MagellanBuffer[256];

    /****************** Open a Window ******************************************/
    sizehints = XAllocSizeHints();
    wmhints = XAllocWMHints();
    classhints = XAllocClassHint();
    if ((sizehints == NULL) || (wmhints == NULL) || (classhints == NULL))
    {
        fprintf(stderr, "Can't allocate memory! Exit ... \n");
        exit(-1);
    };

    display = XOpenDisplay(NULL);
    if (display == NULL)
    {
        fprintf(stderr, "Can't open display! Exit ... \n");
        exit(-1);
    };

    screennumber = DefaultScreen(display);
    width = DisplayWidth(display, screennumber);
    height = DisplayHeight(display, screennumber);
    root = DefaultRootWindow(display);
    //   window = XCreateSimpleWindow( display, root, 0,0, width/5*3,height/8, 20,
    //                                 BlackPixel(display,screennumber),
    //                                 WhitePixel(display,screennumber) );

    window = XCreateSimpleWindow(display, root, 0, 0, 1, 1, 1,
                                 BlackPixel(display, screennumber),
                                 WhitePixel(display, screennumber));

    printf("Magellan: xapp.c\n");
    printf("Magellan Root Window=%08X \nMagellan Application Window=%08X \n\n",
           root, window);

    XStringListToTextProperty(&WinName, 1, &WindowName);

    wmhints->initial_state = NormalState;
    wmhints->input = TRUE;
    wmhints->flags = StateHint | InputHint;

    classhints->res_name = "Magellan 3D Controller";
    classhints->res_class = "BasicWindow";
    XSetWMProperties(display, window, &WindowName, NULL, argv,
                     argc, sizehints, wmhints, classhints);

    XMapWindow(display, window);
    xgcvalues.foreground = BlackPixel(display, 0);
    xgcvalues.background = WhitePixel(display, 0);
    wingc = XCreateGC(display, window, GCForeground | GCBackground, &xgcvalues);

    /************************* Create 3D Event Types ***************************/
    if (!MagellanInit(display, window))
    {
        fprintf(stderr, "No driver is running. Exit ... \n");
        exit(-1);
    };

    /************************* Main Loop ***************************************/
    XSelectInput(display, window, KeyPressMask | KeyReleaseMask);
    double MagellanSensitivity;
    char sendbuffer[2048];
    double X = 0.0, Y = 0.0, Z = 0.0;
    double SENS = STARTSENS;
    double matr[4][4];
    float A = 0.0, B = 0.0, C = 0.0;
    int button = 0;

    while (MagellanDemoEnd == FALSE)
    {

        XNextEvent(display, &report);
        switch (report.type)
        {
        case KeyRelease: /* ... */
            break;

        case KeyPress:
            MagellanDemoEnd = keysym == XK_Escape;
            break;

        case ClientMessage:
            int event = MagellanTranslateEvent(display, &report, &MagellanEvent, 1.0, 1.0);
            switch (event)
            {
            case MagellanInputMotionEvent:
                break;

            case MagellanInputButtonPressEvent:
                button = 1 << (MagellanEvent.MagellanButton - 1);

                switch (MagellanEvent.MagellanButton)
                {
                case 5:
                    SENS *= 2.0;
                    break;
                case 6:
                    SENS *= 0.5;
                    break;
                case 7:
                    SENS = STARTSENS;
                    break;
                case 8:
                    SENS = STARTSENS;
                    X = Y = Z = A = B = C = 0.0;
                };
                break;

            case MagellanInputButtonReleaseEvent:
                button = 0;
                break;

            default: /* another ClientMessage event */
                break;
            };
            if (SENS > MAXSENS)
                SENS = MAXSENS;
            if (SENS < MINSENS)
                SENS = MINSENS;
            X += MagellanEvent.MagellanData[MagellanX] / SENS;
            Y += MagellanEvent.MagellanData[MagellanY] / SENS;
            Z += MagellanEvent.MagellanData[MagellanZ] / SENS;
            A += MagellanEvent.MagellanData[MagellanA] / SENS / 5.0;
            B += MagellanEvent.MagellanData[MagellanB] / SENS / 5.0;
            C += MagellanEvent.MagellanData[MagellanC] / SENS / 5.0;
            if (A > 360.0)
                A -= 360.0;
            if (B > 360.0)
                B -= 360.0;
            if (C > 360.0)
                C -= 360.0;
            if (A < -360.0)
                A += 360.0;
            if (B < -360.0)
                B += 360.0;
            if (C < -360.0)
                C += 360.0;
            ROT;
            sprintf(sendbuffer, "VRC %d %3d [%5.8f %5.8f %5.8f] - [%5.8f %5.8f %5.8f %5.8f %5.8f %5.8f %5.8f %5.8f %5.8f] - [0 0]",
                    stationID, button,
                    X, Z, Y,
                    matr[0][0], matr[0][1], matr[0][2],
                    matr[1][0], matr[1][1], matr[1][2],
                    matr[2][0], matr[2][1], matr[2][2]);
            fprintf(stderr, "%s\n", sendbuffer);
            sender.send(sendbuffer, strlen(sendbuffer) + 1);
            break;
        };
    };

    MagellanClose(display);
    XCloseDisplay(display);
    return true;
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
         << "Examples:\n"
         << "\n"
         << "   " << progName << " 3            Read MAGELLAN device at default port and send\n"
         << "                           data to localhost:7777 with ID=3\n"
         << "\n"
         << "   " << progName << " -t visenso:6666 4\n"
         << "                           Read MAGELLAN device  and send data\n"
         << "                           to Host \"visenso\" Port 6666 with ID=4\n"
         << endl;
}
