/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// PROGRAM f3
//
// Sends commands to a Projectiondesign F3 Projector
//
//
// Initial version: 2006-01-23 Christof Schwenzer
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2006 VISENSO GmbH
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
#include "SerialCom.h"
#include <util/UDP_Sender.h>
#include <util/ArgsParser.h>

// signal handler: correctly shut down flock
void sigHandler(int signo)
{
    fprintf(stderr, "Signal %d caught by Handler\n", signo);
    exit(0);
}
void establishSignalHandlers();

/// ++++ Forward-declaration of functions defined later in the text ++++
// show help
void displayHelp(const char *progName);
void showbuffer(unsigned char *bytes, UDP_Sender &sender, int stationID);
///////// program starts here

int main(int argc, char *argv[])
{
    ArgsParser arg(argc, argv);
    if (argc < 2)
    {
        displayHelp("f3");
        exit(1);
    }

    /// establish some signal handlers
    establishSignalHandlers();
    const char *first = arg.getOpt("-f", "--first", "192.168.0.222:1025");
    const char *second = arg.getOpt("-s", "--second", "192.168.0.223:1025");
    const char *command = arg.getOpt("-c", "--command", ":ivga");
    char commandBuf[1024];
    sprintf(commandBuf, command);
    strcat(commandBuf, "\n");
    UDP_Sender senderFirst(first);
    if (senderFirst.isBad())
    {
        cerr << "Could not establish connection to projector " << endl;
    }
    else
    {
        senderFirst.send(command, 1 + strlen(commandBuf));
        cerr << "Sending the command" << endl << "  " << command << endl << "to the projector with IP address and Port            " << first << endl;
    }
    UDP_Sender senderSecond(second);
    if (senderSecond.isBad())
    {
        cerr << "Could not establish connection to projector " << endl;
    }
    else
    {
        senderSecond.send(command, 1 + strlen(commandBuf));
        cerr << "as well as to the projector with IP address and Port " << second << endl;
    }

    return 0;
}

void establishSignalHandlers()
{
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
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Help

// show help
void displayHelp(const char *progName)
{
    cout << progName << endl << "Send commands to a Projectiondesign F3 Projector " << endl
         << "   -f <host:port>          set IP:Port of the first projector to send the commands" << endl
         << "   --first=host:port      (default: 192.168.0.222:1025)" << endl
         << "   -s <host:port>          set IP:Port of the second projector to send the commands" << endl
         << "   --second=host:port      (default: 192.168.0.223:1025)" << endl
         << "   -c <ommand>          set IP:Port of the second projector to send the commands" << endl
         << "   --command=<command>      (default: \":ivga\"" << endl
         << "------------------Overview of the Projector's commands-------------------------" << endl
         << ":ivga                      Select VGA as source for projector                  " << endl
         << ":idvi                      Select DVI as source for projector                  " << endl
         << ":ibnc                      Select BNC as source for projector                  " << endl
         << ":isvi                      Select S-Video as source for projector              " << endl
         << ":icvi                      Select Composite video as source for projector      " << endl
         << ":powr1                     Power on Projector                                  " << endl
         << ":powr0                     Power off Projector                                 " << endl
         << ":test 1                    Test image on                                       " << endl
         << ":test 0                    Test image off                                      " << endl
         << ":ecom 1                    Eco mode on                                         " << endl
         << ":ecom 0                    Eco mode off                                        " << endl
         << ":brig <number>             Set Brightness to <number> e.g.                     " << endl
         << ":brig 60                   Set Brightness to 60                                " << endl
         << ":cntr <number>             Set contrast to <number>                            " << endl
         << ":ct65                      Set color temperature to 6500K                      " << endl
         << ":ct73                      Set color temperature to 7300K                      " << endl
         << ":ct93                      Set color temperature to 9300K                      " << endl
         << ":desk                      Select orientation Desktop Front                    " << endl
         << ":ceil                      Select orientation Ceiling Front                    " << endl
         << ":rdes                      Select orientation Desktop Rear                     " << endl
         << ":rcei                      Select orientation Rear Ceiling                     " << endl
         << "                                                                               " << endl
         << "     Examples                                                                  " << endl
         << "f3 -c ':powr0'             switch off the projectors with ip address           " << endl
         << "                           192.168.0.222 and 192.168.0.223                     " << endl
         << "f3 -c 'brig 60' -second \"dummy\"  set brightness only                           " << endl
         << "                                 of the projector with address 192.168.0.222   " << endl
         << "                                 to the value 60                               " << endl;
}
