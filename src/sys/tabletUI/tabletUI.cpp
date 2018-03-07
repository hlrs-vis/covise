/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <net/message_types.h>

#define setfill qtsetfill
#define setprecision qtsetprecision
#define setw qtsetw
#include <qapplication.h>
#undef setfill
#undef setprecision
#undef setw
#ifndef _WIN32_WCE
#include <util/common.h>
#else
#include <winsock2.h>
#endif
#include "TUIApplication.h"



TUIMainWindow *qaw = NULL;

int main(int argc, char **argv)
{
#ifdef _WIN32
    unsigned short wVersionRequested = MAKEWORD(1, 1);
    struct WSAData wsaData;
    int err = WSAStartup(wVersionRequested, &wsaData);

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
#ifndef _WIN32_WCE
            AllocConsole();

            freopen("conin$", "r", stdin);
            freopen("conout$", "w", stdout);
            freopen("conout$", "w", stderr);
#endif
        }
        else
        {
            // disable "debug dialog": it prevents the application from exiting,
            // but still all sockets remain open
            DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
            SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
        }
    }

#ifdef NDEBUG
    // disable "debug dialog": it prevents the application from exiting,
    // but still all sockets remain open
    DWORD dwMode = SetErrorMode(SEM_NOGPFAULTERRORBOX);
    SetErrorMode(dwMode | SEM_NOGPFAULTERRORBOX);
#endif
#endif

    QApplication a(argc, argv);
    qaw = TUIMainWindow::getInstance();
    qaw->show();

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-d"))
        {
            int port = atoi(argv[i]);
            std::cerr << "TabletUI: listening for connections on port " << port << std::endl;
            qaw->setPort(port);
        }
    }

    if (qaw->openServer() < 0)
    {
        return -1;
    }
    a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));
    return a.exec();
}
