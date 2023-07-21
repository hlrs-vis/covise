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
    bool verbose = false;
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
            verbose = true;
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
    a.setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
    qaw = TUIMainWindow::getInstance();
    qaw->show();
    int overridePort = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-p") == 0)
        {
            i++;
            if (i < argc)
            {
                sscanf(argv[i], "%d", &overridePort);
            }
            else
            {
                fprintf(stderr, "missing port argument for -p\n");
                break;
            }
        }
        if (strcmp(argv[i], "-3ds") == 0)
        {
            overridePort = 31804;
        }
    }
    if (overridePort != 0)
    {
        qaw->setPort(overridePort);
    }
    if (verbose)
    {
        std::cerr << "TabletUI: listening for connections on port " << overridePort << std::endl;
    }

    if (qaw->openServer() < 0)
    {
        return -1;
    }
    a.connect(&a, &QApplication::lastWindowClosed, qaw, &TUIMainWindow::storeGeometry);
    a.connect(&a, &QApplication::lastWindowClosed, &a, &QApplication::quit);
    return a.exec();
}
