/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define setfill qtsetfill
#define setprecision qtsetprecision
#define setw qtsetw
#include <QApplication>
#undef setfill
#undef setprecision
#undef setw
#include "VRBServer.h"
#ifndef YAC
#include <covise/covise.h>
#endif
#include <config/CoviseConfig.h>
#include <util/environment.h>

#include "VRBapplication.h"

#include "listVRBs.h"

ApplicationWindow *mw;

void printPort(unsigned short port)
{
    std::cout << port << std::endl << std::endl << std::flush;
}

int main(int argc, char **argv)
{
    covise::setupEnvironment(argc, argv);
    bool help = false;
    bool printport = false;

    std::string exec(argv[0]);
    auto lastslash = exec.rfind('/');
    if (lastslash != std::string::npos)
    {
        exec = exec.substr(lastslash + 1);
    }
    bool gui = exec != "vrbc";

    for (size_t i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--console") == 0)
        {
            gui = false;
        }
        if (strcmp(argv[i], "--tui") == 0)
        {
            gui = false;
        }
        if (strcmp(argv[i], "--printport") == 0)
        {
            printport = true;
        }
        if (strcmp(argv[i], "--help") == 0)
        {
            help = true;
        }
        if (strcmp(argv[i], "--list") == 0)
        {
            listShm();
            return 0;
        }
        if (strcmp(argv[i], "--cleanShm") == 0)
        {
            cleanShm();
            return 0;
        }
    }
    if (help)
    {
        std::cerr << argv[0] << " [--help|--console|--tui]" << std::endl;
        std::cerr << "  --help:        print this message" << std::endl;
        std::cerr << "  --tui:         use text user interface" << std::endl;
        std::cerr << "  --console:     use text user interface" << std::endl;
        std::cerr << "  --list:        list all running VRBs" << std::endl;
        std::cerr << "  --printport:   let VRB chose a port and print it to commandline" << std::endl;

        return 0;
    }

    if (gui)
    {
        QApplication a(argc, argv);
#ifdef __APPLE__
        a.setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
        a.setWindowIcon(QIcon(":/icons/vrbIcon.png"));

        mw = new ApplicationWindow();
        mw->setWindowTitle("VRB");
        mw->show();
        a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));
        VRBServer server(gui);
        if (!server.openServer(printport))
        {
            return -1;
        }
        if (!printport && !server.startUdpServer())
        {
            cerr << "failed to open udp socket" << endl;
        }
        mw->setPort("Tcp", server.getPort());
        mw->setPort("Udp", server.getUdpPort());
        auto remover = placeSharedProcessInfo(server.getPort());
        if (printport)
        {
            printPort(server.getPort());
        }
        int exitcode = a.exec();

        return exitcode;
    }
    else
    {
        VRBServer server(gui);
        if (!server.openServer(printport))
        {
            return -1;
        }
        if (!printport && !server.startUdpServer())
        {
            cerr << "failed to open udp socket" << endl;
        }
        auto remover = placeSharedProcessInfo(server.getPort());
        if (printport)
        {
            printPort(server.getPort());
        }
        if (!gui)
        {
            server.loop();
        }
        int exitcode = 0;

        return exitcode;
    }
}
