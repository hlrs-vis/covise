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

VRBServer server;
ApplicationWindow *mw;

int main(int argc, char **argv)
{
    covise::setupEnvironment(argc, argv);

    QApplication a(argc, argv);
#ifdef __APPLE__
    a.setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
    mw = new ApplicationWindow();
    mw->setWindowTitle("VRB");
    mw->show();
    a.connect(&a, SIGNAL(lastWindowClosed()), &a, SLOT(quit()));

    if (server.openServer() < 0)
    {
        return -1;
    }
    return a.exec();
}
