/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   1/18/2010
**
**************************************************************************/

#include <QApplication>
#include "mainwindow.hpp"
#include <iostream>
#ifdef _WIN32
#include <winsock2.h>
#endif
#include "cover/coverconnection.hpp"

int main(int argc, char *argv[])
{
#ifdef _WIN32
    unsigned short wVersionRequested = MAKEWORD(1, 1);
    struct WSAData wsaData;
    int err = WSAStartup(wVersionRequested, &wsaData);
#endif
    std::cout << "\n\nStarting...\n  ODDlot: The OpenDRIVE Designer for Lanes, Objects and Tracks.\n" << std::endl;
    QApplication a(argc, argv);
    MainWindow w;
    QIcon icon(":/icons/oddlot.png"); 
    w.setWindowIcon(icon);
    w.show();
    QStringList args = a.arguments();
    if (args.size() > 1)
    {
        w.open(args.at(1));
    }
    COVERConnection::instance(); // create a cover connection

    return a.exec();
}
