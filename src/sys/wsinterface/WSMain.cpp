/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <QApplication>
#include <QtGlobal>

#include "WSMessageHandler.h"
#include "WSServer.h"

#ifndef YAC
#include "net/covise_socket.h"
#endif

//========================================================
// main loop
//========================================================
int main(int argc, char **argv)
{

#ifndef YAC
    covise::Socket::initialize();
#endif

    // start user interface process
    QApplication a(argc, argv);

    new covise::WSMessageHandler(argc, argv);
    new covise::WSServer();
    return a.exec();
}
