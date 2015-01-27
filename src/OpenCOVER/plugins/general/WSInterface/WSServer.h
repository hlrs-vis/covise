/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WS_SERVER_H
#define WS_SERVER_H

#include <QThread>
#include "opencoverCOVERService.h"

class WSServer : public QThread, private opencover::COVERService
{
public:
    WSServer();
    virtual ~WSServer();

    virtual void run();
    virtual int run(int);

private:
    int port;
};

#endif
