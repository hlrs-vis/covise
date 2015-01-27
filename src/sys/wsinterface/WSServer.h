/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WS_SERVER_H
#define WS_SERVER_H

#include <QRunnable>
#include <QThread>
#include "WSCOVISEService.h"

namespace covise
{

class WSServer : public QThread, private covise::WSCOVISEService
{
public:
    WSServer();
    virtual ~WSServer();
    static WSServer *instance();

    virtual void run();
    virtual int run(int);
    int getPort() const;

private:
    int port;
    static QString coviseDir;

    static WSServer *instancePtr;

    static int httpGet(soap *soap);

    class Thread : public QRunnable
    {
    public:
        Thread(covise::WSCOVISEService *service);
        virtual void run();

    private:
        covise::WSCOVISEService *service;
    };
};
}
#endif
