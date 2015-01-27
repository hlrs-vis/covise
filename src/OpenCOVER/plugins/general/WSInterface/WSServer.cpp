/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "opencover.nsmap"
SOAP_NMAC struct Namespace *namespaces = opencover_namespaces;

#include "WSServer.h"

#include <iostream>

// Try to detect gsoap version
#if SOAP_NULL > 19
#define GSOAP_VERSION 0x02070a
#define GSOAP_MAJOR 2
#define GSOAP_MINOR 7
#define GSOAP_PATCH 10
#else
#define GSOAP_VERSION 0x020709
#define GSOAP_MAJOR 2
#define GSOAP_MINOR 7
#define GSOAP_PATCH 9
#endif

using std::cerr;
using std::endl;

WSServer::WSServer()
{
    this->port = 32190;
    start();
}

WSServer::~WSServer()
{
    terminate();
}

void WSServer::run()
{

    this->send_timeout = 60; // 60 seconds
    this->recv_timeout = 60; // 60 seconds

    cerr << "WSServer::run() info: binding to port " << this->port << endl;

    SOAP_SOCKET masterSocket, slaveSocket;
    masterSocket = bind(0, this->port, 100);

    bool isPortChanged = false;

    while (!soap_valid_socket(masterSocket))
    {
        cerr << "OpenCoverSOAP::Internal::run() info: cannot bind to port " << this->port << endl;
        ++port;
        masterSocket = bind(0, this->port, 100);
        isPortChanged = true;
    }

    cerr << "WSServer::run() info: bound to port " << this->port << endl;

    //if (isPortChanged) acws->portChangedFromInternal(this->port);

    while (true)
    {
        slaveSocket = accept();
        if (!soap_valid_socket(slaveSocket))
        {
            if (this->errnum)
            {
                cerr << "OpenCoverSOAP::Internal::run() err: SOAP error:" << endl;
#if GSOAP_VERSION == 0x020709
                soap_print_fault(this, stderr);
#elif GSOAP_VERSION == 0x02070a
                soap_print_fault(stderr);
#endif
                exit(1); // TODO Handle gracefully
            }
            cerr << "OpenCoverSOAP::Internal::run() err: server timed out" << endl;
            //exit(1);  // TODO Handle gracefully
        }

        serve();
    }
}

int WSServer::run(int value)
{
    return opencover::COVERService::run(value);
}
