/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/time.h>
#else
#include <stdio.h>
#include <process.h>
#include <io.h>
#endif

#include <fcntl.h>
#include <stdlib.h>

#include <util/covise_version.h>
#include "CRBConnection.h"
#include <dmgr/dmgr.h>
#include <covise/Covise_Util.h>
#include <config/CoviseConfig.h>

#ifdef _NEC
#include <sys/socke.h>
#endif

int main(int argc, char *argv[])
{

    //moduleList mod;
    //char *instance;
    //int proc_id;

    if (argc != 4)
    {
        cerr << "crbProxy (CoviseRequestBrokerProxy) with inappropriate arguments called\n";
        exit(-1);
    }

#ifdef _WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(1, 1);

    err = WSAStartup(wVersionRequested, &wsaData);
#endif

    int port = atoi(argv[1]);
    //Host *host = new Host(argv[2]);
    int id = atoi(argv[3]);
    //proc_id = id;
    //instance = argv[4];
    //sprintf(err_name,"err%d",id);

    CRBConnection *datamgrProx = new CRBConnection(port, argv[2], id);

    datamgrProx->execCRB(argv[4]);
    while (1)
    {
        datamgrProx->processMessages();
    }


   
}
