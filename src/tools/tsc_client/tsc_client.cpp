/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:             ++
// ++                                                                     ++
// ++ Author:                  ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#include "tsc_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int AsyncClient::open_conn(Host *host, int port)
{
    int id = 50;

    id = (int)getpid();

    m_pconn = new ClientConnection(host, port, id, SIMPLEPROCESS);
    if (m_pconn)
        return 0;
    else
        return -1;
}

int AsyncClient::send_msg(Message *msg)
{
    m_pconn->send_msg(msg);
    return 0;
}

int main(int argc, char **argv)
{

    Host *p_host;
    Message *p_msg;
    int srv_port;
    AsyncClient myClient;

    /*
   if((argc < 3)||(argc >4))
   {
      cerr << endl << " Error : correct ussage - \"tsc_client host port [msg]\" !!!";
      exit(1);
   }
   */

    p_host = new Host(getenv("HOSTSRV")); //(argv[1]);
    srv_port = atoi(getenv("COVISE_PORT"));

    cerr << endl << " Connecting to host " << p_host->get_name() << " on port " << srv_port;

    if (myClient.open_conn(p_host, srv_port) == -1)
        cerr << endl << " Error open_conn !!!";

    //sleep(5);

    //cerr << endl << " Sending message !!! ";

    p_msg = new Message;
    p_msg->type = QUIT;
    if (argc == 1)
        p_msg->data = "START CLIENT\n";
    else
        p_msg->data = argv[1];

    p_msg->length = strlen(p_msg->data) + 1;

    myClient.send_msg(p_msg);

    delete p_msg;

    //sleep(3);

    //cerr << endl << "---- Client " << getpid() << " Exitting  !!! ";

    return 0;
}
