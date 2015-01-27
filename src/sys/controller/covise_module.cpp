/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <covise/covise_process.h>
#include <covise/covise_msg.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>

const int SIZEOF_IEEE_INT = 4;

#include "CTRLHandler.h"
#include "covise_module.h"

using namespace covise;

int AppModule::send_msg(Message *msg)
{
    if (NULL != conn)
        return conn->send_msg(msg);

    else
        return 0;
}

int AppModule::recv_msg(Message *msg)
{
    if (conn)
        return conn->recv_msg(msg);

    else
        return 0;
}

void AppModule::set_peer(int peer_id, sender_type peer_type)
{
    conn->set_peer(peer_id, peer_type);
    id = peer_id;
}

// CTRL calls this to connect a AppModule to its CRB
int AppModule::connect(AppModule *dmgr)
{
    char msg_data[80];
    int len = 0;

    // Tell CRB to open a socket for the module
    Message *msg = new Message(COVISE_MESSAGE_PREPARE_CONTACT, len, NULL);
    print_comment(__LINE__, __FILE__, "vor PREPARE_CONTACT send");
    if (dmgr->send_msg(msg) == COVISE_SOCKET_INVALID)
        return 0;
    delete msg;

    // Wait for CRB to deliver the opened port number
    Message portmsg;
    dmgr->recv_msg(&portmsg);
    print_comment(__LINE__, __FILE__, "PORT received");
    if (portmsg.type == COVISE_MESSAGE_PORT)
    {
        // copy port number to message buffer:
        //   can be binary, since CRB and Module are on same machine
        memcpy(msg_data, &portmsg.data[0], sizeof(int));

        // copy adress of DMGR host in dot notation into message
        strncpy(&msg_data[SIZEOF_IEEE_INT], dmgr->host->getAddress(), 76);

        // send to module
        len = sizeof(int) + (int)strlen(&msg_data[SIZEOF_IEEE_INT]);
        msg = new Message(COVISE_MESSAGE_APP_CONTACT_DM, len, msg_data, MSG_NOCOPY);
        print_comment(__LINE__, __FILE__, "vor APP_CONTACT_DM send");
        send_msg(msg);
        msg->data = NULL;
        delete msg;
    }
    // else  ????????

    portmsg.delete_data();
    return 1;
}

int AppModule::prepare_connect(AppModule *m)
{
    int len = 0;

    Message *msg = new Message(COVISE_MESSAGE_PREPARE_CONTACT, len, (char *)NULL);
    if (m->send_msg(msg) == COVISE_SOCKET_INVALID)
        return 0;
    delete msg;
    return (1);
}

int AppModule::do_connect(AppModule *m, Message *portmsg)
{
    Message *msg;
    char msg_data[80];
    int len = 0;

    if (portmsg->type == COVISE_MESSAGE_PORT)
    {
        memcpy(msg_data, &portmsg->data[0], sizeof(int));
        strncpy(&msg_data[SIZEOF_IEEE_INT], m->host->getAddress(), 76);
        len = sizeof(int) + (int)strlen(&msg_data[SIZEOF_IEEE_INT]);
        msg = new Message(COVISE_MESSAGE_APP_CONTACT_DM, len, msg_data, MSG_NOCOPY);
        send_msg(msg);
        msg->data = NULL;
        delete msg;
    }
    portmsg->delete_data();
    return 1;
}

int AppModule::connect_datamanager(AppModule *m)
{
    char msg_data[80];
    int len = 0;

    Message *msg = new Message(COVISE_MESSAGE_PREPARE_CONTACT_DM, len, (char *)NULL);
    print_comment(__LINE__, __FILE__, "vor PREPARE_CONTACT_DM send");
    if (m->send_msg(msg) == COVISE_SOCKET_INVALID)
        return 0;
    delete msg;

    Message *portmsg = new Message;
    do
    {
        m->recv_msg(portmsg);
        //fprintf(stderr, "connect datamanager\n");
        print_comment(__LINE__, __FILE__, "PORT received");
        if (portmsg->type == COVISE_MESSAGE_PORT)
        {
            memcpy(msg_data, &portmsg->data[0], sizeof(int));
            strncpy(&msg_data[SIZEOF_IEEE_INT], m->host->getAddress(), 76);
            len = sizeof(int) + (int)strlen(&msg_data[SIZEOF_IEEE_INT]) + 1;
            msg = new Message(COVISE_MESSAGE_DM_CONTACT_DM, len, msg_data, MSG_NOCOPY);
            print_comment(__LINE__, __FILE__, "vor DM_CONTACT_DM send");
            send_msg(msg);
            msg->data = NULL;
            delete msg;
            break;
        }
        else
        {
            fprintf(stderr, "handle other message\n");
            CTRLHandler::instance()->handleAndDeleteMsg(portmsg); // handle all other messages
            portmsg = new Message;
        }
    } while (1);
    portmsg->delete_data();
    portmsg->data = NULL;
    delete portmsg;
    return 1;
}
