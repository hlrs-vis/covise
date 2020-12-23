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

bool AppModule::sendMessage(const Message *msg)
{
    if (conn)
        return conn->sendMessage(msg);
    else
        return false;
}

bool AppModule::sendMessage(const UdpMessage *msg)
{
    return false;
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
    // Tell CRB to open a socket for the module
    Message* msg = new Message(COVISE_MESSAGE_PREPARE_CONTACT, DataHandle{});
    print_comment(__LINE__, __FILE__, "vor PREPARE_CONTACT send");
    if (dmgr->send(msg) == COVISE_SOCKET_INVALID)
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
        DataHandle msg_data{ 80 };
        memcpy(msg_data.accessData(), &portmsg.data.data()[0], sizeof(int));

        // copy adress of DMGR host in dot notation into message
        strncpy(&msg_data.accessData()[SIZEOF_IEEE_INT], dmgr->host->getAddress(), 76);

        // send to module
        msg_data.setLength(sizeof(int) + (int)strlen(&msg_data.data()[SIZEOF_IEEE_INT]));
        msg = new Message(COVISE_MESSAGE_APP_CONTACT_DM, msg_data);
        print_comment(__LINE__, __FILE__, "vor APP_CONTACT_DM send");
        send(msg);
        delete msg;
    }
    // else  ????????

    return 1;
}

int AppModule::prepare_connect(AppModule *m)
{
    int len = 0;

    Message* msg = new Message(COVISE_MESSAGE_PREPARE_CONTACT, DataHandle{});
    if (m->send(msg) == COVISE_SOCKET_INVALID)
        return 0;
    delete msg;
    return (1);
}

int AppModule::do_connect(AppModule *m, Message *portmsg)
{
    char msg_data[80];
    int len = 0;

    if (portmsg->type == COVISE_MESSAGE_PORT)
    {
        memcpy(msg_data, &portmsg->data.data()[0], sizeof(int));
        strncpy(&msg_data[SIZEOF_IEEE_INT], m->host->getAddress(), 76);
        len = sizeof(int) + (int)strlen(&msg_data[SIZEOF_IEEE_INT]);
        Message msg{ COVISE_MESSAGE_APP_CONTACT_DM , DataHandle{msg_data, len, false} };
        send(&msg);

    }
    return 1;
}

int AppModule::connect_datamanager(AppModule* m)
{
    Message* msg = new Message{ COVISE_MESSAGE_PREPARE_CONTACT_DM, DataHandle{} };
    print_comment(__LINE__, __FILE__, "vor PREPARE_CONTACT_DM send");
    if (m->send(msg) == COVISE_SOCKET_INVALID)
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
            DataHandle msg_data{ 80 };
            memcpy(msg_data.accessData(), &portmsg->data.data()[0], sizeof(int));
            strncpy(&msg_data.accessData()[SIZEOF_IEEE_INT], m->host->getAddress(), 76);
            msg_data.setLength(sizeof(int) + (int)strlen(&msg_data.data()[SIZEOF_IEEE_INT]) + 1);
            msg = new Message{COVISE_MESSAGE_DM_CONTACT_DM, msg_data};
            print_comment(__LINE__, __FILE__, "vor DM_CONTACT_DM send");
            send(msg);
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
    delete portmsg;
    return 1;
}
