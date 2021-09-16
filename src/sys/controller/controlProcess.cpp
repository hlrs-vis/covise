/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <process.h>
#else
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <net/route.h>
#include <arpa/inet.h>
#endif

#include <covise/covise.h>
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <util/unixcompat.h>
#include <messages/CRB_EXEC.h>

#include "handler.h"
#include "global.h"
#include "controlProcess.h"
#include "config.h"

#define STDIN_MAPPING

using namespace covise;
using namespace covise::controller;

void Controller::get_shared_memory(const covise::Connection *conn)
{
    Message msg{ COVISE_MESSAGE_GET_SHM_KEY , DataHandle{} };

    print_comment(__LINE__, __FILE__, "in get_shared_memory");
    conn->sendMessage(&msg);
    conn->recv_msg(&msg);
    if (msg.type == COVISE_MESSAGE_GET_SHM_KEY)
    {
        print_comment(__LINE__, __FILE__, "GET_SHM_KEY: %d: %x, %d length: %d", *(int *)msg.data.data(),
                      ((int *)msg.data.data())[1], ((int *)msg.data.data())[2], msg.data.length());
        shm = new ShmAccess(msg.data.accessData(), 0);
    }
    // data of received message can be deleted
}

void Controller::handle_shm_msg(Message *msg)
{
    int tmpkey, size;

    if (msg->conn->get_sender_id() == 1)
    {
        tmpkey = ((int *)msg->data.data())[0];
        size = ((int *)msg->data.data())[1];
        shm->add_new_segment(tmpkey, size);
        print_comment(__LINE__, __FILE__, "new SharedMemory");
    }
}
