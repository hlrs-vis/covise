/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "handler.h"
#include "host.h"
#include "util.h"
#include "module.h"
#include "exception.h"
#include "global.h"

#include <comsg/CRB_EXEC.h>
#include <covise/covise.h>
#include <covise/covise_msg.h>
#include <covise/covise_process.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

const int SIZEOF_IEEE_INT = 4;

using namespace covise;
using namespace covise::controller;

size_t SubProcess::processCount = 1001;

SubProcess::SubProcess(Type t, const RemoteHost &h, sender_type type, const std::string &executableName)
    : host(h), type(type), processId(processCount++), m_type(t), m_executableName(executableName)
{
}

SubProcess::SubProcess(SubProcess &&other)
    : host(other.host), type(other.type), processId(other.processId), m_type(other.m_type), m_executableName(std::move(other.m_executableName)), m_conn(other.m_conn)
{
    other.m_conn = nullptr;
}

SubProcess::~SubProcess()
{
    CTRLGlobal::getInstance()->controller->getConnectionList()->remove(m_conn);
}

const std::string &SubProcess::getHost() const
{
    return host.userInfo().ipAdress;
}

const Connection *SubProcess::conn() const
{
    return &*m_conn;
}

void SubProcess::recv_msg(Message *msg) const
{
    if (m_conn)
    {
        m_conn->recv_msg(msg);
    }
};

bool SubProcess::sendMessage(const Message *msg) const
{
    if (m_conn)
        return m_conn->sendMessage(msg);
    return false;
}

bool SubProcess::sendMessage(const UdpMessage *msg) const
{
    return false;
}

bool SubProcess::connectModuleWithCrb()
{
    return connect(host.getProcess(sender_type::CRB), ConnectionType::ModuleToCrb);
}

bool SubProcess::connect(const SubProcess &crb, ConnectionType type)
{
    constexpr std::array<covise::covise_msg_type, 2> prepareMessages{COVISE_MESSAGE_PREPARE_CONTACT, COVISE_MESSAGE_PREPARE_CONTACT_DM};
    constexpr std::array<covise::covise_msg_type, 2> contactMessages{COVISE_MESSAGE_APP_CONTACT_DM, COVISE_MESSAGE_DM_CONTACT_DM};

    if (&crb == this)
    {
        std::cerr << "can not connect module to itself" << std::endl;
        return false;
    }
    // Tell CRB to open a socket for the module
    Message msg{prepareMessages[type], DataHandle{}};
    print_comment(__LINE__, __FILE__, "before PREPARE_CONTACT send");
    if (!crb.send(&msg))
        return false;
    //std::cerr << "sent " << covise_msg_types_array[prepareMessages[type]] << " to crb on " << crb.host.userInfo().ipAdress << " on port " << crb.conn()->get_port() << std::endl;
    // Wait for CRB to deliver the opened port number
    do
    {
        std::unique_ptr<Message> portmsg{new Message{}};
        crb.recv_msg(&*portmsg);
        print_comment(__LINE__, __FILE__, "PORT received");
        if (portmsg->type == COVISE_MESSAGE_PORT)
        {
            // copy port number to message buffer:
            // byteswap is done by sender
            DataHandle msg_data{80};
            memcpy(msg_data.accessData(), &portmsg->data.data()[0], sizeof(int));

            // copy adress of CRB host in dot notation into message
            strncpy(&msg_data.accessData()[sizeof(int)], crb.host.userInfo().ipAdress.c_str(), 76);

            // send to module

            msg_data.setLength(sizeof(int) + (int)strlen(&msg_data.data()[sizeof(int)]) + 1);
            msg = Message{contactMessages[type], msg_data};
            print_comment(__LINE__, __FILE__, "vor APP_CONTACT_DM send");
            send(&msg);
            //std::cerr << "sent " << covise_msg_types_array[contactMessages[type]] << " to module on " << host.userInfo().ipAdress << " on port " << conn()->get_port() << std::endl;

            return true;
        }
        else
        {
            CTRLHandler::instance()->handleMsg(portmsg); // handle all other messages
        }

    } while (true);
}

bool SubProcess::setupConn(std::function<bool(int)> sendConnMessage)
{
    int port = 0;
    auto conn = createListeningConn<ServerConnection>(&port, processId, (int)CONTROLLER);
    m_conn = conn.get(); //already set this here to shutdown accecpt if launch is rejected
    if (sendConnMessage(port))
    {
        int timeout = 0; // do not timeout
        if (conn->acceptOne(timeout) < 0)
        {
            cerr << "* timelimit in accept for module " << m_executableName << " exceeded!!" << endl;
            return false;
        }
        conn->set_peer(processId, type);
        m_conn = CTRLGlobal::getInstance()->controller->getConnectionList()->add(std::move(conn));
        CTRLGlobal::getInstance()->controller->getConnectionList()->addRemoveNotice(m_conn, [this]() { m_conn = nullptr; });
        return true;
    }
    return false;
}

bool SubProcess::start(const char *instance, const char *category)
{
    return setupConn([this, instance, category](int port) {
        auto &controllerHost = host.hostManager.getLocalHost();
        CRB_EXEC crbExec{covise::ExecFlag::Normal, m_executableName.c_str(), port, controllerHost.userInfo().ipAdress.c_str(), static_cast<int>(processId), instance,
                         host.userInfo().ipAdress.c_str(), host.userInfo().hostName.c_str(),
                         category, host.hostManager.getLocalHost().ID(), vrb::VrbCredentials{}, std::vector<std::string>{}};

        host.launchProcess(crbExec);
        return true;
    });
}
