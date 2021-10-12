/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coverDaemon.h"
#include <boost/algorithm/string.hpp>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <util/coSpawnProgram.h>
#include <thread>
#include <chrono>
#include <vrb/client/LaunchRequest.h>
using namespace covise;

bool CoverDaemon::startup()
{
    m_serverConn = m_connections.tryAddNewListeningConn<covise::SimpleServerConnection>(m_port, 0, 0);
    if (!m_serverConn)
    {
        std::cerr << "A COVER Daemon is already listening for connections on this host on port " << m_port << ", disabling the function to start OpenCOVER slaves!" << std::endl;
        return false;
    }

    m_sn.reset(new QSocketNotifier(m_serverConn->get_id(NULL), QSocketNotifier::Read));

    connect(m_sn.get(), &QSocketNotifier::activated, this, &CoverDaemon::handleConnections);
    std::cerr << "waiting for slaves to connect on port " << m_port << std::endl;
    return true;
}

void CoverDaemon::handleConnections()
{
    const auto *conn = m_connections.check_for_input(0.0001f);
    std::unique_ptr<covise::SimpleServerConnection> clientConn;
    if (conn == m_serverConn) //tcp connection to server port
    {
        clientConn = m_serverConn->spawnSimpleConnection();
    }
    if (clientConn)
    {
        Message msg;
        clientConn->recv_msg(&msg);

        if (msg.type == covise::COVISE_MESSAGE_VRB_MESSAGE)
        {
            vrb::VRB_MESSAGE lrq{msg};
            for(const auto &env : lrq.environment())
            {
                auto delim = env.find_first_of("=");
                auto var = env.substr(0, delim);
                auto val = env.substr(delim + 1);
                if (auto localVal = getenv(var.c_str()))
                {
                   if (localVal != val)
                   {
                       std::cerr << "Warning: COVER host is using different environment for " << var << ":" << std::endl
                                 << "host uses " << val << " local is " << localVal << std::endl
                                 << "The cover slave will be forced to use " << val << std::endl;
                   }
                }
                
            }

            spawnProgram(covise::programNames[lrq.program()], lrq.args(), lrq.environment());
        }
    }
}

CoverDaemon::CoverDaemon()
{
    bool exists = false;
    m_port = coCoviseConfig::getInt("port", "COVER.Daemon", 31090, &exists);
    if (!exists)
        return;
    startup();
}
