/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_ZEROMQRPCCHANNEL_H
#define _AURAL_REALITY_ZEROMQRPCCHANNEL_H

#include <string>

#include <zmq.hpp>
#include <zmq_addon.hpp>

#include <arrpc/arrpc_channel.h>
#include <arrpc/tmt_service.rpc.h>

class ZeroMqRpcChannel : public ::rpc::RpcChannel
{
public:
    ZeroMqRpcChannel(zmq::socket_t &socket);

    void poll();

private:
    virtual rpc::RpcFuture<std::string> perform_request(rpc::RpcId rpc_id, std::string_view data) override;

    zmq::socket_t &socket_;

    int request_id_ = 0;

    std::map<rpc::RpcId, rpc::RpcPromise<std::string>> message_promises_;
};

#endif
