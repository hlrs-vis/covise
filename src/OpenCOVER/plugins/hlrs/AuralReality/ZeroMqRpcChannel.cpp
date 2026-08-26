#include "ZeroMqRpcChannel.h"

namespace ar = auralreality;

ZeroMqRpcChannel::ZeroMqRpcChannel(zmq::socket_t &socket)
    : socket_(socket)
{
}

rpc::RpcFuture<std::string> ZeroMqRpcChannel::perform_request(rpc::RpcId rpc_id, std::string_view data)
{
    using Promise = rpc::RpcPromise<std::string>;

    Promise promise;

    auto request_id = ++request_id_;

    ar::MessageHeader header;
    header.set_type(ar::MessageType::Request);
    header.set_rpc_id(rpc_id);
    header.set_request_id(request_id);
    std::string header_string;
    if (!header.SerializeToString(&header_string))
    {
        std::cerr << "Failed to serialize header" << std::endl;
        promise.failure(rpc::RpcStatus::error(rpc::RpcStatus::Code::InternalError, "failed to serialize message header"));
        return promise.future();
    }

    std::array<zmq::const_buffer, 2> send_msgs = {
        zmq::buffer(header_string),
        zmq::buffer(data)
    };

    if (!zmq::send_multipart(socket_, send_msgs))
    {
        std::cerr << "Failed to send zmq message" << std::endl;
        promise.failure(rpc::RpcStatus::error(rpc::RpcStatus::Code::InternalError, "failed to send zmq message"));
    }
    else
    {
        message_promises_.emplace(request_id, promise);
    }

    return promise.future();
}

static std::string STATUS_CODE_TEXTS[] = {
    "Ok",
    "Unknown method",
    "Invalid argument",
    "Permission denied",
    "Not found",
    "Already exists",
    "Internal error",
    "Serialization error"
};

void ZeroMqRpcChannel::poll()
{
    while (true)
    {
        zmq::message_t header_message;
        if (!socket_.recv(header_message, zmq::recv_flags::dontwait))
        {
            return;
        }
        zmq::message_t content_message;
        if (!socket_.recv(content_message))
        {
            return;
        }

        ar::MessageHeader header;
        if (!header.ParseFromArray(header_message.data(), header_message.size()))
        {
            std::cerr << "Failed to parse header" << std::endl;
            continue;
        }
        auto rpc_id = header.rpc_id();

        auto it = message_promises_.find(rpc_id);
        if (it == message_promises_.end())
        {
            continue;
        }

        auto status_code = header.status_code();
        if (status_code == ar::StatusCode::Ok)
        {
            it->second.success(content_message.to_string());
        }
        else
        {
            std::string text = status_code >= 0 && status_code <= 7 ? STATUS_CODE_TEXTS[status_code] : "Unknown";
            it->second.failure(rpc::RpcStatus::error(status_code,
                std::string("Error response from server: ") + text + " (" + std::to_string((int)status_code) + ")"));
        }

        message_promises_.erase(rpc_id);
    }
}
