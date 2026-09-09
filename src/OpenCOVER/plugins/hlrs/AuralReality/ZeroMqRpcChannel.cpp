#include "ZeroMqRpcChannel.h"
#include "arrpc/message_header.pb.h"
#include <string>

namespace ar = auralreality;
using std::cout, std::cerr, std::endl;

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
        cerr << "Failed to serialize header" << endl;
        promise.failure(rpc::RpcStatus::error(rpc::RpcStatus::Code::InternalError, "failed to serialize message header"));
        return promise.future();
    }

    std::array<zmq::const_buffer, 2> send_msgs = {
        zmq::buffer(header_string),
        zmq::buffer(data)
    };

    if (!zmq::send_multipart(socket_, send_msgs))
    {
        cerr << "Failed to send zmq message" << endl;
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
        zmq::message_t message;

        if (!socket_.recv(message, zmq::recv_flags::dontwait))
        {
            // No error, there is just no message here (`dontwait`).
            return;
        }

        ar::MessageHeader header;
        if (!header.ParseFromArray(message.data(), message.size()))
        {
            cerr << "Failed to parse header, skipping message" << endl;
            continue;
        }

        auto status_code = header.status_code();
        auto request_id = header.request_id();

        bool expect_more = status_code == ar::StatusCode::Ok;

        if (expect_more && !message.more())
        {
            cerr << "ZeroMqRpcChannel: expected more after success header." << endl;
            continue;
        }

        // Consume all message parts, only the last one will stay in `message`
        while (message.more())
        {
            if (!socket_.recv(message))
            {
                cerr << "ZeroMqRpcChannel: failed to read multipart content frame" << endl;
                return;
            }
        }

        auto it = message_promises_.find(request_id);
        if (it == message_promises_.end())
        {
            cerr << "ZeroMqRpcChannel: got response to message " << request_id << ", but this message is not known" << endl;
            continue;
        }

        if (status_code == ar::StatusCode::Ok)
        {
            it->second.success(message.to_string());
        }
        else
        {
            std::string text = status_code >= 0 && status_code <= 7 ? STATUS_CODE_TEXTS[status_code] : "Unknown";
            std::string message = std::string("Error response from server: ") + text + " (" + std::to_string((int)status_code) + ")";
            if (header.has_error_message())
            {
                message += ": ";
                message += header.error_message();
            }

            it->second.failure(rpc::RpcStatus::error(status_code, message));
        }

        message_promises_.erase(request_id);
    }
}
