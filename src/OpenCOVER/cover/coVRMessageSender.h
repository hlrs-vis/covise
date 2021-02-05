#ifndef COVR_MESSAGE_SENDER_H
#define COVR_MESSAGE_SENDER_H
#include <net/message_sender_interface.h>

namespace opencover
{

    class coVRMessageSender : public covise::MessageSenderInterface
    {

    private:
        bool sendMessage(const covise::Message *msg) const override;
        bool sendMessage(const covise::UdpMessage *msg) const override;
    };
} // namespace opencover

#endif // !COVR_MESSAGE_SENDER_H