#ifndef COVR_MESSAGE_SENDER_H
#define COVR_MESSAGE_SENDER_H
#include <net/message_sender_interface.h>

namespace vive
{

    class VVCORE_EXPORT vvMessageSender : public covise::MessageSenderInterface
    {

    private:
        bool sendMessage(const covise::Message *msg) const override;
        bool sendMessage(const covise::UdpMessage *msg) const override;
    };
} // namespace vive

#endif // !COVR_MESSAGE_SENDER_H