#ifndef COVR_MESSAGE_SENDER_H
#define COVR_MESSAGE_SENDER_H
#include <vrb/VrbMessageSenderInterface.h>

namespace opencover
{

    class coVRMessageSender : public vrb::VrbMessageSenderInterface
    {

    private:
        bool sendMessage(const covise::Message *msg) override;
        bool sendMessage(const vrb::UdpMessage *msg) override;
    };
} // namespace opencover

#endif // !COVR_MESSAGE_SENDER_H