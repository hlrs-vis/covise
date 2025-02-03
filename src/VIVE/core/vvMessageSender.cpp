#include "vvMessageSender.h"
#include "vvPluginSupport.h"

#include <net/message.h>
#include <net/udpMessage.h>

using namespace vive;

bool vvMessageSender::sendMessage(const covise::Message *msg) const{
    return vv->sendVrbMessage(msg);
}

bool vvMessageSender::sendMessage(const covise::UdpMessage *msg) const{
    return vv->sendVrbMessage(msg);
}

