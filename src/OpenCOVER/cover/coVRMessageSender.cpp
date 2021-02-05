#include "coVRMessageSender.h"
#include "coVRPluginSupport.h"

#include <net/message.h>
#include <net/udpMessage.h>

using namespace opencover;

bool coVRMessageSender::sendMessage(const covise::Message *msg) const{
    return cover->sendVrbMessage(msg);
}

bool coVRMessageSender::sendMessage(const covise::UdpMessage *msg) const{
    return cover->sendVrbMessage(msg);
}

