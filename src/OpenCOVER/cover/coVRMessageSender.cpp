#include "coVRMessageSender.h"
#include "coVRPluginSupport.h"

#include <net/message.h>
#include <net/udpMessage.h>

using namespace opencover;

bool coVRMessageSender::sendMessage(const covise::Message *msg){
    return cover->sendVrbMessage(msg);
}

bool coVRMessageSender::sendMessage(const covise::UdpMessage *msg){
    return cover->sendVrbMessage(msg);
}

