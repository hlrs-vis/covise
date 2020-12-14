#include "LaunchRequest.h"
#include <net/message.h>
#include <net/message_sender_interface.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>

#include <cassert>
#include <iostream>

namespace vrb{

IMPL_MESSAGE_CLASS(VRB_MESSAGE, Program, program, int, clientID, std::vector<std::string>, args)

IMPL_MESSAGE_WITH_SUB_CLASSES(VRB_LOAD_SESSION, VrbMessageType)
IMPL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Launcher, Program, program, int, clientID, std::vector<std::string>, args)

bool sendLaunchRequestToRemoteLaunchers(const VRB_MESSAGE &lrq, covise::MessageSenderInterface *sender){
    covise::TokenBuffer outerTb, innerTb;
    innerTb << lrq;
    outerTb << Program::VrbRemoteLauncher << covise::COVISE_MESSAGE_VRB_MESSAGE << innerTb;
    return sender->send(outerTb, covise::COVISE_MESSAGE_BROADCAST_TO_PROGRAM);
}

}//vrb


