#include "LaunchRequest.h"
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/message_sender_interface.h>

using namespace vrb;

Program getProgram(covise::TokenBuffer &tb){
    Program p;
    tb >> p;
    return p;
}
int getID(covise::TokenBuffer& tb){
    int id;
    tb >> id;
    return id;
}

std::vector<std::string> getArgs(covise::TokenBuffer& tb){
    std::vector<std::string> args;
    deserialize(tb, args);
    return args;
}

LaunchRequest::LaunchRequest(covise::TokenBuffer &tb)
    :program(getProgram(tb))
    ,clientID(getID(tb))
    ,args(getArgs(tb)){}

LaunchRequest::LaunchRequest(Program p, int clID, const std::vector<std::string> &args)
    :program(p)
    ,clientID(clID)
    ,args(args){}

covise::TokenBuffer &vrb::operator<<(covise::TokenBuffer &tb, const LaunchRequest &launchRequest){
    tb << launchRequest.program << launchRequest.clientID;
    serialize(tb, launchRequest.args);
    return tb;
}
bool vrb::sendLaunchRequestToRemoteLaunchers(const LaunchRequest &lrq, covise::MessageSenderInterface *sender){
    covise::Message msg;
    covise::TokenBuffer outerTb, innerTb;
    innerTb << lrq;
    outerTb << vrb::Program::VrbRemoteLauncher << covise::COVISE_MESSAGE_VRB_MESSAGE << innerTb;
    return sender->send(outerTb, covise::COVISE_MESSAGE_BROADCAST_TO_PROGRAM);
}