#include "LaunchRequest.h"
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>

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