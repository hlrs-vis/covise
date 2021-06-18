#ifndef VRB_CLIENT_LAUNCH_REQUEST_H
#define VRB_CLIENT_LAUNCH_REQUEST_H


#include <vrb/ProgramType.h>
#include <util/coExport.h>

#include <string>
#include <vector>
#include <net/message_macros.h>


namespace vrb{

DECL_MESSAGE_CLASS(VRB_MESSAGE, VRBCLIENTEXPORT, int, senderID, Program, program, int, clientID, std::vector<std::string>, environment, std::vector<std::string>, args)
VRBCLIENTEXPORT bool sendLaunchRequestToRemoteLaunchers(const VRB_MESSAGE &lrq, const covise::MessageSenderInterface *sender);


} // namespace vrb

#endif // !VRB_CLIENT_LAUNCH_REQUEST_H
