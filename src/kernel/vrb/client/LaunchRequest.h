#ifndef VRB_CLIENT_LAUNCH_REQUEST_H
#define VRB_CLIENT_LAUNCH_REQUEST_H


#include <vrb/ProgramType.h>
#include <util/coExport.h>

#include <string>
#include <vector>
#include <net/message_macros.h>


namespace vrb{

DECL_MESSAGE_CLASS(VRB_MESSAGE, VRBCLIENTEXPORT, Program, program, int, clientID, std::vector<std::string>, args)
VRBCLIENTEXPORT bool sendLaunchRequestToRemoteLaunchers(const VRB_MESSAGE &lrq, const covise::MessageSenderInterface *sender);

    enum class VrbMessageType
    {
        Launcher,
        Avatar
    };
    DECL_MESSAGE_WITH_SUB_CLASSES(VRB_LOAD_SESSION, VrbMessageType, VRBCLIENTEXPORT)
    DECL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Launcher, VRBCLIENTEXPORT, Program, program, int, clientID, std::vector<std::string>, args)

    //VRB_MESSAGE_Launcher l{vrb::Program::Cover, 5, std::vector<std::string>{}};

} // namespace vrb

#endif // !VRB_CLIENT_LAUNCH_REQUEST_H
