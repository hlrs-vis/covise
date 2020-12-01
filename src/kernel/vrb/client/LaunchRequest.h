#ifndef VRB_CLIENT_LAUNCH_REQUEST_H
#define VRB_CLIENT_LAUNCH_REQUEST_H


#include <vrb/ProgramType.h>
#include <util/coExport.h>

#include <string>
#include <vector>

namespace covise{
    class TokenBuffer;
}

namespace vrb{
struct VRBCLIENTEXPORT LaunchRequest{
    const Program program;
    const int clientID;
    const std::vector<std::string> args;

    LaunchRequest(covise::TokenBuffer &tb);
    LaunchRequest(Program p, int clID, const std::vector<std::string> &args);

};

VRBCLIENTEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const LaunchRequest &launchRequest);

} // namespace vrb

#endif // !VRB_CLIENT_LAUNCH_REQUEST_H
