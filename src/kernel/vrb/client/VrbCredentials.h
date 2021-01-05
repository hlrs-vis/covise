#ifndef VRB_CREDENTIALS_H
#define VRB_CREDENTIALS_H
#include <util/coExport.h>
#include <net/tokenbuffer_util.h>
#include <string>
namespace covise
{
    class TokenBuffer;
}
namespace vrb
{
struct VRBCLIENTEXPORT VrbCredentials
{
    VrbCredentials(const std::string &ipAddress, unsigned int tcpPort, unsigned int udpPort);
    VrbCredentials();
    const std::string ipAddress;
    const unsigned int tcpPort;
    const unsigned int udpPort;
};
VRBCLIENTEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &, const VrbCredentials &cred);
VRBCLIENTEXPORT bool operator==(const VrbCredentials &cred1, const VrbCredentials &cred2);

} // namespace vrb

namespace covise{
namespace detail{

template<>
inline vrb::VrbCredentials get(covise::TokenBuffer &tb){
    std::string ip;
    int tcp = 0, udp = 0;
    tb >> ip >> tcp >> udp;
    return vrb::VrbCredentials{ip, static_cast<unsigned int>(tcp), static_cast<unsigned int>(udp)};
}
}
}

#endif // !VRB_CREDENTIALS_H