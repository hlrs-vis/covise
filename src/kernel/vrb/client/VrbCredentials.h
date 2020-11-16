#ifndef VRB_CREDENTIALS_H
#define VRB_CREDENTIALS_H
#include <util/coExport.h>
#include <string>
namespace vrb{
struct VRBCLIENTEXPORT VrbCredentials{
    VrbCredentials(const std::string &ipAddress, unsigned int tcpPort, unsigned int udpPort);
    VrbCredentials();
    const std::string ipAddress;
    const unsigned int tcpPort;
    const unsigned int udpPort;
};
} // namespace vrb

#endif // !VRB_CREDENTIALS_H