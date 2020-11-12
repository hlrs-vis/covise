#ifndef VRB_CREDENTIALS_H
#define VRB_CREDENTIALS_H
#include <util/coExport.h>
#include <string>
namespace vrb{
struct VRBCLIENTEXPORT VrbCredentials{
    VrbCredentials(const std::string &ipAddress, int tcpPort, int udpPort);
    VrbCredentials();
    const std::string ipAddress;
    const int tcpPort;
    const int udpPort;
};
} // namespace vrb

#endif // !VRB_CREDENTIALS_H