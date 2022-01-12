#ifndef COVISE_VRB_CLIENT_INTERFACE_H
#define COVISE_VRB_CLIENT_INTERFACE_H

#include "export.h"
#include <string>
#include <vector>

namespace vrbclient
{
namespace detail{
    class VrbClientInternals;
}
class VRBClientInterfaceEXPORT VrbClient
{
public:
    VrbClient(const std::string &ip, unsigned int port);
    VrbClient(const VrbClient &) = delete;
    VrbClient(VrbClient &&) = default;
    VrbClient &operator=(const VrbClient &) = delete;
    VrbClient &operator=(VrbClient &&) = default;
    ~VrbClient();

    bool isConnected() const;

    std::vector<std::string> sessions() const; //list of available sessions
    void joinSession(const std::string &name); //join or create session with name
    std::vector<std::string> partner() const;  //list of partners in current session

    std::vector<std::string> daemons() const; //get a list of available daemons
    void addPartner(const std::string &name); //add a partner to the current session by starting this applicatiopn through the daemon
private:
    detail::VrbClientInternals *internals;
};




}

#endif