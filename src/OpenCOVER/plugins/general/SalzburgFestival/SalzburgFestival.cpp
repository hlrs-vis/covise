#include <config/CoviseConfig.h>

#include "SalzburgFestival.h"

using namespace covise;
using namespace opencover;

SalzburgFestival::SalzburgFestival()
    : coVRPlugin(COVER_PLUGIN_NAME) {}

SalzburgFestival::~SalzburgFestival()
{
    delete udp;
}

bool SalzburgFestival::init()
{
    delete udp;

    const std::string host = coCoviseConfig::getEntry("value", "SalzburgFestival.serverHost", "localhost");
    unsigned short serverPort = coCoviseConfig::getInt("SalzburgFestival.serverPort", 31350);
    unsigned short localPort = coCoviseConfig::getInt("SalzburgFestival.localPort", 31352);
    std::cerr << "SalzburgFestival config: UDP: serverHost: " << host << ", serverPort: " << serverPort << ", localPort: " << localPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);

    if (udp->isBad())
    {
        std::cerr << "SalzburgFestival: failed to open UDP port" << localPort << std::endl;
        return false;
    }

    return true;
}
