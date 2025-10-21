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

    const std::string host = coCoviseConfig::getEntry("value", "COVER.Plugin.SalzburgFestival.serverHost", "localhost");
    unsigned short serverPort = coCoviseConfig::getInt("value", "COVER.Plugin.SalzburgFestival.serverPort", 31350);
    unsigned short localPort = coCoviseConfig::getInt("value", "COVER.Plugin.SalzburgFestival.localPort", 31352);
    std::cerr << "SalzburgFestival config: UDP: serverHost: " << host << ", serverPort: " << serverPort << ", localPort: " << localPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);

    if (udp->isBad())
    {
        std::cerr << "SalzburgFestival: failed to open UDP port" << localPort << std::endl;
        return false;
    }

    return true;
}

bool SalzburgFestival::update()
{
    if (udp)
    {
        char tmpBuf[10000];
        int status = udp->receive(&tmpBuf, 10000);

        if (status > 0)
        {
            std::string msg(tmpBuf, status);
            std::cerr << "SalzburgFestival: received message: " << msg << std::endl;
        }
        else if (status == -1)
        {
            std::cerr << "SalzburgFestival::update: error while reading data" << std::endl;
            return false;
        }
        else
        {
            std::cerr << "SalzburgFestival::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
            return false;
        }

        udp->send("test");
    }
    return true;
}
