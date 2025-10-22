#include <stdio.h>

#include <config/CoviseConfig.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <cover/input/input.h>

#include "SalzburgFestival.h"
#include "VrmlNodeTangible.h"
#include <cover/coVRMSController.h>

using namespace covise;
using namespace opencover;
using namespace vrml;

SalzburgFestival::SalzburgFestival()
    : coVRPlugin(COVER_PLUGIN_NAME)
{
    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeTangible>());
}

SalzburgFestival::~SalzburgFestival()
{
    delete udp;
}

bool SalzburgFestival::init()
{
    udp=nullptr;

    opencover::Input::instance()->discovery()->deviceAdded.connect(&SalzburgFestival::addDevice, this);
    return true;
}

void SalzburgFestival::addDevice(const opencover::deviceInfo *i)
{
    
    std::cerr << "SalzburgFestival::addDevice called" << std::endl;
    unsigned short localPort = coCoviseConfig::getInt("value", "COVER.Plugin.SalzburgFestival.localPort", 31352);
    
    unsigned short serverPort = configInt("TacxFTMS", "serverPort", 31319)->value();
    std::string host = "";
    std::cerr << "Devicename found" << i->deviceName << std::endl;
    if (i->deviceName == "TangibleInterface")
    {
        host = i->address;
        std::cerr << "SalzburgFestival config: UDP: TacxHost: " << host << std::endl;

        udp = new UDPComm(host.c_str(), serverPort, localPort);

        if (udp->isBad())
        {
            std::cerr << "SalzburgFestival: failed to open UDP port" << host << std::endl;
            delete udp;
            udp=nullptr;
        }
    }

}

bool SalzburgFestival::update()
{
    float angle=-100000;
    static float oldAngle=0;

    if (udp)
    {
        int status = udp->receive(&receivedData, sizeof(receivedData),0);

        if (status > 0)
        {
            angle = receivedData.angle;
            std::cerr << "SalzburgFestival::update: received angle=" << angle << std::endl;

        }
    }
    coVRMSController::instance()->syncData(&angle,sizeof(angle));
    

    if(angle != -100000 && angle !=oldAngle)
    {
        oldAngle=angle;
        for (auto node : VrmlNodeTangible::getAllNodeTangibles())
        {
            node->setAngle(angle);
        }
        return true;
    }
    return false;
}
