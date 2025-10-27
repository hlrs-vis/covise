#include <stdio.h>

#include <config/CoviseConfig.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <cover/input/input.h>

#include "TangibleInterface.h"
#include "VrmlNodeTangible.h"
#include <cover/coVRMSController.h>

using namespace covise;
using namespace opencover;
using namespace vrml;

TangibleInterface::TangibleInterface()
    : coVRPlugin(COVER_PLUGIN_NAME)
{
    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeTangible>());
}

TangibleInterface::~TangibleInterface()
{
    delete udp;
}

bool TangibleInterface::init()
{
    udp=nullptr;

    opencover::Input::instance()->discovery()->deviceAdded.connect(&TangibleInterface::addDevice, this);
    return true;
}

void TangibleInterface::addDevice(const opencover::deviceInfo *i)
{
    
    std::cerr << "TangibleInterface::addDevice called" << std::endl;
    unsigned short localPort = coCoviseConfig::getInt("value", "COVER.Plugin.TangibleInterface.localPort", 31352);
    
    unsigned short serverPort = configInt("TangibleInterface", "serverPort", 31319)->value();
    std::string host = "";
    std::cerr << "Devicename found" << i->deviceName << std::endl;
    if (i->deviceName == "TangibleInterface")
    {
        host = i->address;
        std::cerr << "TangibleInterface config: UDP: TacxHost: " << host << std::endl;

        udp = new UDPComm(host.c_str(), serverPort, localPort);

        if (udp->isBad())
        {
            std::cerr << "TangibleInterface: failed to open UDP port" << host << std::endl;
            delete udp;
            udp=nullptr;
        }
    }

}

bool TangibleInterface::update()
{
    float angle=-100000;
    static float oldAngle=0;

    if (udp)
    {
        int status = udp->receive(&receivedData, sizeof(receivedData),0);

        if (status > 0)
        {
            angle = receivedData.angle;
            std::cerr << "TangibleInterface::update: received angle=" << angle << std::endl;

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
