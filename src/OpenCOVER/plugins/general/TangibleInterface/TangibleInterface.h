#ifndef SALZBURG_FESTIVAL_H
#define SALZBURG_FESTIVAL_H

#include <cover/coVRPluginSupport.h>

#include <util/UDPComm.h>
#include <cover/input/deviceDiscovery.h>

struct ReceivedData
{
    float angle; // 4 bytes
};

class PLUGINEXPORT TangibleInterface : public opencover::coVRPlugin
{
private:
    UDPComm *udp = nullptr;

public:
    TangibleInterface();
    ~TangibleInterface();

    bool init();
    bool update();
    void addDevice(const opencover::deviceInfo *dev);

    ReceivedData receivedData;
};

COVERPLUGIN(TangibleInterface)

#endif // SALZBURG_FESTIVAL_H
