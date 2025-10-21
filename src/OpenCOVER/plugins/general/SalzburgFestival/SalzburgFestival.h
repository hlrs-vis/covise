#ifndef SALZBURG_FESTIVAL_H
#define SALZBURG_FESTIVAL_H

#include <cover/coVRPluginSupport.h>

#include <util/UDPComm.h>

struct ReceivedData{
    float angle; // 4 bytes
};

class PLUGINEXPORT SalzburgFestival : public opencover::coVRPlugin
{
private:
    UDPComm *udp = nullptr;

public:
    SalzburgFestival();
    ~SalzburgFestival();

    bool init();
    bool update();

    ReceivedData receivedData;
};

COVERPLUGIN(SalzburgFestival)

#endif // SALZBURG_FESTIVAL_H
