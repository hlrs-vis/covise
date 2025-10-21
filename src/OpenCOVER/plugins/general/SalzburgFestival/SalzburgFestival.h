#ifndef SALZBURG_FESTIVAL_H
#define SALZBURG_FESTIVAL_H

#include <cover/coVRPluginSupport.h>

#include <util/UDPComm.h>

class PLUGINEXPORT SalzburgFestival : public opencover::coVRPlugin
{
private:
    UDPComm *udp = nullptr;

public:
    SalzburgFestival();
    ~SalzburgFestival();

    bool init();
};

COVERPLUGIN(SalzburgFestival)

#endif // SALZBURG_FESTIVAL_H
