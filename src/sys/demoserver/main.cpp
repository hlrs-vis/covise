#include <demo.h>
#include <memory>
#include <thread>
#include <iostream>

#include <net/covise_host.h>

#include "demoserver.h"

int main (int argc, char *argv[])
{
    if(demo::root.empty()) {
        std::cerr << "empty root directory - refusing to serve entire filesystem" << std::endl;
        return 1;
    }

    std::cerr << "Starting demo server: http://" << covise::Host::getHostaddress() << ":" << demo::port << std::endl;
    DemoServer demoServer;
    demoServer.run();
    return 0;
}
