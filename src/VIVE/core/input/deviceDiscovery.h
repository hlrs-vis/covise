/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coExport.h>
#include <string>
#include <vector>
#include <util/UDPComm.h>
#include <thread>
#include <mutex>


namespace vive
{


    class VVCORE_EXPORT deviceInfo
    {
    public:
        deviceInfo(const char *buf,std::string addr);
        virtual ~deviceInfo();

        std::string deviceName;
        std::string pluginName;
        std::string address;

    protected:


    private:
    };

    class VVCORE_EXPORT deviceDiscovery 
    {
        friend class vvVIVE;

    public:
        deviceDiscovery();
        ~deviceDiscovery();
        void init();
        void start(); //<start thread
        void update(); //< called by Input::update()
        void run(); //regularly check for new devices

        // only to be used from main thread
        const std::vector<const deviceInfo *> &getDevices() const;

    private:
        std::vector<const deviceInfo *> devices;
        std::vector<deviceInfo*> toAdd;

        bool running = true;
        std::string broadcastAddress;
        int port;
        UDPComm * dComm=nullptr;
        std::thread *myThread;
        std::mutex mutex;
    };

}
