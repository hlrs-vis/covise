/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputsource.h
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifndef DEVICEDISCOVERY_H
#define DEVICEDISCOVERY_H

#include <util/coExport.h>
#include <string>
#include <vector>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <util/UDPComm.h>

#include <sigslot/signal.hpp>


namespace opencover
{


    class COVEREXPORT deviceInfo
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

    class COVEREXPORT deviceDiscovery : public OpenThreads::Thread
    {
        friend class OpenCOVER;

    public:
        deviceDiscovery();
        ~deviceDiscovery();
        void init();
        void update(); //< called by Input::update()
        void run(); //regularly check for new devices

        // only to be used from main thread
        const std::vector<const deviceInfo *> &getDevices() const;
        // only called on master, slaves don't know about devices
        sigslot::signal<const deviceInfo *> deviceAdded;

    private:
        std::vector<const deviceInfo *> devices;
        std::vector<deviceInfo*> toAdd;

        bool running = true;
        std::string broadcastAddress;
        int port;
        UDPComm * dComm=nullptr;
        OpenThreads::Mutex mutex;
    };

}
#endif
