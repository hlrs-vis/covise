/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * inputsource.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: svnvlad
 */

#ifdef _WIN32
#include <winsock2.h>
#include <WS2tcpip.h>
#else
#include <arpa/inet.h>
#endif
#include "deviceDiscovery.h"
#include <util/UDPComm.h>
#include <config/CoviseConfig.h>
#include <util/threadname.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

namespace opencover
{

    deviceInfo::deviceInfo(const char* buf, std::string addr)
    {
        address = addr;
        char pn[200];
        char dn[200];
        sscanf(buf, "devInfo %s %s", pn, dn);
        pluginName = pn;
        deviceName = dn;
    }
    deviceInfo::~deviceInfo()
    {
    }

    deviceDiscovery::deviceDiscovery()
    {
        broadcastAddress = covise::coCoviseConfig::getEntry("broadcastAddress", "COVER.Input.DeviceDiscovery", "141.58.8.255");
        port = covise::coCoviseConfig::getInt("port", "COVER.Input.DeviceDiscovery", 31320);
        dComm = new UDPComm(broadcastAddress.c_str(), port, port);
        if (dComm->enableBroadcast(true) < 0)
        {
            cerr << "unable to enable Broadcast" << endl;
        }
    }
    void deviceDiscovery::init()
    {
	if (coVRMSController::instance()->isMaster())
	{
            start();
            dComm->send("enum", 5);
	}
    }
    deviceDiscovery::~deviceDiscovery()
    {
        running = false;
    }

    void deviceDiscovery::update()
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
        int numToAdd = toAdd.size();
	if (coVRMSController::instance()->isMaster())
	{  
            coVRMSController::instance()->sendSlaves(&numToAdd, sizeof(numToAdd));
	    for(const auto &i:toAdd)
	    {
		int len = i->pluginName.length();
		coVRMSController::instance()->sendSlaves(&len, sizeof(len));
		coVRMSController::instance()->sendSlaves(i->pluginName.c_str(), len+1);
		cover->addPlugin(i->pluginName.c_str());
	    }
	}
	else
	{
            coVRMSController::instance()->readMaster(&numToAdd, sizeof(numToAdd));
	    for(int i=0;i<numToAdd;i++)
	    {
		std::string pluginName;
		int len = 0;
		char buf[1000];
		coVRMSController::instance()->readMaster(&len, sizeof(len));
		coVRMSController::instance()->readMaster(buf, len+1);
		cover->addPlugin(buf);
	    }
	    toAdd.clear();
	}
    }
    void deviceDiscovery::run()
    {
        covise::setThreadName("deviceDiscoveryThread");
        char buffer[100];
        buffer[99] = '\0';
        while (running)
        {
            int numBytes = dComm->receive(buffer, sizeof(buffer), 60);
            if (numBytes > 0)
            {
                char str[INET_ADDRSTRLEN];

                sockaddr_in sin = dComm->getRemoteAddess();
                inet_ntop(AF_INET, &(sin.sin_addr), str, INET_ADDRSTRLEN);
                if (strncmp(buffer, "devInfo", 7) == 0)
                {
                    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
                    deviceInfo* di = new deviceInfo(buffer, str);
                    devices.push_back(di);
		    toAdd.push_back(di);
                }
                fprintf(stderr, "message received %s from %s\n", buffer,str);
            }

            
        }
    }

}
