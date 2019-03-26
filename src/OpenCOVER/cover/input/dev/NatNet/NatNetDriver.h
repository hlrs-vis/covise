/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * NatNet.h
 *
 *  Created on: Dec 9, 2015
 *      Author: uwe
 */

#ifndef NatNet_DRIVER_H
#define NatNet_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>


#include <cover/input/inputdevice.h>

/**
 * @brief The NatNetDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */



#include "NatNetClient.h"
#include "NatNetCAPI.h"

class NatNetDriver : public opencover::InputDevice
{
    //-------------------NatNet related stuff
    
	NatNetClient *nn;
    
    
    std::string trackerid;
    std::string buttonid;

    virtual bool poll();
    

	size_t m_numBodies; ///Number of NatNet bodies
	std::string m_NatNet_server;
	std::string m_NatNet_local;
	sNatNetClientConnectParams connectParams;

	std::vector< sNatNetDiscoveredServer > discoveredServers;
	char discoveredMulticastGroupAddr[kNatNetIpv4AddrStrLenMax] = NATNET_DEFAULT_MULTICAST_ADDRESS;
	int analogSamplesPerMocapFrame = 0;
	sServerDescription serverDescription;

	int ConnectClient();

public:
	virtual bool needsThread() const { return false; }; //< whether a thread should be spawned - reimplement if not necessary
    NatNetDriver(const std::string &name);
    virtual ~NatNetDriver();

	void DataHandler(sFrameOfMocapData* data);

    
};
#endif
