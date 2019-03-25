/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifndef NatNet_DRIVER_H
#define NatNet_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>

#include "NatNetClient.h"

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
class NatNetDriver : public opencover::InputDevice
{
    //-------------------NatNet related stuff
	NatNetClient *nn; ///ART NatNet SDK class

    size_t m_numBodies; ///Number of NatNet bodies
	std::string m_NatNet_server;
	std::string m_NatNet_local;

    virtual bool poll();
    void initArrays();

	sNatNetClientConnectParams connectParams;

	std::vector< sNatNetDiscoveredServer > discoveredServers;
	char discoveredMulticastGroupAddr[kNatNetIpv4AddrStrLenMax] = NATNET_DEFAULT_MULTICAST_ADDRESS;
	int analogSamplesPerMocapFrame = 0;
	sServerDescription serverDescription;

public:
    NatNetDriver(const std::string &name);
    virtual ~NatNetDriver();

	void DataHandler(sFrameOfMocapData* data);
	int ConnectClient();

    //==============Hardware related interface methods=====================
    bool updateBodyMatrix(size_t idx); ///get NatNet body matrix
    bool updateFlyStick(size_t idx); ///get flystick body matrix
    bool updateHand(size_t idx);	/// get Hand matrix and other data
};
#endif
