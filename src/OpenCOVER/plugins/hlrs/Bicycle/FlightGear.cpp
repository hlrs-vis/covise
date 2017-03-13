/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "FlightGear.h"
#include "UDPComm.h"
#include <config/CoviseConfig.h>
#include "Bicycle.h"
#include <util/byteswap.h>
#include <util/unixcompat.h>

static float zeroAngle = 1152.;


FlightGear::FlightGear(BicyclePlugin* bicycle)
    : udp(NULL)
    , bicycle(bicycle)
{
    init();
}
FlightGear::~FlightGear()
{
    delete udp;
}

void FlightGear::init()
{
    delete udp;

    const std::string host = covise::coCoviseConfig::getEntry("value", "FlightGear.serverHost", "141.58.8.28");
    unsigned short serverPort = covise::coCoviseConfig::getInt("FlightGear.serverPort", 5253);
    unsigned short localPort = covise::coCoviseConfig::getInt("FlightGear.localPort", 5252);
    std::cerr << "FlightGear config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);
    return;
}

void
FlightGear::run()
{
    running = true;
    doStop = false;
    while (running)
    {
        usleep(5000);
        this->update();

    }
}

void
FlightGear::stop()
{
    doStop = true;
}


void FlightGear::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
    if (udp)
    {
	int status = udp->receive(&fgdata, sizeof(FGData));

        if (status == sizeof(FGData))
        {   
	    for (unsigned i = 0; i < 3; ++i)
            byteSwap(fgdata.position[i]);
	    for (unsigned i = 0; i < 3; ++i)
            byteSwap(fgdata.orientation[i]);
            

	    fprintf(stderr, "\r");
/*            fprintf(stderr, "Model: %s ", fgdata.Model);
	    for (unsigned i = 0; i < 3; ++i)
            fprintf(stderr, "Pos: %6f ", fgdata.position[i]);*/
        }
        else if (status == -1)
        {
            std::cerr << "FlightGear::update: error while reading data" << std::endl;
	    init();
	    return;
        }
        else
        {
            std::cerr << "FlightGear::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
	    init();
	    return;
        }
        if (bicycle->power>5)
	{
            fgcontrol.magnetos=1;
            fgcontrol.starter=true;
            fgcontrol.throttle=bicycle->power/10000;
	    fgcontrol.parkingBrake=0.0;
        }
	else
	{
            fgcontrol.magnetos=1;
            fgcontrol.starter=false;
            fgcontrol.throttle=0.0;
            fgcontrol.parkingBrake=1.0;
	}

	fprintf(stderr, "Sent throttle data0: %6f ", fgcontrol.throttle);
        ret = udp->send(&fgcontrol,sizeof(FGControl));
       
    } 
}

osg::Vec3d FlightGear::getPosition()
{
OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex); 
return osg::Vec3d(fgdata.position[0],fgdata.position[1],fgdata.position[2]);
}

osg::Vec3d FlightGear::getOrientation()
{
OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex); 
return osg::Vec3d(fgdata.orientation[0],fgdata.orientation[1],fgdata.orientation[2]);
}
