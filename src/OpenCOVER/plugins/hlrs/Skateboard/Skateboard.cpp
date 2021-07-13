/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Skateboard.h"
#include "UDPComm.h"
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>

static float zeroAngle = 1152.;


Skateboard::Skateboard()
	: udp(NULL)
{
	init();
	sbData.fl = 0;
	sbData.fr = 0;
	sbData.rl = 0;
	sbData.rr = 0;
	sbData.button = 0;
	sbControl.cmd = 0;
	sbControl.value = 0;
        coVRNaviagationManager->registerNavigationProvider("Skateboard",this);
}
Skateboard::~Skateboard()
{
	delete udp;
        coVRNaviagationManager->unregisterNavigationProvider("Skateboard",this);
}

bool Skateboard::init()
{
	delete udp;

	const std::string host = covise::coCoviseConfig::getEntry("value", "Skateboard.serverHost", "192.168.178.36");
	unsigned short serverPort = covise::coCoviseConfig::getInt("Skateboard.serverPort", 31319);
	unsigned short localPort = covise::coCoviseConfig::getInt("Skateboard.localPort", 31319);
	std::cerr << "Skateboard config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
        bool ret = false;
	if(coVRMSController::instance()->isMaster())
	{
	    udp = new UDPComm(host.c_str(), serverPort, localPort);
            if(udp->isConnected())
            {
               ret= true;
            }
            else
            {
	        std::cerr << "Skateboard: falided to open local UDP port" << localPort << std::endl;
	        ret = false;
            }
            coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
	}
        else
        {
            coVRMSController::instance()->readMaster(&ret, sizeof(ret));
        }
        return ret;
        
}

void
Skateboard::run()
{
	running = true;
	doStop = false;
	if(coVRMSController::instance()->isMaster())
	{
		while (running && !doStop)
		{
			usleep(5000);
			this->updateThread();

		}
	}
}

void
Skateboard::stop()
{
	doStop = true;
}


void Skateboard::syncData()
{
        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves(&sbData, sizeof(sbData));
        }
        else
        {
            coVRMSController::instance()->readMaster(&sbData, sizeof(sbData));
        }
}

void Skateboard::updateThread()
{
	if (udp)
	{
		char tmpBuf[10000];
		int status = udp->receive(&tmpBuf, 10000);


		if (status > 0 && status >= sizeof(SBData))
		{

			{
				OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
				memcpy(&sbData, tmpBuf, sizeof(SBData));
			}

		}
		else if (status == -1)
		{
			std::cerr << "Skateboard::update: error while reading data" << std::endl;
			return;
		}
		else
		{
			std::cerr << "Skateboard::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
			return;
		}

	}
}
void Skateboard::Initialize()
{
	sbControl.cmd = 1;
	ret = udp->send(&sbControl, sizeof(SBCtrlData));
}

osg::Vec3d Skateboard::getNormal()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	float w = (sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0;
	float fs = sbData.fl + sbData.fr;
	float rs = sbData.rl + sbData.rr;
	float sl = sbData.fl + sbData.rl;
	float sr = sbData.fr + sbData.rr;

	return osg::Vec3d((sl - sr) / w, (fs - rs) / w, 1.0);
}

float Skateboard::getWeight() // weight in Kg
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
	return float((sbData.fl + sbData.fr + sbData.rl + sbData.rr) / 4.0);
}

unsigned char Skateboard::getButton()
{
	return sbData.button;
}

