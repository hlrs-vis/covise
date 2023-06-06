/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Encoder OpenCOVER Plugin (is polite)                          **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** June 2008  v1	    				       		                                **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "Encoder.h"
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>

#include <cover/coVRAnimationManager.h>

using namespace opencover;
Encoder::Encoder()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Encoder World\n");
#ifdef WIN32
	defaultDevice = "COM7";
#else
	defaultDevice = "/dev/ttyUSB0";
#endif
	devstring = covise::coCoviseConfig::getEntry("device", "OpenCOVER/Plugins/Encoder", defaultDevice);
	loopsPerRev = covise::coCoviseConfig::getInt("loopsPerRev", "OpenCOVER/Plugins/Encoder", 20);
	countsPerRev = covise::coCoviseConfig::getInt("countsPerRev", "OpenCOVER/Plugins/Encoder", 2000);
	direction = covise::coCoviseConfig::getInt("direction", "OpenCOVER/Plugins/Encoder", -1);
	serial = NULL;
	oldAngle = angle = 0.0;
	doRun = true;
	startThread();
}
//! this function is called from the main thread after the state for a frame is set up, just before preFrame()
//! return true, if you need the scene to be rendered immediately
bool Encoder::update()
{
	if (serial == NULL)
	{
		serial = new covise::SerialCom(devstring.c_str(), 115200);
	}
	else
	{

		if (serial->isBad())
		{
			serial = NULL;
		}
	}
	{
		OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
		if (angle != oldAngle)
		{
			oldAngle = angle;

			int numTS = coVRAnimationManager::instance()->getNumTimesteps();
			if (numTS <= 0)
				numTS = 10;

			int newTS = ((int)(direction * (angle / 360.0) * numTS* loopsPerRev)) % (numTS);
			if (newTS < 0)
				newTS = numTS + newTS;
			if (newTS > numTS)
				newTS = newTS - numTS;
			cerr << "newTS: " << newTS << endl;
			cerr << "Angle: " << angle << endl;

			coVRAnimationManager::instance()->requestAnimationFrame(newTS);

			return true; // request that scene be re-rendered
		}
	}
	return false;
}

void Encoder::run()
{

	int bufLen = 0;
	char buf[100]; 
	doRun = true;
	bufLen = 0;
	while(doRun)
	{
		if (bufLen > 100)
			bufLen = 0;
		if (serial != NULL)
		{
			char c;
			int numRead = serial->read(&c, 1);
			if (numRead > 0)
			{
				if (c == '\r')
				{
					buf[bufLen] = '\0';
					OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
					int count;
					sscanf(buf, "%d", &count);
					angle = (360 * count) / (double)countsPerRev;
					bufLen = 0;
				}
				else if (c == '\n')
				{
					bufLen = 0;
				}
				else
				{
					buf[bufLen] = c;
					bufLen++;
				}
			}
		}
	}
}

// this is called if the plugin is removed at runtime
Encoder::~Encoder()
{
    fprintf(stderr, "Goodbye\n");
	doRun = false;
	delete serial;
}

COVERPLUGIN(Encoder)
