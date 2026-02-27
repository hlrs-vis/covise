/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ENCODER_PLUGIN_H
#define _ENCODER_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <util/SerialCom.h>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>

// reads  encoder and sets timestep
class Encoder : public opencover::coVRPlugin, public OpenThreads::Thread
{
public:
    Encoder();
    ~Encoder();
	//! this function is called from the main thread after the state for a frame is set up, just before preFrame()
	//! return true, if you need the scene to be rendered immediately
	virtual bool update();

	virtual void run();
private:
	covise::SerialCom *serial;
	double angle,oldAngle;
	std::string defaultDevice;
	std::string devstring;
	int direction;
	int loopsPerRev;
	int countsPerRev;
	OpenThreads::Mutex mutex;
	volatile bool doRun;
};
#endif
