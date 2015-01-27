/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FFWheel.h"

#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

FFWheel::FFWheel()
{
}

FFWheel::~FFWheel()
{
    /*
   if(coVRMSController::instance()->isMaster() && doRun)
   {
      doRun=false;
      fprintf(stderr,"waiting1\n");
      endBarrier.block(2); // wait until communication thread finishes
      fprintf(stderr,"done1\n");
   }
	*/
}

void FFWheel::softResetWheel()
{
    std::cerr << "No steering wheel reset implemented!" << std::endl;
}

void FFWheel::cruelResetWheel()
{
    std::cerr << "No steering wheel reset implemented!" << std::endl;
}

void FFWheel::shutdownWheel()
{
    std::cerr << "No steering wheel shutdown implemented!" << std::endl;
}
