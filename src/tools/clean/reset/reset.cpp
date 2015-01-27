/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "ServoStar.h"
#include "CanOpenBus.h"
#include "PcanPci.h"

int main()
{
    PcanPci can(1, CAN_BAUD_1M);
    CanOpenBus bus(&can);
    ServoStar drive(&bus, 1);

    std::cerr << "Resetting Servostar...";
    if (drive.resetNode())
    {
        drive.microsleep(2000000);
        while (!bus.recvEmergencyObject(1, NULL))
            ;
        std::cerr << "done!" << std::endl;

        std::cerr << "Shutting down Servostar...";
        if (drive.shutdown())
            std::cerr << "done!" << std::endl;
        else
            std::cerr << "failed!" << std::endl;
    }
    else
        std::cerr << "failed!" << std::endl;
}
