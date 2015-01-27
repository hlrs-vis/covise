/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BrakePedal.h"

// BrakePedal ////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
BrakePedal *BrakePedal::p_brakepedal = NULL;

// constructor, destructor, instance ---------------------------------
BrakePedal::BrakePedal()
    : CanOpenDevice(*(CANProvider::instance()->p_CANOpenDisplay), (uint8_t)covise::coCoviseConfig::getInt("nodeID", "COVER.VehicleUtil.BrakePedal", 21))
{
    p_CANProv->registerDevice(this);
}

BrakePedal::~BrakePedal()
{
    p_brakepedal = NULL;
}

// singleton
BrakePedal *BrakePedal::instance()
{
    if (p_brakepedal == NULL)
    {
        p_brakepedal = new BrakePedal();
    }
    return p_brakepedal;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
int32_t BrakePedal::getPosition()
{
    position = *((int32_t *)readTPDO(1));
    return releasedPosition - position;
}

void BrakePedal::initCANOpenDevice()
{
    /*std::cerr << "Starting brake pedal..." << std::endl;*/

    enterPreOp();
    rt_task_sleep(100000000);

    startNode();
}
//--------------------------------------------------------------------
