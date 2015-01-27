/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GasPedalControlTask.h"

//GasPedal////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
GasPedal *GasPedal::p_gaspedal = NULL;

// constructor, destructor, instance ---------------------------------
GasPedal::GasPedal()
{
    demandFrame.can_id = demandCanId;
    demandFrame.can_dlc = 8;
    demandFrame.data[0] = 0;
    demandFrame.data[1] = 0;
    demandFrame.data[2] = 0;
    demandFrame.data[3] = 0;
    demandFrame.data[4] = 0;
    demandFrame.data[5] = 0;
    demandFrame.data[6] = 0;
    demandFrame.data[7] = 0;
}

GasPedal::~GasPedal()
{
    p_seat = NULL;
}

// singleton
GasPedal *GasPedal::instance()
{
    if (p_gaspedal == NULL)
    {
        p_gaspedal = new GasPedal();
    }
    return p_gaspedal;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
int GasPedal::getChecksumStatus()
{
    return statusFrame.data[0];
}

int GasPedal::getAliveStatus()
{
    return (statusFrame.data[1] & 0xf);
}

int GasPedal::getStatusMode()
{
    return ((statusFrame.data[1] & 0x30) >> 4);
}

int GasPedal::getStatusAck()
{
    return ((statusFrame.data[1] & 0xc0) >> 6);
}

int GasPedal::getActualPositionValue()
{
    return (statusFrame.data[2] + ((statusFrame.data[3] & 0xf) << 8));
}

int GasPedal::getStatusActualPosition()
{
    return ((statusFrame.data[3] & 0x30) >> 4);
}

int GasPedal::getThermalReserve()
{
    return statusFrame.data[4];
}

int GasPedal::getActualCurrent()
{
    return statusFrame.data[5];
}

void GasPedal::setChecksumDemand(int cs)
{
    demandFrame.data[0] = cs;
}

void GasPedal::setAliveDemand(int ad)
{
    demandFrame.data[1] = (demandFrame.data[1] & ~0xf) | (ad & 0xf);
}

void GasPedal::incrementAliveDemand()
{
    demandFrame.data[1] = (demandFrame.data[1] & ~0xf) | ((((demandFrame.data[1] & 0xf) + 1) % 15) & 0xf);
}

void GasPedal::setTargetPositionValue(int tpv)
{
    demandFrame.data[1] = (demandFrame.data[1] & ~0xf0) + ((tpv & 0xf) << 4);
    demandFrame.data[2] = (tpv & 0xff0) >> 4;
}

void GasPedal::setMaxTargetForce(int maxtf)
{
    demandFrame.data[3] = maxtf;
}

void GasPedal::setMinTargetForce(int mintf)
{
    demandFrame.data[4] = mintf;
}

void GasPedal::setJitterAmplitude(int amp)
{
    demandFrame.data[5] = (demandFrame.data[5] & ~0x7) | (amp & 0x7);
}

void GasPedal::setJitterSignalForm(int form)
{
    demandFrame.data[5] = (demandFrame.data[5] & ~0x18) | ((form << 3) & 0x18);
}

void GasPedal::setJitterFrequency(int freq)
{
    demandFrame.data[5] = (demandFrame.data[5] & ~0x60) | ((freq << 5) & 0x60);
}

void GasPedal::lock()
{
    demandFrame.data[6] = 0;
}

void GasPedal::unlock()
{
    demandFrame.data[6] = 1;
}

void GasPedal::setStiffness(int stiffness)
{
    demandFrame.data[7] = stiffness;
}

void GasPedal::run()
{
    can->setRecvTimeout(1000000000);
    while (1)
    {
        can->recvFrame(statusFrame);

        /*RTIME currentTicks = rt_timer_read();
      SRTIME diffTime = rt_timer_ticks2ns(currentTicks - lastTicks);
      lastTicks = currentTicks;
      std::cerr << "Time diff: " << diffTime << std::endl;*/

        can->sendFrame(demandFrame);
        incrementAliveDemand();
    }
}
//--------------------------------------------------------------------
