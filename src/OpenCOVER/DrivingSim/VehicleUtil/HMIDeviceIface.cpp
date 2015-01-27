/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HMIDeviceIface.h"

// HMIDeviceIface ////////////////////////////////////////////////////

// constructor, destructor, instance ---------------------------------
HMIDeviceIface::HMIDeviceIface()
{
    /*std::cerr << "HMIDeviceIface::HMIDeviceIface()" << std::endl;*/

    VehicleUtil::instance()->addDevice(this);

    p_CANProv = CANProvider::instance();

    m_state = DEFAULT;
}

HMIDeviceIface::~HMIDeviceIface()
{
    /*std::cerr << "HMIDeviceIface::~HMIDeviceIface()" << std::endl;*/

    /* do not delete CANProvider - this is done by VehicleUtil class
   delete p_CANProv; */
}
//--------------------------------------------------------------------

// HMIDeviceIface methods --------------------------------------------
bool HMIDeviceIface::keyInLock(int /*keynumber*/)
{
    return true;
}

bool HMIDeviceIface::ignitionOn()
{
    return true;
}

bool HMIDeviceIface::initDevice()
{
    return true;
}

bool HMIDeviceIface::startDevice()
{
    return true;
}

bool HMIDeviceIface::stopDevice()
{
    return true;
}

bool HMIDeviceIface::shutDownDevice()
{
    return true;
}
//--------------------------------------------------------------------
