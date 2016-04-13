/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VehicleUtil.h"
#include <unistd.h>

// VehicleUtil ///////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
VehicleUtil *VehicleUtil::p_vutil = NULL;

// constructor, destructor, instance ---------------------------------
VehicleUtil::VehicleUtil()
{
    p_DeviceList = new std::list<HMIDeviceIface *>;

    //opencover::coShutDownHandlerList::instance()->addHandler(this);

    p_CANProv = CANProvider::instance();

    m_vehiclestate = KEYOUT;
}

VehicleUtil::~VehicleUtil()
{
    // free list
    delete p_DeviceList;
}

// singleton
VehicleUtil *VehicleUtil::instance()
{
    if (p_vutil == NULL)
    {
        p_vutil = new VehicleUtil();
    }
    return p_vutil;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
/**
 * HMI devices use this method to add themselves to a list which is 
 * applied to set a consolidated vehicle state and to shutdown the
 * devices
 *
 * @param HMIDeviceIface* a HMI device
 * @return none
 */
void VehicleUtil::addDevice(HMIDeviceIface *device)
{
    std::list<HMIDeviceIface *>::iterator i = p_DeviceList->begin();

    // check if the device is not already in list - if yes, return
    while (i != p_DeviceList->end())
    {
        if ((*i) == device)
        {
            return;
        }
        i++;
    }

    // add device to device list
    p_DeviceList->push_back(device);
}

/**
 * With this method the state of the vehicle can be controlled. It
 * provides for a consolidated state of all ecus / HMI devices. The
 * state is also controlled by comparing m_vehiclestate with the 
 * newly demanded state 
 *
 * @param int a state, see enum in header file
 * @return none
 */
void VehicleUtil::setVehicleState(int demandedstate)
{
    std::list<HMIDeviceIface *>::iterator i = p_DeviceList->begin();

    // key is out of ignition lock
    if (demandedstate == KEYOUT && m_vehiclestate == KEYIN)
    {
        p_CANProv->keyIsOut();

        i = p_DeviceList->begin(); // get position of first list entry

        while (i != p_DeviceList->end())
        {
            (*i)->shutDownDevice();
            i++;
        }

        m_vehiclestate = KEYOUT;
    }
    // key is in ignition lock
    else if (demandedstate == KEYIN && m_vehiclestate == KEYOUT)
    {
        p_CANProv->keyIsIn();

        i = p_DeviceList->begin();

        while (i != p_DeviceList->end())
        {
            (*i)->keyInLock(0); // keynumber should be added in the future
            i++;
        }

        m_vehiclestate = KEYIN;
    }
    // key was turned to "Ignite" position
    else if (demandedstate == KEYIN_IGNITED && m_vehiclestate == KEYIN)
    {
        p_CANProv->ignitionIsOn();

        i = p_DeviceList->begin();

        while (i != p_DeviceList->end())
        {
            (*i)->ignitionOn();
            i++;
        }

        m_vehiclestate = KEYIN_IGNITED;
    }
    // key is in "Start engine" position / engine is started
    else if (demandedstate == KEYIN_ESTART && m_vehiclestate == KEYIN_IGNITED)
    {
        p_CANProv->engineIsStarted();

        i = p_DeviceList->begin();

        while (i != p_DeviceList->end())
        {
            (*i)->initDevice();
            i++;
        }

        m_vehiclestate = KEYIN_ESTART;
    }
    // engine was started
    else if (demandedstate == KEYIN_ERUNNING && m_vehiclestate == KEYIN_ESTART)
    {
        p_CANProv->engineIsRunning();

        i = p_DeviceList->begin();

        while (i != p_DeviceList->end())
        {
            (*i)->startDevice();
            i++;
        }

        m_vehiclestate = KEYIN_ERUNNING;
    }
    // key was turned to "Stop engine" position
    else if (demandedstate == KEYIN_ESTOP && (m_vehiclestate == KEYIN_ERUNNING || m_vehiclestate == KEYIN_IGNITED || m_vehiclestate == KEYIN_ESTART))
    {
        p_CANProv->keyIsIn();

        i = p_DeviceList->begin();

        while (i != p_DeviceList->end())
        {
            (*i)->stopDevice();
            i++;
        }

        m_vehiclestate = KEYIN;
    }
}

/**
 * With this method the HMI devices and the CANProvider are stopped,
 * shutdown and the CANProvider singleton and the HMI device classses 
 * singletons are deleted subsequently!
 *
 * @param none
 * @return none
 */
void VehicleUtil::shutDown()
{
    std::list<HMIDeviceIface *>::iterator i;

    // stop all HMIDeviceIfaces in list
    setVehicleState(KEYIN_ESTOP);

    sleep(1);

    // shutDown all HMIDeviceIfaces in list
    setVehicleState(KEYOUT);

    sleep(1);

    // shut down CANProvider
    p_CANProv->shutdown();

    // call destructor of all HMIDeviceIfaces in list
    for (i = p_DeviceList->begin(); i != p_DeviceList->end(); ++i)
    {
        delete (*i);
    }

    // call destructor of CANProvider
    delete p_CANProv;
}
//--------------------------------------------------------------------
