/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __VehicleUtil_h
#define __VehicleUtil_h

//--------------------------------------------------------------------
// PROJECT        VehicleUtil                              © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        20-July-09, S. Franz
// MODIFIED       21-July-09, S. Franz
//                - Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "cover/coShutDownHandler.h"
#include "HMIDeviceIface.h"
#include "CANProvider.h"
#include <list>
#include <iostream>

class HMIDeviceIface; // fwd declaration

//--------------------------------------------------------------------
class VEHICLEUTILEXPORT VehicleUtil : public opencover::coShutDownHandler
{
public:
    virtual ~VehicleUtil();

    void addDevice(HMIDeviceIface *device);
    void setVehicleState(int demandedstate);
    int getVehicleState()
    {
        return m_vehiclestate;
    };
    void shutDown();

    static VehicleUtil *instance(); // singleton

    // Enum to control vehicle state from GUI or from steering wheel
    // plugin using setVehicleState(int thestate)
    enum m_vehiclestates
    {
        KEYOUT = 0,
        KEYIN = 1,
        KEYIN_IGNITED = 2,
        KEYIN_ESTART = 3,
        KEYIN_ERUNNING = 4,
        KEYIN_ESTOP = 5
    };

protected:
    VehicleUtil();

    static VehicleUtil *p_vutil;

private:
    std::list<HMIDeviceIface *> *p_DeviceList;
    CANProvider *p_CANProv;
    int m_vehiclestate;
};
//--------------------------------------------------------------------

#endif
