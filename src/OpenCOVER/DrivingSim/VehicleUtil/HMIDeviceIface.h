/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __HMIDeviceIface_h
#define __HMIDeviceIface_h

//--------------------------------------------------------------------
// PROJECT			HMIDeviceIface						           © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    This is an interface which should be used for all
//                HMI device classes. The methods guarantee a
//                consistent start / stop / init behavior for all ECUs
//
// CREATED			15-May-09, S. Franz
// MODIFIED    	17-July-09, S. Franz
//						- Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "VehicleUtil.h"
#include "CANProvider.h"
#include <iostream>

//--------------------------------------------------------------------
class HMIDeviceIface
{
public:
    HMIDeviceIface();
    virtual ~HMIDeviceIface();

    // IFACE!
    // Please overwrite the following fcts if necessary

    virtual bool keyInLock(int keynumber); // Key is in lock (clamp S = 1)
    virtual bool ignitionOn(); // Ignition is on (clamp 15 = 1)
    virtual bool initDevice(); // Starting the engine (clamp X, 50 = 1)
    virtual bool startDevice(); // Engine is running (clamp X, 50 = 0, Motorlauf = 1)
    virtual bool stopDevice(); // Stopping the engine (Motorlauf = 0)
    virtual bool shutDownDevice(); // Key is out of lock (clamp S = 0)
    CANProvider *p_CANProv;

protected:
    // These states can be useful in HMI device classes for easy
    // control of the different ignition lock conditions
    enum m_states
    {
        DEFAULT = 0,
        KEY = 1,
        IGNITED = 2,
        INITIALIZED = 3,
        STARTED = 4
    };

    int m_state;
};
//--------------------------------------------------------------------

#endif
