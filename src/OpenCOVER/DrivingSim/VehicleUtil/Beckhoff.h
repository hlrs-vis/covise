/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Beckhoff_h
#define __Beckhoff_h

//--------------------------------------------------------------------
// PROJECT        Beckhoff                          © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        15-May-09, U. Woessner
// MODIFIED       22-July-09, S. Franz
//                - Application of HLRS style guide
//                - Changed class to singleton pattern
//                - Added HMIDeviceIface
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "CanOpenDevice.h"
#include "CanOpenController.h"
#include "XenomaiTask.h"
#include "HMIDeviceIface.h"

#include <native/timer.h>
#include <deque>
#include <cstdlib>

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class Beckhoff : public CanOpenDevice, public HMIDeviceIface
{
public:
    virtual ~Beckhoff();

    void initCANOpenDevice(); // override CanOpenDevice fct

    bool getDigitalIn(int module, int port);
    uint8_t getDigitalIn(int module);
    void setDigitalOut(int module, int port, bool state);

    float getAnalogIn(int module, int port);
    void setAnalogOut(int module, int port, float voltage);

    // HMIDeviceIface methods
    bool keyInLock(int /*keynumber*/)
    {
        return true;
    };
    bool ignitionOn()
    {
        return true;
    };
    bool initDevice()
    {
        return true;
    };
    bool startDevice()
    {
        return true;
    };
    bool stopDevice()
    {
        return true;
    };
    bool shutDownDevice()
    {
        return true;
    };

    static Beckhoff *instance(); // singleton

protected:
    Beckhoff();

    static Beckhoff *p_Beckhoff;
    uint8_t RPDODigital[6];
    uint8_t *TPDODigital;
    uint8_t RPDOAnalog[8];
    uint8_t *TPDOAnalog;
};
//--------------------------------------------------------------------

#endif
