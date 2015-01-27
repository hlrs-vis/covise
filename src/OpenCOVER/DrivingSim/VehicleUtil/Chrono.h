/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Chrono_h
#define __Chrono_h

//--------------------------------------------------------------------
// PROJECT        Chrono                                   © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    This singleton class enables you to control the
//                Porsche stopwatch using CAN signals
//
// CREATED        15-May-09, S. Franz
// MODIFIED       17-July-09, S. Franz
//                - Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "HMIDeviceIface.h"
#include "CANProvider.h"
#include <cover/coVRPluginSupport.h>

//--------------------------------------------------------------------
class Chrono : public HMIDeviceIface, public covise::coUpdateable
{
public:
    virtual ~Chrono();

    // public methods to control the chrono / stopwatch by CAN
    void resetChrono();
    void startChrono();
    void stopChrono();

    // HMIDeviceIface methods
    bool keyInLock(int keynumber);
    bool ignitionOn();
    bool initDevice();
    bool startDevice();
    bool stopDevice();
    bool shutDownDevice();

    static Chrono *instance(); // singleton

protected:
    Chrono();

    static Chrono *p_chrono;

private:
    bool update();

    double m_tstamp_chrono;
};
//--------------------------------------------------------------------

#endif
