/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __IgnitionLock_h
#define __IgnitionLock_h

//--------------------------------------------------------------------
// PROJECT        Ignition Lock                            © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        21-July-09, S. Franz
// MODIFIED       22-July-09, S. Franz
//                - Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "HMIDeviceIface.h"
#include "Beckhoff.h"
#include "fasiUpdateManager.h"

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class IgnitionLock : public HMIDeviceIface, public fasiUpdateable
{
public:
    virtual ~IgnitionLock();

    int getLockState();
    void releaseKey();
    virtual bool update();

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

    static IgnitionLock *instance(); // singleton

    enum m_lockstates
    {
        KEYOUT = 0,
        KEYIN = 1,
        IGNITION = 2,
        ENGINESTART = 3,
        ENGINESTOP = 4
    };

protected:
    IgnitionLock();

    static IgnitionLock *p_ignitionlock;
    Beckhoff *p_beckhoff;
    bool unlock;
    double startTime;
};
//--------------------------------------------------------------------

#endif
