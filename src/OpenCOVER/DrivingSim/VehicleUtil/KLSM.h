/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __KLSM_h
#define __KLSM_h

//--------------------------------------------------------------------
// PROJECT        KLSM                                     © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
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

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class KLSM : public HMIDeviceIface
{
public:
    virtual ~KLSM();

    // Methods to get status of klsm controls (for polling)
    int getShiftStat();
    bool getHornStat();
    bool getReturnStat();
    bool getBlinkLeftStat()
    {
        return p_CANProv->LSS_1.values.canmsg.cansignals.Blk_links;
    };
    bool getBlinkRightStat()
    {
        return p_CANProv->LSS_1.values.canmsg.cansignals.Blk_rechts;
    };

    static KLSM *instance(); // singleton

    enum ShiftStatus
    {
        ShiftUp = 1,
        NoShift = 0,
        ShiftDown = -1
    };

protected:
    KLSM();

    static KLSM *p_klsm;
};
//--------------------------------------------------------------------

#endif
