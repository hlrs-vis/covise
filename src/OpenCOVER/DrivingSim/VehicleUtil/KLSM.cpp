/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KLSM.h"

//KLSM//////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
KLSM *KLSM::p_klsm = NULL;

// constructor, destructor, instance ---------------------------------
KLSM::KLSM()
{
}

KLSM::~KLSM()
{
    p_klsm = NULL;
}

// singleton
KLSM *KLSM::instance()
{
    if (p_klsm == NULL)
    {
        p_klsm = new KLSM();
    }
    return p_klsm;
}
//--------------------------------------------------------------------

// public methods to get status of klsm controls (for polling) -------
int KLSM::getShiftStat()
{
    if (p_CANProv->LSS_3.values.canmsg.cansignals.Tip_Down == 1)
        return ShiftDown;
    else if (p_CANProv->LSS_3.values.canmsg.cansignals.Tip_Up == 1)
        return ShiftUp;
    else
        return NoShift;
}

bool KLSM::getHornStat()
{
    if (p_CANProv->LSS_1.values.canmsg.cansignals.T_Hupe == 1)
        return true;
    else
        return false;
}

bool KLSM::getReturnStat()
{
    if (p_CANProv->LSS_2.values.canmsg.cansignals.T_MFL_3)
        return true;
    else
        return false;
}
//--------------------------------------------------------------------
