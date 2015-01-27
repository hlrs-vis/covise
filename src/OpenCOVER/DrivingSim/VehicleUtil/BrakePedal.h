/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __BrakePedal_h
#define __BrakePedal_h

//--------------------------------------------------------------------
// PROJECT        BrakePedal                               © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    A class that is responsible for the control of the
//                brake pedal's CANopen interface. Besides the init
//                method there is the getPosition() function which
//                returns the brake pedals current position.
//
// CREATED        15-May-09, F. Seybold
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
#include "HMIDeviceIface.h"
#include "XenomaiTask.h"

#include <native/timer.h>

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class BrakePedal : public CanOpenDevice, public HMIDeviceIface
{
public:
    virtual ~BrakePedal();

    void initCANOpenDevice(); // override CanOpenDevice fct
    int32_t getPosition();

    static const int32_t maxPosition = 380;
    static BrakePedal *instance(); // singleton

protected:
    BrakePedal();

    static BrakePedal *p_brakepedal;
    int32_t position;
    static const int32_t releasedPosition = 268033224;
    static const int32_t pressedPosition = 268032844;
};
//--------------------------------------------------------------------

#endif
