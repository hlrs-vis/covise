/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Klima_h
#define __Klima_h

//--------------------------------------------------------------------
// PROJECT        Klima                                     © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    BLA BLA BLA BLA BLA BLA
//
// CREATED        Uwe
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "HMIDeviceIface.h"

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class Klima : public HMIDeviceIface
{
public:
    virtual ~Klima();

    // Methods to get status of Klima controls (for polling)
    float getFanPower();
    int getSeatHeaterLeftStat();
    int getSeatHeaterRightStat();
    bool getFanButtonStat();

    static Klima *instance(); // singleton

    enum HeaterStatus
    {
        Off = 0,
        Low = 1,
        High = 2
    };

protected:
    Klima();

    static Klima *p_Klima;
};
//--------------------------------------------------------------------

#endif
