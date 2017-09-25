/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Seat_h
#define __Seat_h

//--------------------------------------------------------------------
// PROJECT        Seat                                     © 2009 HLRS
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
#include <OpenVRUI/coUpdateManager.h>

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class Seat : public HMIDeviceIface, public vrui::coUpdateable
{
public:
    virtual ~Seat();

    // methods to control the seat module
    void move2Mempos1();
    void saveMempos1();
    void move2Mempos2();
    void saveMempos2();

    void setKeyNumber(int keynumber);

    void move2Memposkey();
    void saveMemposkey();
    void delMemposkey();

    static Seat *instance(); // singleton
    static const int MaxKeyNr = 6;

protected:
    Seat();

    static Seat *p_seat;

private:
    bool update();

    double m_tstamp_move2Mempos1;
    double m_tstamp_saveMempos1;

    double m_tstamp_move2Mempos2;
    double m_tstamp_saveMempos2;

    double m_tstamp_move2Memposkey;
    double m_tstamp_saveMemposkey;

    int m_keynumber;
};
//--------------------------------------------------------------------

#endif
