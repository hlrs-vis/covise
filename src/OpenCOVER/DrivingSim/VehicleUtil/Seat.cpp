/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Seat.h"

// Seat //////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
Seat *Seat::p_seat = NULL;

// constructor, destructor, instance ---------------------------------
Seat::Seat()
{
    m_keynumber = 0;

    m_tstamp_move2Mempos1 = 0.0;
    m_tstamp_saveMempos1 = 0.0;
    m_tstamp_move2Mempos2 = 0.0;
    m_tstamp_saveMempos2 = 0.0;
    m_tstamp_move2Memposkey = 0.0;
    m_tstamp_saveMemposkey = 0.0;
    opencover::cover->getUpdateManager()->add(this);
}

Seat::~Seat()
{
    p_seat = NULL;
}

// singleton
Seat *Seat::instance()
{
    if (p_seat == NULL)
    {
        p_seat = new Seat();
    }
    return p_seat;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
void Seat::move2Mempos1()
{
    m_tstamp_move2Mempos1 = opencover::cover->frameTime();
}

void Seat::saveMempos1()
{
    m_tstamp_saveMempos1 = opencover::cover->frameTime();
}

void Seat::move2Mempos2()
{
    m_tstamp_move2Mempos2 = opencover::cover->frameTime();
}
void Seat::saveMempos2()
{
    m_tstamp_saveMempos2 = opencover::cover->frameTime();
}

void Seat::setKeyNumber(int keynumber)
{
    if (0 < keynumber && keynumber <= MaxKeyNr)
    {
        m_keynumber = keynumber;
        p_CANProv->FahrID_2.values.canmsg.cansignals.Funkschl_Nr = m_keynumber;
    }
    else
        std::cerr << "Key number was higher than 15 or 0!" << std::endl;
}

void Seat::move2Memposkey()
{
    m_tstamp_move2Memposkey = opencover::cover->frameTime();
}

void Seat::saveMemposkey()
{
    m_tstamp_saveMemposkey = opencover::cover->frameTime();
}

void Seat::delMemposkey()
{
    //TODO(sebastian): FOR WHAT?
}
//--------------------------------------------------------------------

// Private methods ---------------------------------------------------
bool Seat::update()
{
    // generate falling edges
    if (m_tstamp_move2Memposkey + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPosSchl_Anf = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPosSchl_Anf = 0;

    if (m_tstamp_saveMemposkey + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPosSchl_Sp = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPosSchl_Sp = 0;

    if (m_tstamp_move2Mempos1 + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos1_Anf = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos1_Anf = 0;

    if (m_tstamp_saveMempos1 + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos1_Sp = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos1_Sp = 0;

    if (m_tstamp_move2Mempos2 + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos2_Anf = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos2_Anf = 0;

    if (m_tstamp_saveMempos2 + 0.5 < opencover::cover->frameTime())
    {
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos2_Sp = 1;
    }
    else
        p_CANProv->TSG_FT_2.values.canmsg.cansignals.MemPos2_Sp = 0;

    return true;
}
//--------------------------------------------------------------------
