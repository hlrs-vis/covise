/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Chrono.h"

// Chrono ////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
Chrono *Chrono::p_chrono = NULL;

// constructor, destructor, instance ---------------------------------
Chrono::Chrono()
{
    m_tstamp_chrono = 0.0;
    opencover::cover->getUpdateManager()->add(this);
}

Chrono::~Chrono()
{
    p_chrono = NULL;
}

// singleton
Chrono *Chrono::instance()
{
    if (p_chrono == NULL)
    {
        p_chrono = new Chrono();
    }
    return p_chrono;
}
//--------------------------------------------------------------------

// public methods to control the chrono / stopwatch by CAN -----------
void Chrono::resetChrono()
{
    m_tstamp_chrono = opencover::cover->frameTime();
}

void Chrono::startChrono()
{
    p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Start_KI = 1;
    p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Stop_KI = 0;
}

void Chrono::stopChrono()
{
    p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Start_KI = 0;
    p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Stop_KI = 1;
}
//--------------------------------------------------------------------

// HMIDeviceIface methods --------------------------------------------
bool Chrono::keyInLock(int /*keynumber*/)
{
    if (m_state == Chrono::DEFAULT)
    {
        m_state = Chrono::KEY;
        return true;
    }
    else
        return false;
}

bool Chrono::ignitionOn()
{
    if (m_state == Chrono::KEY)
    {
        m_state = Chrono::IGNITED;
        return true;
    }
    else
        return false;
}

bool Chrono::initDevice()
{
    if (m_state == Chrono::IGNITED)
    {
        resetChrono();

        m_state = Chrono::INITIALIZED;
        return true;
    }
    else
        return false;
}

bool Chrono::startDevice()
{
    if (m_state == Chrono::INITIALIZED)
    {
        p_CANProv->KOMBI_6.values.canmsg.cansignals.St_KI_SDisp = 1;

        m_state = Chrono::STARTED;
        return true;
    }
    else
        return false;
}

bool Chrono::stopDevice()
{
    if (m_state == Chrono::STARTED)
    {
        p_CANProv->KOMBI_6.values.canmsg.cansignals.St_KI_SDisp = 0;

        m_state = Chrono::KEY;
        return true;
    }
    else
        return false;
}

bool Chrono::shutDownDevice()
{
    if (m_state == Chrono::KEY)
    {
        m_state = Chrono::DEFAULT;
        return true;
    }
    else
        return false;
}
//--------------------------------------------------------------------

// private methods ---------------------------------------------------
bool Chrono::update()
{
    // generate falling edges
    if (m_tstamp_chrono + 0.2 > opencover::cover->frameTime())
    {
        p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Reset_KI = 1;
    }
    else
        p_CANProv->KOMBI_6.values.canmsg.cansignals.SU_Reset_KI = 0;

    return true;
}
//--------------------------------------------------------------------
