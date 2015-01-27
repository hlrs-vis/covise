/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Klima.h"

//Klima//////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
Klima *Klima::p_Klima = NULL;

// constructor, destructor, instance ---------------------------------
Klima::Klima()
{
}

Klima::~Klima()
{
    p_Klima = NULL;
}

// singleton
Klima *Klima::instance()
{
    if (p_Klima == NULL)
    {
        p_Klima = new Klima();
    }
    return p_Klima;
}
//--------------------------------------------------------------------

// public methods to get status of Klima controls (for polling) -------
float Klima::getFanPower()
{
    return (float)(p_CANProv->Klima.values.canmsg.cansignals.Ventilator) / (float)0xfa;
}

int Klima::getSeatHeaterLeftStat()
{
    return p_CANProv->Klima.values.canmsg.cansignals.SitzheizungLinks;
}

int Klima::getSeatHeaterRightStat()
{
    return p_CANProv->Klima.values.canmsg.cansignals.SitzheizungRechts;
}
bool Klima::getFanButtonStat()
{
    if (p_CANProv->Klima.values.canmsg.cansignals.Lueftungsknopf != 0)
        return true;
    else
        return false;
}
//--------------------------------------------------------------------
