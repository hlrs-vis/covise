/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CMS_CANKomfort.h"

//CMS_CANDisplay//////////////////////////////////////////////////////

// setting the fields in the canmsg structs to the initial values
// CAN messages based on Porsche CAN Komfort DBC, 9x7, V ?.? ---------

BC_1_Struct::BC_1_Struct() // sent by KLSM
{
    values.canmsg.ID = 865;
    values.canmsg.DLC = 1;
    cycle_time = 1000;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals)); // set all signals to zero
}

BUG_1_Struct::BUG_1_Struct()
{
    values.canmsg.ID = 354;
    values.canmsg.DLC = 2;
    cycle_time = 400;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

FahrID_1k_Struct::FahrID_1k_Struct()
{
    values.canmsg.ID = 357;
    values.canmsg.DLC = 1;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

FahrID_2_Struct::FahrID_2_Struct()
{
    values.canmsg.ID = 608;
    values.canmsg.DLC = 8;
    cycle_time = 300;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_K_2_Struct::GW_K_2_Struct()
{
    values.canmsg.ID = 805;
    values.canmsg.DLC = 8;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

KOMBI_1_K_Struct::KOMBI_1_K_Struct()
{
    values.canmsg.ID = 546;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

KOMBI_7_K_Struct::KOMBI_7_K_Struct()
{
    values.canmsg.ID = 1578;
    values.canmsg.DLC = 8;
    cycle_time = 1000;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

LSS_1_Struct::LSS_1_Struct() // sent by KLSM
{
    values.canmsg.ID = 352;
    values.canmsg.DLC = 5;
    cycle_time = 400;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

LSS_2_Struct::LSS_2_Struct() // sent by KLSM
{
    values.canmsg.ID = 1392;
    values.canmsg.DLC = 4;
    cycle_time = 500;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

LSS_3_Struct::LSS_3_Struct() // sent by KLSM
{
    values.canmsg.ID = 611;
    values.canmsg.DLC = 4;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

NM_SM_Struct::NM_SM_Struct() // sent by SM_Fahrer
{
    values.canmsg.ID = 1030;
    values.canmsg.DLC = 6;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

SITZM_V_Struct::SITZM_V_Struct() // sent by SM_Fahrer
{
    values.canmsg.ID = 1798;
    values.canmsg.DLC = 8;
    cycle_time = 1000;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

TSG_FT_1_Struct::TSG_FT_1_Struct()
{
    values.canmsg.ID = 612;
    values.canmsg.DLC = 8;
    cycle_time = 300;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

TSG_FT_2_Struct::TSG_FT_2_Struct()
{
    values.canmsg.ID = 1314;
    values.canmsg.DLC = 8;
    cycle_time = 300;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}
//--------------------------------------------------------------------

Klima_Struct::Klima_Struct()
{
    values.canmsg.ID = 1122;
    values.canmsg.DLC = 8;
    cycle_time = 300;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}
