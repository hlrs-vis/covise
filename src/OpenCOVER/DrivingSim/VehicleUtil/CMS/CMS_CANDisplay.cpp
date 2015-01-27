/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CMS_CANDisplay.h"

//CMS_CANDisplay//////////////////////////////////////////////////////

// setting the fields in the canmsg structs to the initial values
// CAN messages based on Porsche CAN Display DBC, 9x7, V 5.3 ---------

BC_1_D_Struct::BC_1_D_Struct()
{
    values.canmsg.ID = 865;
    values.canmsg.DLC = 1;
    cycle_time = 50; // changed value for faster response on BC operations

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals)); // set all signals to zero
}

BREMSE_2_D_Struct::BREMSE_2_D_Struct()
{
    values.canmsg.ID = 530;
    values.canmsg.DLC = 8;
    cycle_time = 50;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

FahrID_1k_D_Struct::FahrID_1k_D_Struct()
{
    values.canmsg.ID = 273;
    values.canmsg.DLC = 1;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_1_Struct::GW_D_1_Struct()
{
    values.canmsg.ID = 772;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_2_Struct::GW_D_2_Struct()
{
    values.canmsg.ID = 529;
    values.canmsg.DLC = 8;
    cycle_time = 40;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_3_Struct::GW_D_3_Struct()
{
    values.canmsg.ID = 1283;
    values.canmsg.DLC = 8;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_4_Struct::GW_D_4_Struct()
{
    values.canmsg.ID = 272;
    values.canmsg.DLC = 8;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_5_Struct::GW_D_5_Struct()
{
    values.canmsg.ID = 1551;
    values.canmsg.DLC = 8;
    cycle_time = 500;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_6_Struct::GW_D_6_Struct()
{
    values.canmsg.ID = 880;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_7_Struct::GW_D_7_Struct()
{
    values.canmsg.ID = 881;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_8_Struct::GW_D_8_Struct()
{
    values.canmsg.ID = 882;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_D_V_Struct::GW_D_V_Struct()
{
    values.canmsg.ID = 1803;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

GW_SVB_D_Struct::GW_SVB_D_Struct()
{
    values.canmsg.ID = 1583;
    values.canmsg.DLC = 8;
    cycle_time = 500;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

KOMBI_6_Struct::KOMBI_6_Struct()
{
    values.canmsg.ID = 773;
    values.canmsg.DLC = 8;
    cycle_time = 50; // according to dbc: 1000

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

MOTOR_4_D_Struct::MOTOR_4_D_Struct()
{
    values.canmsg.ID = 785;
    values.canmsg.DLC = 8;
    cycle_time = 100;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

MOTOR_6_D_Struct::MOTOR_6_D_Struct()
{
    values.canmsg.ID = 1579;
    values.canmsg.DLC = 8;
    cycle_time = 1000;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

PCM_1_Struct::PCM_1_Struct()
{
    values.canmsg.ID = 624;
    values.canmsg.DLC = 8;
    cycle_time = 500;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

RDK_1_D_Struct::RDK_1_D_Struct()
{
    values.canmsg.ID = 1574;
    values.canmsg.DLC = 8;
    cycle_time = 800;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

RDK_2_D_Struct::RDK_2_D_Struct()
{
    values.canmsg.ID = 775;
    values.canmsg.DLC = 4;
    cycle_time = 800;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

RDK_4_D_Struct::RDK_4_D_Struct()
{
    values.canmsg.ID = 1120;
    values.canmsg.DLC = 8;
    cycle_time = 800;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

UHR_1_Struct::UHR_1_Struct()
{
    values.canmsg.ID = 256;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

UHR_2_Struct::UHR_2_Struct()
{
    values.canmsg.ID = 257;
    values.canmsg.DLC = 8;
    cycle_time = 200;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

ButtonState_Struct::ButtonState_Struct()
{
    values.canmsg.ID = 50;
    values.canmsg.DLC = 4;
    cycle_time = 800;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

LEDState_Struct::LEDState_Struct()
{
    values.canmsg.ID = 50;
    values.canmsg.DLC = 2;
    cycle_time = 50;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}
//--------------------------------------------------------------------

// CAN messages for Continental Temic HLZ FFP gas pedal --------------
ANFORDERUNG_FFP_Struct::ANFORDERUNG_FFP_Struct()
{
    values.canmsg.ID = 351;
    values.canmsg.DLC = 8;
    cycle_time = 20;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}

STATUS_FFP_Struct::STATUS_FFP_Struct()
{
    values.canmsg.ID = 354;
    values.canmsg.DLC = 6;
    cycle_time = 20;

    memset(&values.canmsg.cansignals, 0, sizeof(values.canmsg.cansignals));
}
//--------------------------------------------------------------------
