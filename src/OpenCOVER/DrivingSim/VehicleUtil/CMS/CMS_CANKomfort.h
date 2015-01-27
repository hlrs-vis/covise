/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CMS_CANKomfort_h
#define __CMS_CANKomfort_h

//--------------------------------------------------------------------
// PROJECT        CMS_CANKomfort                           © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    Definition of CAN messages for CAN Komfort
//
// CREATED        15-May-09, S. Franz
// MODIFIED       23-July-09, S. Franz
//                Application of HLRS style guide
//
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "../XenomaiSocketCan.h"
#include "CMS_Struct.h"

//--------------------------------------------------------------------

// CAN messages based on Porsche CAN Komfort DBC, 9x7, V ?.? ---------

struct BC_1_Struct : CANMsgIface // sent by KLSM
{
    BC_1_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned BC_Up_Cursor : 1;
                unsigned BC_Down_Cursor : 1;
                unsigned BC_Reset : 1;
                unsigned BC_Set : 1;
            } cansignals __attribute__((aligned(1)));
        } canmsg;
    } values;
};

struct BUG_1_Struct : CANMsgIface
{
    BUG_1_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned BL_l_an : 1;
                unsigned BL_r_an : 1;
                unsigned St_FL_an : 1;
                unsigned S_SLicht : 1;
                unsigned S_ALicht : 1;
                unsigned St_ALicht : 1;
                unsigned BegL_an : 1;
                unsigned S_DV_offen : 1;
                unsigned A_Inst_ABel : 1;
                unsigned ph1 : 1;
                unsigned TFL_aus : 1;
                unsigned St_WBL : 1;
                unsigned ph2 : 4;
            } cansignals __attribute__((aligned(2)));
        } canmsg;
    } values;
};

struct FahrID_1k_Struct : CANMsgIface
{
    FahrID_1k_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned Kl_S : 1;
                unsigned Kl_15 : 1;
                unsigned Kl_X : 1;
                unsigned Kl_50 : 1;
                unsigned ph1 : 2;
                unsigned Kl_15SV : 1;
                unsigned ph2 : 1;
            } cansignals __attribute__((aligned(1)));
        } canmsg;
    } values;
};

struct FahrID_2_Struct : CANMsgIface
{
    FahrID_2_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned ph1 : 8;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned Funkschl_Nr : 4;
                unsigned ph5 : 4;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_K_2_Struct : CANMsgIface
{
    GW_K_2_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned Vref_K : 15;
                unsigned E_Sport_K : 1;
                unsigned LWS_LRW_K : 15;
                unsigned LWS_LRW_Sign_K : 1;
                unsigned St_LWS_K : 2;
                unsigned St_LWS_KL30_K : 1;
                unsigned poshebel_K : 4;
                unsigned ph1 : 1;
                unsigned FOT_Temp_K : 7;
                unsigned ph2 : 8;
                unsigned ph3 : 1;
                unsigned F_Bremse_1 : 1;
                unsigned F_LWS_1 : 1;
                unsigned F_Getriebe_1_K_2 : 1;
                unsigned F_Bremse_2 : 1;
                unsigned F_PCM_RAD_4 : 1;
                unsigned ph4 : 3;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct KOMBI_1_K_Struct : CANMsgIface
{
    KOMBI_1_K_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned t_stand_K : 12;
                unsigned F_stand_K : 1;
                unsigned Tenv_Einh_K : 1;
                unsigned ph1 : 2;
                unsigned tankf_K : 8;
                unsigned Tenv_anz_K : 8;
                unsigned Nachtank_K : 1;
                unsigned Dimmung_K : 7;
                unsigned ph2 : 2;
                unsigned T_VD_K : 2;
                unsigned T_TG_K : 3;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 1;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct KOMBI_7_K_Struct : CANMsgIface
{
    KOMBI_7_K_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned ph1 : 8;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct LSS_1_Struct : CANMsgIface // sent by KLSM
{
    LSS_1_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned Blk_links : 1;
                unsigned Blk_rechts : 1;
                unsigned T_Lichthupe : 1;
                unsigned S_Fernlicht : 1;
                unsigned ph1 : 1;
                unsigned Parklicht_links : 1;
                unsigned Parklicht_rechts : 1;
                unsigned T_Hupe : 1;
                unsigned Tipwischen : 1;
                unsigned Intervall : 1;
                unsigned WischenStufe_1 : 1;
                unsigned WischenStufe_2 : 1;
                unsigned Waschen : 1;
                unsigned ph2 : 1;
                unsigned Heckwischen : 1;
                unsigned SRA : 1;
                unsigned Intervallstufen : 4;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 4;
            } cansignals; // alignment must be a power of 2, so alignment of 5 byte is not possible
        } canmsg;
    } values;
};

struct LSS_2_Struct : CANMsgIface // sent by KLSM
{
    LSS_2_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned T_MFL_1 : 1;
                unsigned T_MFL_2 : 1;
                unsigned T_MFL_3 : 1;
                unsigned T_MFL_4 : 1;
                unsigned ph1 : 1;
                unsigned T_Push_to_Talk : 1;
                unsigned ph2 : 2;
                unsigned DW_Push_L : 1;
                unsigned DW_DR_L : 1;
                unsigned DW_Ticks_L : 6;
                unsigned DW_Push_R : 1;
                unsigned DW_DR_R : 1;
                unsigned DW_Ticks_R : 6;
                unsigned ph3 : 8;
            } cansignals __attribute__((aligned(4)));
        } canmsg;
    } values;
};

struct LSS_3_Struct : CANMsgIface // sent by KLSM
{
    LSS_3_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned GRA_Checksum : 8;
                unsigned GRA_HSchalt : 1;
                unsigned GRA_Tip_aus_GRA : 1;
                unsigned GRA_Tip_Verzoegern : 1;
                unsigned GRA_Tip_Beschl : 1;
                unsigned GRA_Verzoeg : 1;
                unsigned GRA_Beschl : 1;
                unsigned GRA_BT_Fehler : 1;
                unsigned LSS_Ident : 1;
                unsigned GRA_Tip_Setzen : 1;
                unsigned GRA_Tip_Wiederauf : 1;
                unsigned ph1 : 2;
                unsigned GRA_BZ : 4;
                unsigned Tip_Down : 1;
                unsigned Tip_Up : 1;
                unsigned ph2 : 6;
            } cansignals __attribute__((aligned(4)));
        } canmsg;
    } values;
};

struct NM_SM_Struct : CANMsgIface // sent by SM_Fahrer
{
    NM_SM_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned NMSM_Receiver : 8;
                unsigned NMSM_CmdRing : 1;
                unsigned NMSM_CmdAlive : 1;
                unsigned NMSM_CmdLimpHome : 1;
                unsigned ph1 : 1;
                unsigned NMSM_SleepInd : 1;
                unsigned NMSM_SleepAck : 1;
                unsigned ph2 : 2;
                unsigned NMSM_KAng : 1;
                unsigned NMSM_KL15 : 1;
                unsigned NMSM_CAN : 1;
                unsigned NMSM_Int : 1;
                unsigned NMSM_Ext : 1;
                unsigned NMSM_KL30 : 1;
                unsigned ph3 : 2;
                unsigned NMSM_Userdata : 24;
            } cansignals; // alignment has to be a power of 2, so alignment of 6 byte is not possible
        } canmsg;
    } values;
};

struct SITZM_V_Struct : CANMsgIface // sent by SM_Fahrer
{
    SITZM_V_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned SM_SW_Vers : 8;
                unsigned SM_CAN_STAND : 8;
                unsigned SM_SW_Tag : 5;
                unsigned ph1 : 3;
                unsigned SM_SW_Monat : 4;
                unsigned SM_KD_F : 1;
                unsigned SM_F_BusOff : 1;
                unsigned SM_F_CAN_Time : 1;
                unsigned SM_F_CAN_Sig : 1;
                unsigned SM_SW_Jahr : 8;
                unsigned SM_F_BusOff_Anz : 8;
                unsigned SM_TEC : 8;
                unsigned SM_REC : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct TSG_FT_1_Struct : CANMsgIface
{
    TSG_FT_1_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned ph1 : 8;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct TSG_FT_2_Struct : CANMsgIface
{
    TSG_FT_2_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned MemPos1_Anf : 1;
                unsigned MemPos1_Sp : 1;
                unsigned MemPos2_Anf : 1;
                unsigned MemPos2_Sp : 1;
                unsigned MemPosSchl_Anf : 1;
                unsigned MemPosSchl_Sp : 1;
                unsigned MemPosSL : 1;
                unsigned ph1 : 1;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};
//--------------------------------------------------------------------

struct Klima_Struct : CANMsgIface
{
    Klima_Struct();

    can_frame &theCANframe()
    {
        return values.frame;
    }

    union a
    {
        can_frame frame;
        struct b
        {
            uint32_t ID;
            uint8_t DLC;
            struct c
            {
                unsigned ph1 : 8;
                unsigned SitzheizungLinks : 2;
                unsigned SitzheizungRechts : 2;
                unsigned Lueftungsknopf : 1;
                unsigned ph21 : 1;
                unsigned ph22 : 1;
                unsigned ph23 : 1;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned Ventilator : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

#endif
