/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CMS_CANDisplay_h
#define __CMS_CANDisplay_h

//--------------------------------------------------------------------
// PROJECT        CMS_CANDisplay                           © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    Definition of CAN messages for CAN Display
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

// CAN messages based on Porsche CAN Display DBC, 9x7, V 5.3 ---------

static const int MaxOilLevel = 100;
static const int MaxPetrolLevel = 100;
static const int MinOilLevel = 0;
static const int MinPetrolLevel = 0;
static const int MinSpeed = 0;
static const int MaxSpeed = 655;
static const int MinATempC = -50;
static const int MaxATempC = 70;
static const int MinRPM = 0;
static const int MaxRPM = 16256;
static const int MinTMot = -54;
static const int MaxTMot = 287;
static const int MinOilP = 0;
static const int MaxOilP = 10;
static const int MinTOil = -54;
static const int MaxTOil = 287;

struct BC_1_D_Struct : public CANMsgIface
{
    BC_1_D_Struct(); // ructor

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
                unsigned int BC_Up_Cursor_D : 1;
                unsigned int BC_Down_Cursor_D : 1;
                unsigned int BC_Reset_D : 1;
                unsigned int BC_Set_D : 1;
            } cansignals __attribute__((aligned(1)));
        } canmsg;
    } values;
};

struct BREMSE_2_D_Struct : public CANMsgIface
{
    BREMSE_2_D_Struct();

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
                unsigned ph1 : 1; // placeholder for bit alignment
                unsigned RG_VL_D : 15;
                unsigned E_Sport_D : 1;
                unsigned RG_VR_D : 15;
                unsigned St_Sport_ESP_D : 1;
                unsigned RG_HL_D : 15;
                unsigned W_ESP_Tas_D : 1;
                unsigned RG_HR_D : 15;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct FahrID_1k_D_Struct : CANMsgIface
{
    FahrID_1k_D_Struct();

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
                unsigned Kl_S_D : 1;
                unsigned Kl_15_D : 1;
                unsigned Kl_X_D : 1;
                unsigned Kl_50_D : 1;
                unsigned ph1 : 2;
                unsigned Kl_15SV_D : 1;
                unsigned ph2 : 1;
            } cansignals __attribute__((aligned(1)));
        } canmsg;
    } values;
};

struct GW_D_1_Struct : CANMsgIface
{
    GW_D_1_Struct();

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
                unsigned Atemp_ungef_D : 8;
                unsigned MUL_Verdeck_D : 1;
                unsigned S_DH_offen_D : 1;
                unsigned W_Heckspoiler_D : 1;
                unsigned SF_VD_987_D : 1;
                unsigned SF_VD_997_D : 1;
                unsigned S_Targa_Dach_D : 1;
                unsigned W_VD_DH_D : 1;
                unsigned S_Targa_Kl_D : 1;
                unsigned St_Verdeck_997_D : 3;
                unsigned ph1 : 1;
                unsigned St_Verdeck_987_D : 3;
                unsigned ph2 : 1;
                unsigned BL_l_an_D : 1;
                unsigned BL_r_an_D : 1;
                unsigned St_FL_an_D : 1;
                unsigned S_SLicht_D : 1;
                unsigned S_ALicht_D : 1;
                unsigned St_ALicht_D : 1;
                unsigned BegL_an_D : 1;
                unsigned S_DV_offen_D : 1;
                unsigned A_Inst_ABel_D : 1;
                unsigned ph3 : 1;
                unsigned TFL_aus_D : 1;
                unsigned St_WBL_D : 1;
                unsigned Parklicht_links_D : 1;
                unsigned Parklicht_rechts_D : 1;
                unsigned ph4 : 1;
                unsigned St_CAN_Komf : 1;
                unsigned W_Verdeck_997_D : 3;
                unsigned W_Verdeck_987_D : 3;
                unsigned ph5 : 2;
                unsigned DWA_Status_D : 3;
                unsigned ph6 : 5;
                unsigned F_Heck_1_D_1 : 1;
                unsigned F_Verdeck_1 : 1;
                unsigned F_Bug_1 : 1;
                unsigned F_Bug_2_D_1 : 1;
                unsigned F_LSS_1 : 1;
                unsigned F_Targa_1 : 1;
                unsigned ph7 : 2;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_2_Struct : CANMsgIface
{
    GW_D_2_Struct();

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
                unsigned nmot_D : 16;
                unsigned gangauti_D : 4;
                unsigned poshebel_D : 4;
                unsigned gnot_D : 4;
                unsigned ph1 : 2;
                unsigned stat_fgr_D : 2;
                unsigned Tmot_D : 8;
                unsigned E_tm_D : 1;
                unsigned Land_D : 7;
                unsigned Overboost_D : 1;
                unsigned E_EGAS_D : 1;
                unsigned Hochschalt_Anz_D : 1;
                unsigned ph2 : 5;
                unsigned F_Getriebe_1_D_2 : 1;
                unsigned F_Motor_1_D_2 : 1;
                unsigned F_Motor_2_D_2 : 1;
                unsigned F_Motor_5_D_2 : 1;
                unsigned F_Motor_3_D_2 : 1;
                unsigned ph3 : 3;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_3_Struct : CANMsgIface
{
    GW_D_3_Struct();

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
                unsigned FT_FH_Oeffnung_D : 8;
                unsigned BT_FH_Oeffnung_D : 8;
                unsigned FT_Tuer_geoffnet_D : 1;
                unsigned FT_verriegelt_D : 1;
                unsigned BT_Tuer_geoffnet_D : 1;
                unsigned BT_verriegelt_D : 1;
                unsigned Sportmodus_BSG_D : 1;
                unsigned S_RFL_D : 1;
                unsigned St_Wischer_D : 1;
                unsigned T_Sport_BSG_D : 1;
                unsigned Funkschl_Nr_D : 4;
                unsigned ph1 : 4;
                unsigned Innenr_Hell_D : 8;
                unsigned ph2 : 2;
                unsigned FLuft_Geb_D : 3;
                unsigned ph3 : 3;
                unsigned ph4 : 8;
                unsigned F_TSG_FT_1 : 1;
                unsigned F_TSG_BT_1 : 1;
                unsigned F_BSG_1_D_3 : 1;
                unsigned F_FahrID_2_D_3 : 1;
                unsigned F_Klima_1 : 1;
                unsigned F_Bug_1_D_3 : 1;
                unsigned F_Heck_1_D_3 : 1;
                unsigned ph5 : 1;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_4_Struct : CANMsgIface
{
    GW_D_4_Struct();

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
                unsigned ph1 : 7;
                unsigned PSM_Licht_Blinkt : 1;
                unsigned ABS_failure : 1;
                unsigned Warning_Brake_Failure : 1;
                unsigned ph2 : 3;
                unsigned PSM_Off : 1;
                unsigned ph2_2 : 2;
                unsigned PSM_in_diagnosis_mode_PSM_ABS_BRAKE_blinkt : 1;
                unsigned PTM_failure : 1;
                unsigned PTM_briefly_inactive_too_hot : 1;
                unsigned PTM_temperature_protection : 1;
                unsigned ph3 : 4;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned AirbagAnzeige : 1;
                unsigned ph7 : 2;
                unsigned SystemFaultAirbag : 1;
                unsigned S_Gurt_F_D : 1;
                unsigned ph8 : 1;
                unsigned S_Gurt_BF_D : 1;
                unsigned CheckPassengerSeatSetting : 1;
                unsigned SystemFaultVisitWorkshop : 1;
                unsigned NothingVisible1 : 1;
                unsigned SystemFaultAirbag_GurtLich_AirbagLicht : 1;
                unsigned ph9 : 5;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_5_Struct : CANMsgIface
{
    GW_D_5_Struct();

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

struct GW_D_6_Struct : CANMsgIface
{
    GW_D_6_Struct();

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
                unsigned Lenkwinkel : 16;
                unsigned lenkwinkelfehler : 1;
                unsigned ph4 : 2;
                unsigned Eing_Gang_D : 3;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
                unsigned ph9 : 2;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_7_Struct : CANMsgIface
{
    GW_D_7_Struct();

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

struct GW_D_8_Struct : CANMsgIface
{
    GW_D_8_Struct();

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
                unsigned Vref_D : 15;
                unsigned ph1 : 1;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned F_Bremse_1_D_8 : 1;
                unsigned ph7 : 7;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct GW_D_V_Struct : CANMsgIface
{
    GW_D_V_Struct();

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

struct GW_SVB_D_Struct : CANMsgIface
{
    GW_SVB_D_Struct();

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
                unsigned ph1 : 1;
                unsigned SVB_Getriebe_D : 1;
                unsigned ph2 : 3;
                unsigned SVB_ACD_D : 1;
                unsigned ph3 : 6;
                unsigned SVB_Verdeck_D : 1;
                unsigned ph4 : 4;
                unsigned SVB_RDK_D : 1;
                unsigned ph5 : 2;
                unsigned SVB_Kombi_D : 1; // byte 4 (von 0) bit 0
                unsigned ph6 : 1;
                unsigned SVB_PCM_D : 1; // byte 4 bit 2
                unsigned ph7 : 3;
                unsigned ph8 : 8;
                unsigned ph9 : 8;
                unsigned ph10 : 8;
                unsigned ph11 : 7;
                unsigned SVB_BC_D : 1;
                unsigned SVB_GRA_D : 1;
                unsigned ph12 : 2;
                unsigned SVB_ALR_D : 1;
                unsigned ph13 : 2;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct KOMBI_6_Struct : CANMsgIface
{
    KOMBI_6_Struct();

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
                unsigned Sprache : 3;
                unsigned SU_RZeit_KI : 1;
                unsigned SU_Start_KI : 1;
                unsigned SU_Stop_KI : 1;
                unsigned SU_Reset_KI : 1;
                unsigned SU_ZZeit_KI : 1;
                unsigned ph1 : 8;
                unsigned ph2 : 8;
                unsigned ph3 : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 1;
                unsigned ph6 : 8;
                unsigned ph7 : 4;
                unsigned St_KI_SDisp : 1;
                unsigned ph8 : 2;
                unsigned Beep_1 : 1;
                unsigned Beep_3 : 1;
                unsigned Gong : 1;
                unsigned ph9 : 5;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct MOTOR_4_D_Struct : CANMsgIface
{
    MOTOR_4_D_Struct();

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
                unsigned L_checkE_D : 2;
                unsigned B_tankd_D : 1;
                unsigned B_redmd_D : 1;
                unsigned E_mspg_D : 1;
                unsigned E_Toel_D : 1;
                unsigned S_Oeldr_D : 1;
                unsigned KL_Gen_D : 1;
                unsigned KuehlerL_Stg_D : 7;
                unsigned Motorlauf_D : 1;
                unsigned kva_D : 16;
                unsigned P_Lade_D : 8;
                unsigned Toel_D : 8;
                unsigned Oeldruck_D : 8;
                unsigned S_KuehlWES_D : 1;
                unsigned ph1 : 7;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct MOTOR_6_D_Struct : CANMsgIface
{
    MOTOR_6_D_Struct();

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

struct PCM_1_Struct : CANMsgIface
{
    PCM_1_Struct();

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
                unsigned PCM_Std : 5;
                unsigned PCM_Uebern : 1;
                unsigned ph1 : 1;
                unsigned Reset_BC_KI : 1;
                unsigned PCM_Min : 6;
                unsigned ph2 : 2;
                unsigned PCM_Sek : 6;
                unsigned ph3 : 2;
                unsigned PCM_JJ : 7;
                unsigned ph4 : 1;
                unsigned PCM_Mon : 4;
                unsigned ph5 : 4;
                unsigned PCM_Tag : 5;
                unsigned ph6 : 3;
                unsigned Helligkeit : 8;
                unsigned ph7 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct RDK_1_D_Struct : CANMsgIface
{
    RDK_1_D_Struct();

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
                unsigned Diff_Druck_VL_D : 7;
                unsigned ph3 : 1;
                unsigned Diff_Druck_VR_D : 7;
                unsigned ph4 : 1;
                unsigned Diff_Druck_HL_D : 7;
                unsigned ph5 : 1;
                unsigned Diff_Druck_HR_D : 7;
                unsigned ph6 : 1;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct RDK_2_D_Struct : CANMsgIface
{
    RDK_2_D_Struct();

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
            } cansignals __attribute__((aligned(4)));
        } canmsg;
    } values;
};

struct RDK_4_D_Struct : CANMsgIface
{
    RDK_4_D_Struct();

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
                unsigned Druck_VL_unkomp_D : 6;
                unsigned ph1 : 2;
                unsigned Druck_VR_unkomp_D : 6;
                unsigned ph2 : 2;
                unsigned Druck_HL_unkomp_D : 6;
                unsigned ph3 : 2;
                unsigned Druck_HR_unkomp_D : 6;
                unsigned ph4 : 2;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct UHR_1_Struct : CANMsgIface // sent by Stoppuhr
{
    UHR_1_Struct();

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

struct UHR_2_Struct : CANMsgIface // sent by Stoppuhr
{
    UHR_2_Struct();

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
//--------------------------------------------------------------------

struct ButtonState_Struct : CANMsgIface // sent by AVR
{
    ButtonState_Struct();

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

                unsigned Button : 8;
                unsigned Light : 8;
                unsigned Joystick : 8;
                unsigned ph4 : 8;
                unsigned ph5 : 8;
                unsigned ph6 : 8;
                unsigned ph7 : 8;
                unsigned ph8 : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct LEDState_Struct : CANMsgIface // to AVR
{
    LEDState_Struct();

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

                unsigned LED : 8;
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

// CAN messages for Continental Temic HLZ FFP gas pedal --------------
struct ANFORDERUNG_FFP_Struct : CANMsgIface
{
    ANFORDERUNG_FFP_Struct();

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
                unsigned CHKSM_RQ_ANG_FFP : 8;
                unsigned ALIV_RQ_ANG_FFP : 4;
                unsigned ANGLE_FFP : 12;
                unsigned FORCE_MAX_FFP : 8;
                unsigned FORCE_MIN_FFP : 8;
                unsigned FUNC_JITTER_FFP : 8;
                unsigned FUNC_FFP : 2;
                unsigned ph1 : 6;
                unsigned STIFFNESS_FFP : 8;
            } cansignals __attribute__((aligned(8)));
        } canmsg;
    } values;
};

struct STATUS_FFP_Struct : CANMsgIface
{
    STATUS_FFP_Struct();

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
                unsigned CHKSM_ST_FFP : 8;
                unsigned ALIV_ST_FFP : 4;
                unsigned ST_FFP_MOD : 2;
                unsigned ST_RCPT_FFP : 2;
                unsigned ANG_AVL_FFP : 12;
                unsigned ST_ANG_AVL_FFP : 2;
                unsigned ph1 : 2;
                unsigned THR_SPAR_FFP : 8;
                unsigned STROM_IST_FFP : 8;
            } cansignals; // alignment has to be a power of 2, so alignment of 6 byte is not possible
        } canmsg;
    } values;
};
//--------------------------------------------------------------------
#endif
