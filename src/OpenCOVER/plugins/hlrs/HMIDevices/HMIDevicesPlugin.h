/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _HMIDevicesPlugin_H
#define _HMIDevicesPlugin_H

//--------------------------------------------------------------------
// PROJECT        HMIDevicesPlugin                         © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    This COVISE plugin uses the (singleton) classes of
//                the KI, of the KLSM, of the chrono as well as of the
//                seat and enables you to control these ECUs with a
//                TabletUI GUI for test purposes
//
// CREATED        15-May-09, S. Franz
// MODIFIED       17-July-09, S. Franz
//                Restructured the GUI and added new elements to
//                control chrono and seat ECU functions
//                21-July-09, S. Franz
//                Added deletes for TUI Elements
//                30-July-09, S. Franz
//                - Updated comments / documentation
//                - Added gas pedal
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/RenderObject.h>
#include <cover/coVRTui.h>

#include <sstream>

#include "KI.h" // Kombined Instrument
#include "KLSM.h" // Drop-arm Switch
#include "Chrono.h" // Stopwatch
#include "Seat.h" // Seat (electrical adjustable)
#include "GasPedal.h" // Conti Teves Gas Pedal
#include "BrakePedal.h" // ??? - UWE

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class HMIDevicesPlugin : public coVRPlugin, public coTUIListener
{
public:
    HMIDevicesPlugin();
    ~HMIDevicesPlugin();

    // initialization - coVRPlugin IFace
    bool init();

    // will be called in PreFrame - oVRPlugin IFace
    void preFrame();

private:
    // TUI EventHandlers
    void tabletPressEvent(coTUIElement *tUIItem);
    void tabletSelectEvent(coTUIElement *tUIItem);
    void tabletChangeModeEvent(coTUIElement *tUIItem);
    void tabletCurrentEvent(coTUIElement *tUIItem);
    void tabletReleaseEvent(coTUIElement *tUIItem);
    void tabletEvent(coTUIElement *tUItem);

    void initIgLockGui(coTUIFrame *theFrame);
    void initKIGuiSliders(coTUIFrame *theFrame);
    void initKIGuiButtons(coTUIFrame *theFrame);
    void initChronoGui(coTUIFrame *theFrame);
    //void initSeatGui1 (coTUIFrame* theFrame);
    void initSeatGui2(coTUIFrame *theFrame);
    void initKLSMGui(coTUIFrame *theFrame);
    void initGasPedalStatusGui(coTUIFrame *theFrame);
    void initGasPedalControlGui(coTUIFrame *theFrame);
    void initTestGUI(coTUIFrame *theFrame);

    double m_ts_enginestart; // time stamp for engine start delay
    bool m_startflag; // flag for engine start delay

    // Used HMIDevices
    KI *p_ki;
    KLSM *p_klsm;
    Chrono *p_chrono;
    Seat *p_seat;
    GasPedal *p_gaspedal;

    // etc
    int m_seatkeynumber;

    coTUIEditField *testTextEdit;
    //Plugin tab
    coTUITab *p_HMIDevicesTab;

    // Labels for headings
    coTUILabel *p_kiFrame1Heading;
    coTUILabel *p_kiFrame2Heading;
    coTUILabel *p_kiFrame3Heading;
    coTUILabel *p_kiFrame4Heading;
    coTUILabel *p_kiFrame5Heading;
    coTUILabel *p_kiFrame6Heading;
    coTUILabel *p_kiFrame7Heading;
    coTUILabel *p_kiFrame8Heading;

    // Frames
    coTUIFrame *p_IgLockFrame;
    coTUIFrame *p_kiFrame1;
    coTUIFrame *p_kiFrame2;
    coTUIFrame *p_chronoFrame;
    coTUIFrame *p_seatFrame1;
    coTUIFrame *p_klsmFrame;
    coTUIFrame *p_gasPedalFrame1;
    coTUIFrame *p_gasPedalFrame2;

    // Class data members for ignition lock
    coTUIToggleButton *p_igKeyIsInOutToggleButton;
    coTUIButton *p_igOnButton;
    coTUIButton *p_igStartEngButton;
    coTUIButton *p_igStopEngButton;

    // Class data members for kombined instrument
    coTUIToggleButton *p_kiWwWarningToggleButton;
    coTUIToggleButton *p_kiBpadWarningToggleButton;
    coTUIToggleButton *p_kiIlluToggleButton;
    coTUIToggleButton *p_kiIndLeftButton;
    coTUIToggleButton *p_kiIndRightButton;
    coTUIButton *p_kiBcUpButton;
    coTUIButton *p_kiBcDownButton;
    coTUIButton *p_kiBcResetButton;
    coTUIButton *p_kiBcSetButton;
    coTUIToggleButton *p_kiDriverBeltToggleButton;
    coTUIToggleButton *p_kiCoDriverBeltToggleButton;
    coTUIButton *p_kiFreeButton1;

    coTUILabel *p_kiPetrolLevelLabel;
    coTUILabel *p_kiPetrolLevelLabelUnit;
    coTUILabel *p_kiOilPressureLabel;
    coTUILabel *p_kiOilPressureLabelUnit;
    coTUILabel *p_kiOutTempLabel;
    coTUILabel *p_kiOutTempLabelUnit;
    coTUILabel *p_kiSpeedLabel;
    coTUILabel *p_kiSpeedLabelUnit;
    coTUILabel *p_kiRevLabel;
    coTUILabel *p_kiRevLabelUnit;
    coTUILabel *p_kiWaterTempLabel;
    coTUILabel *p_kiWaterTempLabelUnit;
    coTUILabel *p_kiOilTempLabel;
    coTUILabel *p_kiOilTempLabelUnit;
    coTUILabel *p_kiGearLabel;
    coTUILabel *p_kiGearLabelUnit;
    coTUILabel *p_kiGearLeverLabel;
    coTUILabel *p_kiGearLeverLabelUnit;

    coTUISlider *p_kiPetrolLevelSlider;
    coTUIFloatSlider *p_kiOilPressureSlider;
    coTUIFloatSlider *p_kiOutTempSlider;
    coTUIFloatSlider *p_kiSpeedSlider;
    coTUIFloatSlider *p_kiRevSlider;
    coTUIFloatSlider *p_kiWaterTempSlider;
    coTUIFloatSlider *p_kiOilTempSlider;
    coTUISlider *p_kiGearSlider;
    coTUISlider *p_kiGearLevelSlider;

    // Class data members for stopwatch
    coTUIButton *p_chronoResetButton;
    coTUIButton *p_chronoStartButton;
    coTUIButton *p_chronoStopButton;
    coTUIToggleButton *p_chronoIlluToggleButton;

    // Class data members for seat
    //coTUIButton* p_seatmove2Mempos1Button;
    //coTUIButton* p_seatsaveMempos1;
    coTUIButton *p_seatMove2KeyNoButton;
    coTUIButton *p_seatSave2MemNoButton;
    coTUIToggleButton *p_seatKeyNo1Button;
    coTUIToggleButton *p_seatKeyNo2Button;
    coTUIToggleButton *p_seatKeyNo3Button;
    coTUIToggleButton *p_seatKeyNo4Button;
    coTUIToggleButton *p_seatKeyNo5Button;
    coTUIToggleButton *p_seatKeyNo6Button;

    // Class data members for klsm
    coTUIButton *p_klsmShiftUpButton;
    ;
    coTUIButton *p_klsmShiftDownButton;
    coTUIButton *p_klsmHornButton;
    coTUIButton *p_klsmReturnButton;
    coTUIButton *p_klsmFreeButton2;
    coTUIButton *p_klsmFreeButton3;
    coTUIButton *p_klsmFreeButton4;
    coTUIButton *p_klsmFreeButton5;

    // Class data members for gas pedal
    coTUIToggleButton *p_gasPedLockUnlockButton;

    coTUILabel *p_gasPedPositionLabel;
    coTUIProgressBar *p_gasPedPositionProgressBar;

    coTUILabel *p_gasPedStatusFFPModeLabel;
    coTUIEditField *p_gasPedStatusFFPModeLabelValue;
    coTUILabel *p_gasPedStatusAvAngleLabel;
    coTUIEditField *p_gasPedStatusAvAngleLabelValue;
    coTUILabel *p_gasPedThermReserveLabel;
    coTUIEditField *p_gasPedThermReserveLabelValue;
    coTUILabel *p_gasPedActCurrentLabel;
    coTUIEditField *p_gasPedActCurrentLabelValue;

    coTUILabel *p_gasPedTargetPosLabel;
    coTUILabel *p_gasPedMaxTargetForceLabel;
    coTUILabel *p_gasPedMinTargetForceLabel;
    coTUILabel *p_gasPedStiffnessLabel;
    coTUILabel *p_gasPedJitterAmpLabel;
    coTUILabel *p_gasPedJitterFreqLabel;

    coTUILabel *p_gasPedTargetPosLabelUnit;
    coTUILabel *p_gasPedMaxTargetForceLabelUnit;
    coTUILabel *p_gasPedMinTargetForceLabelUnit;
    coTUILabel *p_gasPedStiffnessLabelUnit;
    coTUILabel *p_gasPedJitterAmpLabelUnit;
    coTUILabel *p_gasPedJitterFreqLabelUnit;

    coTUIFloatSlider *p_gasPedTargetPosSlider;
    coTUIFloatSlider *p_gasPedMaxTargetForceSlider;
    coTUIFloatSlider *p_gasPedMinTargetForceSlider;
    coTUIFloatSlider *p_gasPedStiffnessSlider;
    coTUISlider *p_gasPedJitterAmpSlider;
    coTUISlider *p_gasPedJitterFreqSlider;

    coTUIComboBox *p_gasPedJitterFormComboBox;
};
//--------------------------------------------------------------------

#endif
