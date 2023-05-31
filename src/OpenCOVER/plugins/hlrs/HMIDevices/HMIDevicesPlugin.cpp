/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HMIDevicesPlugin.h"

// constructor, destructor, instance ---------------------------------
HMIDevicesPlugin::HMIDevicesPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    p_ki = KI::instance();
    p_klsm = KLSM::instance();
    p_chrono = Chrono::instance();
    p_seat = Seat::instance();
    p_gaspedal = GasPedal::instance();

    m_startflag = false;
}

// This is called if the plugin is removed at runtime
HMIDevicesPlugin::~HMIDevicesPlugin()
{
    // deletes not necessary any more because of shutDownHandler Class!
    // CANProvider::instance()->shutdown();//
    // delete p_ki;
    // delete p_klsm;
    // delete p_chrono;
    // delete p_seat;
    // delete p_beckhoff;
    // delete CANProvider::instance();

    // Class data members for ignition lock
    delete p_igKeyIsInOutToggleButton;
    delete p_igOnButton;
    delete p_igStartEngButton;
    delete p_igStopEngButton;

    // Class data members for kombined instrument
    delete p_kiWwWarningToggleButton;
    delete p_kiBpadWarningToggleButton;
    delete p_kiIlluToggleButton;
    delete p_kiIndLeftButton;
    delete p_kiIndRightButton;
    delete p_kiBcUpButton;
    delete p_kiBcDownButton;
    delete p_kiBcResetButton;
    delete p_kiBcSetButton;
    delete p_kiDriverBeltToggleButton;
    delete p_kiCoDriverBeltToggleButton;
    delete p_kiFreeButton1;

    delete p_kiPetrolLevelLabel;
    delete p_kiPetrolLevelLabelUnit;
    delete p_kiOilPressureLabel;
    delete p_kiOilPressureLabelUnit;
    delete p_kiOutTempLabel;
    delete p_kiOutTempLabelUnit;
    delete p_kiSpeedLabel;
    delete p_kiSpeedLabelUnit;
    delete p_kiRevLabel;
    delete p_kiRevLabelUnit;
    delete p_kiWaterTempLabel;
    delete p_kiWaterTempLabelUnit;
    delete p_kiOilTempLabel;
    delete p_kiOilTempLabelUnit;
    delete p_kiGearLabel;
    delete p_kiGearLabelUnit;
    delete p_kiGearLeverLabel;
    delete p_kiGearLeverLabelUnit;

    delete p_kiPetrolLevelSlider;
    delete p_kiOilPressureSlider;
    delete p_kiOutTempSlider;
    delete p_kiSpeedSlider;
    delete p_kiRevSlider;
    delete p_kiWaterTempSlider;
    delete p_kiOilTempSlider;
    delete p_kiGearSlider;
    delete p_kiGearLevelSlider;

    // Class data members for stopwatch
    delete p_chronoResetButton;
    delete p_chronoStartButton;
    delete p_chronoStopButton;
    delete p_chronoIlluToggleButton;

    // Class data members for seat
    // 	delete p_seatmove2Mempos1Button;
    // 	delete p_seatsaveMempos1;
    delete p_seatMove2KeyNoButton;
    delete p_seatSave2MemNoButton;
    delete p_seatKeyNo1Button;
    delete p_seatKeyNo2Button;
    delete p_seatKeyNo3Button;
    delete p_seatKeyNo4Button;
    delete p_seatKeyNo5Button;
    delete p_seatKeyNo6Button;

    // Class data members for klsm
    delete p_klsmShiftUpButton;
    ;
    delete p_klsmShiftDownButton;
    delete p_klsmHornButton;
    delete p_klsmReturnButton;
    delete p_klsmFreeButton2;
    delete p_klsmFreeButton3;
    delete p_klsmFreeButton4;
    delete p_klsmFreeButton5;

    // Class data members for gas pedal
    delete p_gasPedLockUnlockButton;

    delete p_gasPedPositionLabel;
    delete p_gasPedPositionProgressBar;

    delete p_gasPedStatusFFPModeLabel;
    delete p_gasPedStatusFFPModeLabelValue;
    delete p_gasPedStatusAvAngleLabel;
    delete p_gasPedStatusAvAngleLabelValue;
    delete p_gasPedThermReserveLabel;
    delete p_gasPedThermReserveLabelValue;
    delete p_gasPedActCurrentLabel;
    delete p_gasPedActCurrentLabelValue;

    delete p_gasPedTargetPosLabel;
    delete p_gasPedMaxTargetForceLabel;
    delete p_gasPedMinTargetForceLabel;
    delete p_gasPedStiffnessLabel;
    delete p_gasPedJitterAmpLabel;
    delete p_gasPedJitterFreqLabel;

    delete p_gasPedTargetPosLabelUnit;
    delete p_gasPedMaxTargetForceLabelUnit;
    delete p_gasPedMinTargetForceLabelUnit;
    delete p_gasPedStiffnessLabelUnit;
    delete p_gasPedJitterAmpLabelUnit;
    delete p_gasPedJitterFreqLabelUnit;

    delete p_gasPedTargetPosSlider;
    delete p_gasPedMaxTargetForceSlider;
    delete p_gasPedMinTargetForceSlider;
    delete p_gasPedStiffnessSlider;
    delete p_gasPedJitterAmpSlider;
    delete p_gasPedJitterFreqSlider;

    delete p_gasPedJitterFormComboBox;

    // Labels for headings
    delete p_kiFrame1Heading;
    delete p_kiFrame2Heading;
    delete p_kiFrame3Heading;
    delete p_kiFrame4Heading;
    delete p_kiFrame5Heading;
    delete p_kiFrame6Heading;
    delete p_kiFrame7Heading;
    delete p_kiFrame8Heading;

    // Frames
    delete p_IgLockFrame;
    delete p_kiFrame1;
    delete p_kiFrame2;
    delete p_chronoFrame;
    delete p_seatFrame1;
    delete p_klsmFrame;
    delete p_gasPedalFrame1;
    delete p_gasPedalFrame2;

    // Tab
    delete p_HMIDevicesTab;
}
//--------------------------------------------------------------------
void HMIDevicesPlugin::preFrame()
{

    //TODO(sebastian): DIESER BLOCK HIER WIRD 2x AUSGEFÜHRT - WARUM ?

    if (cover->frameTime() > m_ts_enginestart + 1.0 && m_startflag == true)
    {
        VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYIN_ERUNNING);
        m_startflag = false;
        cout << "HMIDevicesPlugin::update - Once!" << endl;
    }

    //TODO(sebastian): ANZEIGE GASPEDALWINKEL FUNKT NICHT - CAST ?!

    p_gasPedPositionProgressBar->setValue((int)p_gaspedal->getActualAngle());

    switch (p_gaspedal->getStatusMode())
    {
    case GasPedal::StatusModeReady:
        p_gasPedStatusFFPModeLabelValue->setText("Ready");
        break;
    case GasPedal::StatusModeInit:
        p_gasPedStatusFFPModeLabelValue->setText("Init");
        break;
    case GasPedal::StatusModeError:
        p_gasPedStatusFFPModeLabelValue->setText("Error");
        break;
    default:
        p_gasPedStatusFFPModeLabelValue->setText("Invalid");
    }

    switch (p_gaspedal->getStatusActualAngle())
    {
    case GasPedal::StatusPositionDemandMet:
        p_gasPedStatusAvAngleLabelValue->setText("Demand met");
        break;
    case GasPedal::StatusPositionDemandPartlyMet:
        p_gasPedStatusAvAngleLabelValue->setText("Demand partly met");
        break;
    case GasPedal::StatusPositionDemandNotMet:
        p_gasPedStatusAvAngleLabelValue->setText("Demand not met");
        break;
    default:
        p_gasPedStatusAvAngleLabelValue->setText("Invalid");
    }

    std::stringstream thermalReserveStringstream;
    thermalReserveStringstream << p_gaspedal->getThermalReserve();
    p_gasPedThermReserveLabelValue->setText(thermalReserveStringstream.str().c_str());

    std::stringstream actualCurrentStringstream;
    actualCurrentStringstream << p_gaspedal->getActualCurrent();
    p_gasPedActCurrentLabelValue->setText(actualCurrentStringstream.str().c_str());
}

bool HMIDevicesPlugin::init()
{
    // create tab
    p_HMIDevicesTab = new coTUITab("HMIDevices", coVRTui::instance()->mainFolder->getID());
    p_HMIDevicesTab->setPos(0, 0);

    p_kiFrame1Heading = new coTUILabel("IGNITION LOCK", p_HMIDevicesTab->getID());
    p_kiFrame1Heading->setPos(0, 1);

    // create frames for ECUs
    p_IgLockFrame = new coTUIFrame("Ignition", p_HMIDevicesTab->getID());
    p_IgLockFrame->setPos(0, 2);
    p_IgLockFrame->setShape(0x0006);
    p_IgLockFrame->setStyle(0x0010);

    p_kiFrame2Heading = new coTUILabel("STOPWATCH", p_HMIDevicesTab->getID());
    p_kiFrame2Heading->setPos(1, 1);

    p_chronoFrame = new coTUIFrame("Stopwatch", p_HMIDevicesTab->getID());
    p_chronoFrame->setPos(1, 2);
    p_chronoFrame->setShape(0x0006);
    p_chronoFrame->setStyle(0x0010);

    p_kiFrame3Heading = new coTUILabel("KOMBINED INSTRUMENT GAUGE CONTROL", p_HMIDevicesTab->getID());
    p_kiFrame3Heading->setPos(0, 3);

    p_kiFrame1 = new coTUIFrame("Kombined Instrument Sliders", p_HMIDevicesTab->getID());
    p_kiFrame1->setPos(0, 4);
    p_kiFrame1->setShape(0x0006);
    p_kiFrame1->setStyle(0x0010);

    p_kiFrame4Heading = new coTUILabel("KOMBINED INSTRUMENT FUNCTION CONTROL", p_HMIDevicesTab->getID());
    p_kiFrame4Heading->setPos(1, 3);

    p_kiFrame2 = new coTUIFrame("Kombined Instrument Buttons", p_HMIDevicesTab->getID());
    p_kiFrame2->setPos(1, 4);
    p_kiFrame2->setShape(0x0006);
    p_kiFrame2->setStyle(0x0010);

    p_kiFrame5Heading = new coTUILabel("ELECTRICALLY ADJUSTABLE SEAT CONTROLS", p_HMIDevicesTab->getID());
    p_kiFrame5Heading->setPos(0, 5);

    p_seatFrame1 = new coTUIFrame("Seat", p_HMIDevicesTab->getID());
    p_seatFrame1->setPos(0, 6);
    p_seatFrame1->setShape(0x0006);
    p_seatFrame1->setStyle(0x0010);

    p_kiFrame6Heading = new coTUILabel("KLSM", p_HMIDevicesTab->getID());
    p_kiFrame6Heading->setPos(1, 5);

    p_klsmFrame = new coTUIFrame("KLSM", p_HMIDevicesTab->getID());
    p_klsmFrame->setPos(1, 6);
    p_klsmFrame->setShape(0x0006);
    p_klsmFrame->setStyle(0x0010);

    p_kiFrame8Heading = new coTUILabel("GAS PEDAL CONTROL", p_HMIDevicesTab->getID());
    p_kiFrame8Heading->setPos(0, 7);

    p_gasPedalFrame2 = new coTUIFrame("Gas Pedal Control", p_HMIDevicesTab->getID());
    p_gasPedalFrame2->setPos(0, 8);
    p_gasPedalFrame2->setShape(0x0006);
    p_gasPedalFrame2->setStyle(0x0010);

    p_kiFrame7Heading = new coTUILabel("GAS PEDAL STATUS", p_HMIDevicesTab->getID());
    p_kiFrame7Heading->setPos(1, 7);

    p_gasPedalFrame1 = new coTUIFrame("Gas Pedal Status", p_HMIDevicesTab->getID());
    p_gasPedalFrame1->setPos(1, 8);
    p_gasPedalFrame1->setShape(0x0006);
    p_gasPedalFrame1->setStyle(0x0010);

    //init ECU frames

    initIgLockGui(p_IgLockFrame);
    initKIGuiSliders(p_kiFrame1);
    initKIGuiButtons(p_kiFrame2);
    initChronoGui(p_chronoFrame);
    initSeatGui2(p_seatFrame1);
    initKLSMGui(p_klsmFrame);
    initGasPedalStatusGui(p_gasPedalFrame1);
    initGasPedalControlGui(p_gasPedalFrame2);

    // UWE
    //    p_beckhoff->setDigitalOut(0,0,false);
    //    p_beckhoff->setDigitalOut(0,1,false);
    //    p_beckhoff->setDigitalOut(0,2,false);
    //    p_beckhoff->setDigitalOut(0,3,false);
    //    p_beckhoff->setDigitalOut(0,4,false);
    //    p_beckhoff->setDigitalOut(0,5,false);
    //    p_beckhoff->setDigitalOut(0,6,false);
    //    p_beckhoff->setDigitalOut(0,7,false);

    return (true);
}
//--------------------------------------------------------------------

// private methods ---------------------------------------------------
void HMIDevicesPlugin::initIgLockGui(coTUIFrame *theFrame)
{
    p_igKeyIsInOutToggleButton = new coTUIToggleButton("Key is IN/OUT\n(clamp S)", theFrame->getID());
    p_igKeyIsInOutToggleButton->setEventListener(this);
    p_igKeyIsInOutToggleButton->setPos(0, 0);

    p_igOnButton = new coTUIButton("Ignition\n(clamp 15, X, 15SV)", theFrame->getID());
    p_igOnButton->setEventListener(this);
    p_igOnButton->setPos(1, 0);

    p_igStartEngButton = new coTUIButton("Start Engine\n", theFrame->getID());
    p_igStartEngButton->setEventListener(this);
    p_igStartEngButton->setPos(2, 0);

    p_igStopEngButton = new coTUIButton("Stop Engine\n", theFrame->getID());
    p_igStopEngButton->setEventListener(this);
    p_igStopEngButton->setPos(3, 0);

    testTextEdit = new coTUIEditField("0", theFrame->getID());
    testTextEdit->setEventListener(this);
    testTextEdit->setPos(0, 1);
}

void HMIDevicesPlugin::initKIGuiButtons(coTUIFrame *theFrame)
{
    // Buttons
    p_kiIndLeftButton = new coTUIToggleButton("Indicator left\nON/OFF", theFrame->getID());
    p_kiIndLeftButton->setEventListener(this);
    p_kiIndLeftButton->setPos(0, 0);

    p_kiIndRightButton = new coTUIToggleButton("Indicator right\nON/OFF", theFrame->getID());
    p_kiIndRightButton->setEventListener(this);
    p_kiIndRightButton->setPos(1, 0);

    p_kiWwWarningToggleButton = new coTUIToggleButton("Wash Water Warning ON/OFF\n(Beckhoff)", theFrame->getID());
    p_kiWwWarningToggleButton->setEventListener(this);
    p_kiWwWarningToggleButton->setPos(0, 1);

    p_kiBpadWarningToggleButton = new coTUIToggleButton("Brake Pad Warning ON/OFF\n(Beckhoff)", theFrame->getID());
    p_kiBpadWarningToggleButton->setEventListener(this);
    p_kiBpadWarningToggleButton->setPos(1, 1);

    p_kiIlluToggleButton = new coTUIToggleButton("KI illum.\nON/OFF", theFrame->getID());
    p_kiIlluToggleButton->setEventListener(this);
    p_kiIlluToggleButton->setPos(0, 2);

    p_kiFreeButton1 = new coTUIButton("\n", theFrame->getID());
    p_kiFreeButton1->setEventListener(this);
    p_kiFreeButton1->setPos(1, 2);

    p_kiDriverBeltToggleButton = new coTUIToggleButton("Driver Belt\nWarning ON/OFF", theFrame->getID(), true);
    p_kiDriverBeltToggleButton->setEventListener(this);
    p_kiDriverBeltToggleButton->setPos(0, 3);

    p_kiCoDriverBeltToggleButton = new coTUIToggleButton("Co-driver Belt\nWarning ON/OFF", theFrame->getID(), true);
    p_kiCoDriverBeltToggleButton->setEventListener(this);
    p_kiCoDriverBeltToggleButton->setPos(1, 3);

    p_kiBcUpButton = new coTUIButton("Board Computer\nUP", theFrame->getID());
    p_kiBcUpButton->setEventListener(this);
    p_kiBcUpButton->setPos(0, 4);

    p_kiBcDownButton = new coTUIButton("Board Computer\nDOWN", theFrame->getID());
    p_kiBcDownButton->setEventListener(this);
    p_kiBcDownButton->setPos(0, 5);

    p_kiBcSetButton = new coTUIButton("Board Computer\nSET", theFrame->getID());
    p_kiBcSetButton->setEventListener(this);
    p_kiBcSetButton->setPos(1, 4);

    p_kiBcResetButton = new coTUIButton("Board Computer\nRESET", theFrame->getID());
    p_kiBcResetButton->setEventListener(this);
    p_kiBcResetButton->setPos(1, 5);
}

void HMIDevicesPlugin::initKIGuiSliders(coTUIFrame *theFrame)
{
    // Sliders

    //--------
    p_kiPetrolLevelLabel = new coTUILabel("Petrol Level:", theFrame->getID());
    p_kiPetrolLevelLabel->setPos(0, 1);

    p_kiPetrolLevelSlider = new coTUISlider("Petrol Level", theFrame->getID());
    p_kiPetrolLevelSlider->setEventListener(this);
    p_kiPetrolLevelSlider->setRange(0, 100);
    p_kiPetrolLevelSlider->setPos(1, 1);

    p_kiPetrolLevelLabelUnit = new coTUILabel("[%]", theFrame->getID());
    p_kiPetrolLevelLabelUnit->setPos(3, 1);
    //--------

    //--------
    p_kiOilPressureLabel = new coTUILabel("Oil Pressure:", theFrame->getID());
    p_kiOilPressureLabel->setPos(0, 2);

    p_kiOilPressureSlider = new coTUIFloatSlider("Oil Pressure", theFrame->getID());
    p_kiOilPressureSlider->setEventListener(this);
    p_kiOilPressureSlider->setRange(0.0, 10.16);
    p_kiOilPressureSlider->setPos(1, 2);
    p_kiOilPressureSlider->setValue(6.0);

    p_kiOilPressureLabelUnit = new coTUILabel("[bar]", theFrame->getID());
    p_kiOilPressureLabelUnit->setPos(3, 2);
    //--------

    //--------
    p_kiOutTempLabel = new coTUILabel("Outside Temp:", theFrame->getID());
    p_kiOutTempLabel->setPos(0, 3);

    p_kiOutTempSlider = new coTUIFloatSlider("Outside Temp", theFrame->getID());
    p_kiOutTempSlider->setEventListener(this);
    p_kiOutTempSlider->setRange(-50.0, 77.0);
    p_kiOutTempSlider->setPos(1, 3);
    p_kiOutTempSlider->setValue(25.0);

    p_kiOutTempLabelUnit = new coTUILabel("[C]", theFrame->getID());
    p_kiOutTempLabelUnit->setPos(3, 3);
    //--------

    //--------
    p_kiWaterTempLabel = new coTUILabel("Water Temp:", theFrame->getID());
    p_kiWaterTempLabel->setPos(0, 4);

    p_kiWaterTempSlider = new coTUIFloatSlider("Water Temp", theFrame->getID());
    p_kiWaterTempSlider->setEventListener(this);
    p_kiWaterTempSlider->setRange(-48.0, 142.5);
    p_kiWaterTempSlider->setPos(1, 4);
    p_kiWaterTempSlider->setValue(96.0);

    p_kiWaterTempLabelUnit = new coTUILabel("[C]", theFrame->getID());
    p_kiWaterTempLabelUnit->setPos(3, 4);
    //--------

    //--------

    p_kiOilTempLabel = new coTUILabel("Oil Temp:", theFrame->getID());
    p_kiOilTempLabel->setPos(0, 5);

    p_kiOilTempSlider = new coTUIFloatSlider("Oil Temp", theFrame->getID());
    p_kiOilTempSlider->setEventListener(this);
    p_kiOilTempSlider->setRange(-48.0, 142.5);
    p_kiOilTempSlider->setPos(1, 5);
    p_kiOilTempSlider->setValue(95.0);

    p_kiOilTempLabelUnit = new coTUILabel("[C]", theFrame->getID());
    p_kiOilTempLabelUnit->setPos(3, 5);
    //--------

    //--------
    p_kiGearLabel = new coTUILabel("Gear:", theFrame->getID());
    p_kiGearLabel->setPos(0, 6);

    p_kiGearSlider = new coTUISlider("Gear", theFrame->getID());
    p_kiGearSlider->setEventListener(this);
    p_kiGearSlider->setRange(-1, 5);
    p_kiGearSlider->setPos(1, 6);
    p_kiGearSlider->setValue(1);

    p_kiGearLabelUnit = new coTUILabel("[gear no.]", theFrame->getID());
    p_kiGearLabelUnit->setPos(3, 6);
    //--------

    //--------
    p_kiGearLeverLabel = new coTUILabel("Gear Lever:", theFrame->getID());
    p_kiGearLeverLabel->setPos(0, 7);

    p_kiGearLevelSlider = new coTUISlider("Gear Lever", theFrame->getID());
    p_kiGearLevelSlider->setEventListener(this);
    p_kiGearLevelSlider->setRange(0, 4);
    p_kiGearLevelSlider->setPos(1, 7);
    p_kiGearLevelSlider->setValue(0);

    p_kiGearLeverLabelUnit = new coTUILabel("[leverpos.]", theFrame->getID());
    p_kiGearLeverLabelUnit->setPos(3, 7);
    //--------

    //--------
    p_kiSpeedLabel = new coTUILabel("Speed:", theFrame->getID());
    p_kiSpeedLabel->setPos(0, 8);

    p_kiSpeedSlider = new coTUIFloatSlider("Speed", theFrame->getID());
    p_kiSpeedSlider->setEventListener(this);
    p_kiSpeedSlider->setRange(0.0, 655.32);
    p_kiSpeedSlider->setPos(1, 8);

    p_kiSpeedLabelUnit = new coTUILabel("[km/h]", theFrame->getID());
    p_kiSpeedLabelUnit->setPos(3, 8);
    //--------

    //--------
    p_kiRevLabel = new coTUILabel("Revolutions:", theFrame->getID());
    p_kiRevLabel->setPos(0, 9);

    p_kiRevSlider = new coTUIFloatSlider("Revolutions", theFrame->getID());
    p_kiRevSlider->setEventListener(this);
    p_kiRevSlider->setRange(0.0, 16256.0);
    p_kiRevSlider->setPos(1, 9);

    p_kiRevLabelUnit = new coTUILabel("[rpm]", theFrame->getID());
    p_kiRevLabelUnit->setPos(3, 9);
    //--------
}

void HMIDevicesPlugin::initChronoGui(coTUIFrame *theFrame)
{
    p_chronoResetButton = new coTUIButton("Reset\nStopwatch", theFrame->getID());
    p_chronoResetButton->setEventListener(this);
    p_chronoResetButton->setPos(0, 0);

    p_chronoStartButton = new coTUIButton("Start\nStopwatch", theFrame->getID());
    p_chronoStartButton->setEventListener(this);
    p_chronoStartButton->setPos(1, 0);

    p_chronoStopButton = new coTUIButton("Stop\nStopwatch", theFrame->getID());
    p_chronoStopButton->setEventListener(this);
    p_chronoStopButton->setPos(2, 0);

    p_chronoIlluToggleButton = new coTUIToggleButton("Stopwatch illum.\nON/OFF", theFrame->getID());
    p_chronoIlluToggleButton->setEventListener(this);
    p_chronoIlluToggleButton->setPos(3, 0);
}

// void HMIDevicesPlugin::initSeatGui1(coTUIFrame* theFrame)
// {
//    p_seatMove2Mempos1Button =
//    new coTUIButton("Move seat to\nMemPos1", theFrame->getID());
//    p_seatMove2Mempos1Button->setEventListener(this);
//    p_seatMove2Mempos1Button->setPos(0, 0);
//
//    p_seatMove2Mempos2Button =
//    new coTUIButton("Move seat to\nMemPos2", theFrame->getID());
//    p_seatMove2Mempos2Button->setEventListener(this);
//    p_seatMove2Mempos2Button->setPos(1, 0);
//
//    p_seatSaveMempos1Button =
//    new coTUIButton("Save seat pos to\nMemPos1", theFrame->getID());
//    p_seatSaveMempos1Button->setEventListener(this);
//    p_seatSaveMempos1Button->setPos(0, 1);
//
//    p_seatSaveMempos2Button =
//    new coTUIButton("Save seat pos to\nMemPos2", theFrame->getID());
//    p_seatSaveMempos2Button->setEventListener(this);
//    p_seatSaveMempos2Button->setPos(1, 1);
// }

void HMIDevicesPlugin::initSeatGui2(coTUIFrame *theFrame)
{
    //    p_seatmove2Mempos1Button =
    //    new coTUIButton("Move seat to\nemPos 1", theFrame->getID());
    //    p_seatmove2Mempos1Button->setEventListener(this);
    //    p_seatmove2Mempos1Button->setPos(0,0);
    //
    //    p_seatsaveMempos1 =
    //    new coTUIButton("Save seat pos to\nMemPos 1", theFrame->getID());
    //    p_seatsaveMempos1->setEventListener(this);
    //    p_seatsaveMempos1->setPos(0, 1);

    p_seatMove2KeyNoButton = new coTUIButton("Move seat to\nKeyPos", theFrame->getID());
    p_seatMove2KeyNoButton->setEventListener(this);
    p_seatMove2KeyNoButton->setPos(0, 0);

    p_seatSave2MemNoButton = new coTUIButton("Save seat pos to\nKeyPos", theFrame->getID());
    p_seatSave2MemNoButton->setEventListener(this);
    p_seatSave2MemNoButton->setPos(0, 1);

    p_seatKeyNo1Button = new coTUIToggleButton("1", theFrame->getID());
    p_seatKeyNo1Button->setEventListener(this);
    p_seatKeyNo1Button->setPos(1, 0);

    p_seatKeyNo2Button = new coTUIToggleButton("2", theFrame->getID());
    p_seatKeyNo2Button->setEventListener(this);
    p_seatKeyNo2Button->setPos(1, 1);

    p_seatKeyNo3Button = new coTUIToggleButton("3", theFrame->getID());
    p_seatKeyNo3Button->setEventListener(this);
    p_seatKeyNo3Button->setPos(2, 0);

    p_seatKeyNo4Button = new coTUIToggleButton("4", theFrame->getID());
    p_seatKeyNo4Button->setEventListener(this);
    p_seatKeyNo4Button->setPos(2, 1);

    p_seatKeyNo5Button = new coTUIToggleButton("5", theFrame->getID());
    p_seatKeyNo5Button->setEventListener(this);
    p_seatKeyNo5Button->setPos(3, 0);

    p_seatKeyNo6Button = new coTUIToggleButton("6", theFrame->getID());
    p_seatKeyNo6Button->setEventListener(this);
    p_seatKeyNo6Button->setPos(3, 1);
}

void HMIDevicesPlugin::initKLSMGui(coTUIFrame *theFrame)
{
    p_klsmShiftUpButton = new coTUIButton("Shift\nUP", theFrame->getID());
    p_klsmShiftUpButton->setEventListener(this);
    p_klsmShiftUpButton->setPos(0, 0);

    p_klsmShiftDownButton = new coTUIButton("Shift\nDOWN", theFrame->getID());
    p_klsmShiftDownButton->setEventListener(this);
    p_klsmShiftDownButton->setPos(0, 1);

    p_klsmHornButton = new coTUIButton("HONK!\n", theFrame->getID());
    p_klsmHornButton->setEventListener(this);
    p_klsmHornButton->setPos(1, 0);

    p_klsmReturnButton = new coTUIButton("RETURN!\n", theFrame->getID());
    p_klsmReturnButton->setEventListener(this);
    p_klsmReturnButton->setPos(1, 1);

    p_klsmFreeButton2 = new coTUIButton("\n", theFrame->getID());
    p_klsmFreeButton2->setEventListener(this);
    p_klsmFreeButton2->setPos(2, 0);

    p_klsmFreeButton3 = new coTUIButton("\n", theFrame->getID());
    p_klsmFreeButton3->setEventListener(this);
    p_klsmFreeButton3->setPos(2, 1);

    p_klsmFreeButton4 = new coTUIButton("\n", theFrame->getID());
    p_klsmFreeButton4->setEventListener(this);
    p_klsmFreeButton4->setPos(3, 0);

    p_klsmFreeButton5 = new coTUIButton("\n", theFrame->getID());
    p_klsmFreeButton5->setEventListener(this);
    p_klsmFreeButton5->setPos(3, 1);
}

void HMIDevicesPlugin::initGasPedalStatusGui(coTUIFrame *theFrame)
{
    p_gasPedJitterFormComboBox = new coTUIComboBox("Jitter form", theFrame->getID());
    p_gasPedJitterFormComboBox->setEventListener(this);
    p_gasPedJitterFormComboBox->setPos(0, 0);
    p_gasPedJitterFormComboBox->addEntry("Jitter form Sine");
    p_gasPedJitterFormComboBox->addEntry("Jitter form Saw tooth");
    p_gasPedJitterFormComboBox->addEntry("Jitter form Square");

    p_gasPedLockUnlockButton = new coTUIToggleButton("LOCK/UNLOCK", theFrame->getID(), true);
    p_gasPedLockUnlockButton->setEventListener(this);
    p_gasPedLockUnlockButton->setPos(1, 0);

    p_gasPedPositionLabel = new coTUILabel("Gas pedal position:", theFrame->getID());
    p_gasPedPositionLabel->setPos(0, 1);

    p_gasPedPositionProgressBar = new coTUIProgressBar("Gas pedal position", theFrame->getID());
    p_gasPedPositionProgressBar->setPos(1, 1);

    p_gasPedStatusFFPModeLabel = new coTUILabel("Mode:", theFrame->getID());
    p_gasPedStatusFFPModeLabel->setPos(0, 2);

    p_gasPedStatusAvAngleLabel = new coTUILabel("Position demand:", theFrame->getID());
    p_gasPedStatusAvAngleLabel->setPos(0, 3);

    p_gasPedThermReserveLabel = new coTUILabel("Thermal reserve:", theFrame->getID());
    p_gasPedThermReserveLabel->setPos(0, 4);

    p_gasPedActCurrentLabel = new coTUILabel("Actual current:", theFrame->getID());
    p_gasPedActCurrentLabel->setPos(0, 5);

    p_gasPedStatusFFPModeLabelValue = new coTUIEditField("not applicable", theFrame->getID());
    p_gasPedStatusFFPModeLabelValue->setPos(1, 2);

    p_gasPedStatusAvAngleLabelValue = new coTUIEditField("not applicable", theFrame->getID());
    p_gasPedStatusAvAngleLabelValue->setPos(1, 3);

    p_gasPedThermReserveLabelValue = new coTUIEditField("not applicable", theFrame->getID());
    p_gasPedThermReserveLabelValue->setPos(1, 4);

    p_gasPedActCurrentLabelValue = new coTUIEditField("not applicable", theFrame->getID());
    p_gasPedActCurrentLabelValue->setPos(1, 5);
}

void HMIDevicesPlugin::initGasPedalControlGui(coTUIFrame *theFrame)
{
    //--------
    p_gasPedTargetPosLabel = new coTUILabel("Target angle:", theFrame->getID());
    p_gasPedTargetPosLabel->setPos(0, 1);

    p_gasPedTargetPosSlider = new coTUIFloatSlider("Target angle", theFrame->getID());
    p_gasPedTargetPosSlider->setEventListener(this);
    p_gasPedTargetPosSlider->setRange(0, 100);
    p_gasPedTargetPosSlider->setPos(1, 1);

    p_gasPedTargetPosLabelUnit = new coTUILabel("[%]", theFrame->getID());
    p_gasPedTargetPosLabelUnit->setPos(3, 1);
    //--------

    //--------
    p_gasPedMaxTargetForceLabel = new coTUILabel("Max target force:", theFrame->getID());
    p_gasPedMaxTargetForceLabel->setPos(0, 2);

    p_gasPedMaxTargetForceSlider = new coTUIFloatSlider("Max target forc", theFrame->getID());
    p_gasPedMaxTargetForceSlider->setEventListener(this);
    p_gasPedMaxTargetForceSlider->setRange(0, 100);
    p_gasPedMaxTargetForceSlider->setPos(1, 2);

    p_gasPedMaxTargetForceLabelUnit = new coTUILabel("[%]", theFrame->getID());
    p_gasPedMaxTargetForceLabelUnit->setPos(3, 2);
    //--------

    //--------
    p_gasPedMinTargetForceLabel = new coTUILabel("Min target force:", theFrame->getID());
    p_gasPedMinTargetForceLabel->setPos(0, 3);

    p_gasPedMinTargetForceSlider = new coTUIFloatSlider("Min target forc", theFrame->getID());
    p_gasPedMinTargetForceSlider->setEventListener(this);
    p_gasPedMinTargetForceSlider->setRange(0, 100);
    p_gasPedMinTargetForceSlider->setPos(1, 3);

    p_gasPedMinTargetForceLabelUnit = new coTUILabel("[%]", theFrame->getID());
    p_gasPedMinTargetForceLabelUnit->setPos(3, 3);
    //--------

    //--------
    p_gasPedStiffnessLabel = new coTUILabel("Stiffness:", theFrame->getID());
    p_gasPedStiffnessLabel->setPos(0, 4);

    p_gasPedStiffnessSlider = new coTUIFloatSlider("Stiffness", theFrame->getID());
    p_gasPedStiffnessSlider->setEventListener(this);
    p_gasPedStiffnessSlider->setRange(0, 100);
    p_gasPedStiffnessSlider->setPos(1, 4);

    p_gasPedStiffnessLabelUnit = new coTUILabel("[%]", theFrame->getID());
    p_gasPedStiffnessLabelUnit->setPos(3, 4);
    //--------

    //--------
    p_gasPedJitterAmpLabel = new coTUILabel("Jitter amplitude:", theFrame->getID());
    p_gasPedJitterAmpLabel->setPos(0, 5);

    p_gasPedJitterAmpSlider = new coTUISlider("Jitter amplitude", theFrame->getID());
    p_gasPedJitterAmpSlider->setEventListener(this);
    p_gasPedJitterAmpSlider->setRange(0, 7);
    p_gasPedJitterAmpSlider->setPos(1, 5);

    p_gasPedJitterAmpLabelUnit = new coTUILabel("[10%]", theFrame->getID());
    p_gasPedJitterAmpLabelUnit->setPos(3, 5);
    //--------

    //--------
    p_gasPedJitterFreqLabel = new coTUILabel("Jitter frequency:", theFrame->getID());
    p_gasPedJitterFreqLabel->setPos(0, 6);

    p_gasPedJitterFreqSlider = new coTUISlider("Jitter frequency", theFrame->getID());
    p_gasPedJitterFreqSlider->setEventListener(this);
    p_gasPedJitterFreqSlider->setRange(0, 3);
    p_gasPedJitterFreqSlider->setPos(1, 6);

    p_gasPedJitterFreqLabelUnit = new coTUILabel("[10Hz+n*10Hz]", theFrame->getID());
    p_gasPedJitterFreqLabelUnit->setPos(3, 6);
    //--------
}

// Private methods ---------------------------------------------------

// Event handling ----------------------------------------------------
void HMIDevicesPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    // Events for ignition Lock
    if (p_igKeyIsInOutToggleButton->getState()) // is key in?
    {
        if (tUIItem == p_igOnButton)
        {
            VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYIN_IGNITED);
        }
        else if (tUIItem == p_igStartEngButton)
        {
            m_startflag = true;
            VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYIN_ESTART);
            m_ts_enginestart = cover->frameTime();
        }
        else if (tUIItem == p_igStopEngButton)
        {
            VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYIN_ESTOP);
        }
    }

    // IMPORTANT: NO "else if" here cause p_igKeyIsInOutToggleButton->getState()
    // is true most of the time!!!!!!!

    // Events for KI
    if (tUIItem == p_kiBcUpButton)
    {
        p_ki->doBCUp();
    }
    else if (tUIItem == p_kiBcDownButton)
    {
        p_ki->doBCDown();
    }
    else if (tUIItem == p_kiBcResetButton)
    {
        p_ki->doBCReset();
    }
    else if (tUIItem == p_kiBcSetButton)
    {
        p_ki->doBCSet();
    }
    else if (tUIItem == p_kiFreeButton1)
    {
    }

    //TODO(sebastian): CHRONO LÄSST SICH NICHT STEUERN!

    // Events for chrono
    else if (tUIItem == p_chronoResetButton)
    {
        p_chrono->resetChrono();
    }
    else if (tUIItem == p_chronoStartButton)
    {
        p_chrono->startChrono();
    }
    else if (tUIItem == p_chronoStopButton)
    {
        p_chrono->stopChrono();
    }

    // Events for seat
    //    else if (tUIItem == p_seatmove2Mempos1Button)
    //    {
    //       p_seat->move2Mempos1();
    //    }
    //    else if (tUIItem == p_seatsaveMempos1)
    //    {
    //       p_seat->saveMempos1();
    //    }
    else if (tUIItem == p_seatMove2KeyNoButton)
    {
        p_seat->setKeyNumber(m_seatkeynumber);
        p_seat->move2Memposkey();
    }
    else if (tUIItem == p_seatSave2MemNoButton)
    {
        p_seat->setKeyNumber(m_seatkeynumber);
        p_seat->saveMemposkey();
    }

    // Events for klsm
    else if (tUIItem == p_klsmShiftUpButton)
    {
    }
    else if (tUIItem == p_klsmShiftDownButton)
    {
    }
    else if (tUIItem == p_klsmHornButton)
    {
    }
    else if (tUIItem == p_klsmReturnButton)
    {
    }
    else if (tUIItem == p_klsmFreeButton2)
    {
    }
    else if (tUIItem == p_klsmFreeButton3)
    {
    }
    else if (tUIItem == p_klsmFreeButton4)
    {
    }
    else if (tUIItem == p_klsmFreeButton5)
    {
    }
}

void HMIDevicesPlugin::tabletSelectEvent(coTUIElement *tUIItem)
{
    fprintf(stderr, "tabletSelectEvent\n");
}

void HMIDevicesPlugin::tabletChangeModeEvent(coTUIElement *tUIItem)
{
    fprintf(stderr, "tabletChangeModeEvent\n");
}

void HMIDevicesPlugin::tabletCurrentEvent(coTUIElement *tUIItem)
{
    fprintf(stderr, "tabletCurrentEvent\n");
}

void HMIDevicesPlugin::tabletReleaseEvent(coTUIElement *tUIItem)
{
    // Events for KI sliders
    if (tUIItem == p_kiPetrolLevelSlider)
    {
        p_ki->setPetrolLevel(p_kiPetrolLevelSlider->getValue());
    }
    else if (tUIItem == p_kiOilPressureSlider)
    {
        p_ki->setOilPressure(p_kiOilPressureSlider->getValue());
    }
    else if (tUIItem == p_kiOutTempSlider)
    {
        p_ki->setOutsideTemp(p_kiOutTempSlider->getValue());
    }
    else if (tUIItem == p_kiSpeedSlider)
    {
        p_ki->setSpeed(p_kiSpeedSlider->getValue());
    }
    else if (tUIItem == p_kiRevSlider)
    {
        p_ki->setRevs(p_kiRevSlider->getValue());
    }
    else if (tUIItem == p_kiWaterTempSlider)
    {
        p_ki->setWaterTemp(p_kiWaterTempSlider->getValue());
    }
    else if (tUIItem == p_kiOilTempSlider)
    {
        p_ki->setOiltemp(p_kiOilTempSlider->getValue());
    }
    else if (tUIItem == p_kiGearSlider)
    {
        p_ki->setGear(p_kiGearSlider->getValue());
    }
    else if (tUIItem == p_kiGearLevelSlider)
    {
        p_ki->setGearshiftLever(p_kiGearLevelSlider->getValue());
    }

    // Events for gas pedal sliders
    else if (tUIItem == p_gasPedTargetPosSlider)
    {
        p_gaspedal->setTargetValueAngle(p_gasPedTargetPosSlider->getValue());
    }
    else if (tUIItem == p_gasPedMaxTargetForceSlider)
    {
        p_gaspedal->setTargetValueMaxTargetForce(p_gasPedMaxTargetForceSlider->getValue());
    }
    else if (tUIItem == p_gasPedMinTargetForceSlider)
    {
        p_gaspedal->setTargetValueMinTargetForce(p_gasPedMinTargetForceSlider->getValue());
    }
    else if (tUIItem == p_gasPedStiffnessSlider)
    {
        p_gaspedal->setTargetValueStiffness(p_gasPedStiffnessSlider->getValue());
    }
    else if (tUIItem == p_gasPedJitterAmpSlider)
    {
        p_gaspedal->setJitterAmplitude(p_gasPedJitterAmpSlider->getValue());
        //TODO(sebastian):
    }
    else if (tUIItem == p_gasPedJitterFreqSlider)
    {
        p_gaspedal->setJitterFrequency(p_gasPedJitterFreqSlider->getValue());
        //TODO(sebastian):
    }
}

void HMIDevicesPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == testTextEdit)
    {
        int val = 0;
        sscanf(testTextEdit->getText().c_str(), "%d", &val);
        CANProvider::instance()->GW_D_4.values.canmsg.cansignals.ph8 = val;
    }
    // Events for KI toggle buttons
    if (tUIItem == p_igKeyIsInOutToggleButton)
    {
        // is ToggleButton active (=true)?
        if (p_igKeyIsInOutToggleButton->getState())
        {
            VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYIN);
        }
        else
            VehicleUtil::instance()->setVehicleState(VehicleUtil::KEYOUT);
    }
    else if (tUIItem == p_kiWwWarningToggleButton)
    {
        p_ki->toggleWashWaterWarning();
    }
    else if (tUIItem == p_kiBpadWarningToggleButton)
    {
        p_ki->toggleBrakePadWarning();
    }
    else if (tUIItem == p_kiIlluToggleButton)
    {
        // is ToggleButton active (=true)?
        if (p_kiIlluToggleButton->getState())
        {
            p_ki->activateIllum();
        }
        else
            p_ki->deactivateIllum();
    }
    else if (tUIItem == p_kiIndLeftButton)
    {
    }
    else if (tUIItem == p_kiIndRightButton)
    {
    }
    else if (tUIItem == p_kiDriverBeltToggleButton)
    {
        if (p_kiDriverBeltToggleButton->getState())
        {
            p_ki->activateDriverBeltWarning();
        }
        else
            p_ki->deactivateDriverBeltWarning();
    }
    else if (tUIItem == p_kiCoDriverBeltToggleButton)
    {
        if (p_kiCoDriverBeltToggleButton->getState())
        {
            p_ki->activateCoDriverBeltWarning();
        }
        else
            p_ki->deactivateCoDriverBeltWarning();
    }

    // Events for chrono toggle buttons
    else if (tUIItem == p_chronoIlluToggleButton)
    {
    }

    // Events for seat toggle buttons
    else if (tUIItem == p_seatKeyNo1Button)
    {
        if (p_seatKeyNo1Button->getState())
        {
            m_seatkeynumber = 1;
            p_seatKeyNo2Button->setState(false);
            p_seatKeyNo3Button->setState(false);
            p_seatKeyNo4Button->setState(false);
            p_seatKeyNo5Button->setState(false);
            p_seatKeyNo6Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }
    else if (tUIItem == p_seatKeyNo2Button)
    {
        if (p_seatKeyNo2Button->getState())
        {
            m_seatkeynumber = 2;
            p_seatKeyNo1Button->setState(false);
            p_seatKeyNo3Button->setState(false);
            p_seatKeyNo4Button->setState(false);
            p_seatKeyNo5Button->setState(false);
            p_seatKeyNo6Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }
    else if (tUIItem == p_seatKeyNo3Button)
    {
        if (p_seatKeyNo3Button->getState())
        {
            m_seatkeynumber = 3;
            p_seatKeyNo1Button->setState(false);
            p_seatKeyNo2Button->setState(false);
            p_seatKeyNo4Button->setState(false);
            p_seatKeyNo5Button->setState(false);
            p_seatKeyNo6Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }
    else if (tUIItem == p_seatKeyNo4Button)
    {
        if (p_seatKeyNo4Button->getState())
        {
            m_seatkeynumber = 4;
            p_seatKeyNo1Button->setState(false);
            p_seatKeyNo2Button->setState(false);
            p_seatKeyNo3Button->setState(false);
            p_seatKeyNo5Button->setState(false);
            p_seatKeyNo6Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }
    else if (tUIItem == p_seatKeyNo5Button)
    {
        if (p_seatKeyNo5Button->getState())
        {
            m_seatkeynumber = 5;
            p_seatKeyNo1Button->setState(false);
            p_seatKeyNo2Button->setState(false);
            p_seatKeyNo3Button->setState(false);
            p_seatKeyNo4Button->setState(false);
            p_seatKeyNo6Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }
    else if (tUIItem == p_seatKeyNo6Button)
    {
        if (p_seatKeyNo6Button->getState())
        {
            m_seatkeynumber = 6;
            p_seatKeyNo1Button->setState(false);
            p_seatKeyNo2Button->setState(false);
            p_seatKeyNo3Button->setState(false);
            p_seatKeyNo4Button->setState(false);
            p_seatKeyNo5Button->setState(false);
        }
        else
            m_seatkeynumber = 0;
    }

    // Events for gas pedal toggle buttons

    else if (tUIItem == p_gasPedLockUnlockButton)
    {
        if (p_gasPedLockUnlockButton->getState())
        {
            p_gaspedal->unlockGasPedal();
        }
        else
            p_gaspedal->lockGasPedal();
    }

    // Events for gas pedal combo box
    else if (tUIItem == p_gasPedJitterFormComboBox)
    {
        switch (p_gasPedJitterFormComboBox->getSelectedEntry())
        {
        case 0:
        {
            p_gaspedal->setJitterSignalForm(GasPedal::DemandJitterSignalFormSine);
            break;
        }
        case 1:
        {
            p_gaspedal->setJitterSignalForm(GasPedal::DemandJitterSignalFormSawTooth);
            break;
        }
        case 2:
        {
            p_gaspedal->setJitterSignalForm(GasPedal::DemandJitterSignalFormSquare);
            break;
        }
        default:
        {
            cout << "HMIDevicesPlugin::tabletEvent - p_gasPedJitterFormComboBox error" << endl;
        }
        }
    }
}

//--------------------------------------------------------------------

COVERPLUGIN(HMIDevicesPlugin)
