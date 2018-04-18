/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SteeringWheel.h"
#include "InputDevice.h"

#include "CAN.h"
#include "Keyboard.h"
#include "PorscheController.h"

#include "EinspurDynamik.h"
#include "HLRSRealtimeDynamics.h"
#ifdef __XENO__
#include <VehicleUtil/GasPedal.h>
#include <VehicleUtil/KI.h>
#include <VehicleUtil/KLSM.h>
#include <VehicleUtil/Klima.h>
#include <VehicleUtil/Beckhoff.h>
#include <VehicleUtil/IgnitionLock.h>
#ifdef HAVE_CARDYNAMICSCGA
#include "CarDynamicsCGARealtime.h"
#endif
#include "FourWheelDynamicsRealtime.h"
#include "EinspurDynamikRealtime.h"
#endif

#include <OpenVRUI/osg/mathUtils.h>
#ifndef USE_CAR_SOUND
Player *InputDevice::player = NULL;

static void myplayerUnavailableCB()
{
    InputDevice::player = NULL;
}
#endif

//INPUTDEVICE
InputDevice::InputDevice()
{
    pedalA = 0; // Acceleration	[0,1]
    pedalB = 0; // Brake				[0,1]
    pedalC = 0; // Clutch			[0,1]

    steeringWheelAngle = 0; // Wheel angle in Radians
    resetButton = false; // Reset button	[true, false]
    gear = 0; // Present gear	[-1, 0, 1, ...]
    hornButton = false; // Horn button		[true, false]

    automatic = coCoviseConfig::isOn("automaticShift", "COVER.Plugin.SteeringWheel.InputDevice", true, NULL);
    powerShift = automatic;

    velocitySetpoint = coCoviseConfig::getFloat("velocityControl", "COVER.Plugin.SteeringWheel.InputDevice", -1.0) / 3.6;

    angleRatio = coCoviseConfig::getFloat("steeringAngleRatio", "COVER.Plugin.SteeringWheel.InputDevice", 1.0);

    shiftdirection = 0;

    mirrorLightLeft = 0;
    mirrorLightRight = 0;
#ifndef USE_CAR_SOUND
    if (player == NULL)
    {
        player = cover->usePlayer(myplayerUnavailableCB);
        if (player == NULL)
        {
            cover->unusePlayer(myplayerUnavailableCB);
            cover->addPlugin("Vrml97");
            player = cover->usePlayer(myplayerUnavailableCB);
            if (player == NULL)
            {
                cerr << "sorry, no VRML, no Sound support " << endl;
            }
        }
    }
#endif
}

int InputDevice::getType()
{
    return type;
}
std::string InputDevice::getName()
{
    return name;
}
float InputDevice::getAccelerationPedal()
{
    return pedalA;
}
float InputDevice::getBrakePedal()
{
    return pedalB;
}
float InputDevice::getClutchPedal()
{
    return pedalC;
}
float InputDevice::getSteeringWheelAngle()
{
    return steeringWheelAngle;
}
int InputDevice::getGear()
{
    return gear;
}
bool InputDevice::getHornButton()
{
    return hornButton;
}
bool InputDevice::getResetButton()
{
    return resetButton;
}

bool InputDevice::getIsAutomatic()
{
    return automatic;
}

bool InputDevice::getIsPowerShifting()
{
    return powerShift;
}

int InputDevice::getMirrorLightLeft()
{
    return mirrorLightLeft;
}

int InputDevice::getMirrorLightRight()
{
    return mirrorLightRight;
}

//SAITEK
InputDeviceSaitek::InputDeviceSaitek()
    : InputDevice()
{
    type = VEHICLE_INPUT_SAITEK;
    name = "SAITEK";

    oldButton1 = 0;
    oldButton2 = 0;
}

void InputDeviceSaitek::updateInputState()
{
    int button1 = cover->buttons[0][4];
    int button2 = cover->buttons[0][5];
    if (button1 == 1 && !oldButton1)
    {
        if (gear < 5)
            gear++;
    }
    if (button2 == 1 && !oldButton2)
    {
        if (gear > -1)
            gear--;
    }
    oldButton1 = button1;
    oldButton2 = button2;

    steeringWheelAngle = -cover->axes[0][0] * M_PI / 6.0;

    pedalA = (1.0 - cover->axes[0][2]) / 2.0;
    pedalB = (1.0 - cover->axes[0][1]) / 2.0;
    pedalC = 0;

    resetButton = false;
    hornButton = false;
}

//THRUSTMASTER
InputDeviceThrustmaster::InputDeviceThrustmaster()
    : InputDevice()
{
    type = VEHICLE_INPUT_THRUSTMASTER;
    name = "THRUSTMASTER";

    oldButton1 = 0;
    oldButton2 = 0;
}

void InputDeviceThrustmaster::updateInputState()
{
    steeringWheelAngle = -cover->axes[0][0] * M_PI / 6.0;
    if (cover->axes[0][1] < 0.0)
        pedalA = -cover->axes[0][1];
    else if (cover->axes[0][1] > 0.0001)
        pedalB = cover->axes[0][1];

    int button1 = cover->buttons[0][1];
    int button2 = cover->buttons[0][0];
    if (button1 == 1 && !oldButton1)
    {
        if (gear < 6)
            gear++;
    }
    if (button2 == 1 && !oldButton2)
    {
        if (gear > 0)
            gear--;
    }
    oldButton1 = button1;
    oldButton2 = button2;

    resetButton = false;
    hornButton = false;
    pedalC = 0;
}

//MOMO
InputDeviceMomo::InputDeviceMomo()
    : InputDevice()
{
    type = VEHICLE_INPUT_MOMO;
    name = "MOMO";

    oldButton1 = 0;
    oldButton2 = 0;
}

void InputDeviceMomo::updateInputState()
{
    steeringWheelAngle = -cover->axes[0][0] * M_PI / 6.0;
    if (cover->axes[0][1] < 0.0)
        pedalA = -cover->axes[0][1];
    else if (cover->axes[0][1] > 0.0001)
        pedalB = cover->axes[0][1];

    int button1 = cover->buttons[0][1];
    int button2 = cover->buttons[0][0];
    if (cover->buttons[0][2])
        pedalC = 1.0;
    resetButton = cover->buttons[0][3];

    if (button1 == 1 && !oldButton1)
    {
        if (gear < 6)
            gear++;
    }
    if (button2 == 1 && !oldButton2)
    {
        if (gear > 0)
            gear--;
    }
    oldButton1 = button1;
    oldButton2 = button2;
    hornButton = false;
}

//LOGITECH
InputDeviceLogitech::InputDeviceLogitech()
    : InputDevice()
{
    type = VEHICLE_INPUT_LOGITECH;
    name = "LOGITECH";

    oldButton1 = 0;
    oldButton2 = 0;
}

void InputDeviceLogitech::updateInputState()
{
    steeringWheelAngle = -cover->axes[0][0] * M_PI / 6.0;
    if (cover->axes[0][1] < 0.0)
        pedalA = -cover->axes[0][1];
    else if (cover->axes[0][1] > 0.0001)
        pedalB = cover->axes[0][1];

    int button1 = cover->buttons[0][4];
    int button2 = cover->buttons[0][5];
    if (button1 == 1 && !oldButton1)
    {
        if (gear < 4)
            gear++;
    }
    if (button2 == 1 && !oldButton2)
    {
        if (gear > 0)
            gear--;
    }
    oldButton1 = button1;
    oldButton2 = button2;

    resetButton = false;
    pedalC = 0.0;
    hornButton = false;
}

//SITZKISTE
InputDeviceSitzkiste::InputDeviceSitzkiste()
    : InputDevice()
{
    type = VEHICLE_INPUT_SITZKISTE;
    name = "SITZKISTE";

    if (!cover->axes[BRAKE_NUMBER] || !cover->axes[GAS_NUMBER])
    {
        std::cerr << "SteeringWheelPlugin(): InputDeviceSitzkiste(): No joysticks!" << std::endl;
    }
}

void InputDeviceSitzkiste::updateInputState()
{
#ifdef HAVE_PCAN
    if (dynamic_cast<CAN *>(SteeringWheelPlugin::plugin->sitzkiste))
    {
        steeringWheelAngle = SteeringWheelPlugin::plugin->sitzkiste->getAngle();
    }
#endif
    if (cover->axes[BRAKE_NUMBER] && cover->axes[GAS_NUMBER])
    {
        pedalC = (1.0 - cover->axes[BRAKE_NUMBER][2]) / 2.0;
        pedalB = (1.0 - cover->axes[BRAKE_NUMBER][1]) / 2.0;
        pedalA = (1.0 - cover->axes[GAS_NUMBER][3]) / 2.0;
        gear = 0;
        if (cover->buttons[GAS_NUMBER][6] == 1)
        {
            gear = 1;
        }
        if (cover->buttons[GAS_NUMBER][4] == 1)
        {
            gear = 2;
        }
        if (cover->buttons[GAS_NUMBER][8] == 1)
        {
            gear = 3;
        }
        if (cover->buttons[GAS_NUMBER][9] == 1)
        {
            gear = 4;
        }
        if (cover->buttons[GAS_NUMBER][10] == 1)
        {
            gear = 5;
        }
        if (cover->buttons[GAS_NUMBER][5] == 1)
        {
            gear = -1;
        }
    }
    else
    {
        pedalC = 0.0;
        pedalB = 0.0;
        pedalA = 0.0;
        gear = 0;
    }

    if (cover->numJoysticks > 2 && cover->number_buttons[2] > 3 && cover->buttons[2][3] == 1)
        hornButton = true;
    else
        hornButton = false;

    resetButton = false;
}

InputDevicePorscheRealtimeSim::InputDevicePorscheRealtimeSim()
    : InputDevice()
{
    type = VEHICLE_INPUT_PORSCHE_REALTIME_SIM;
    name = "PORSCHE_REALTIME_SIM";

    oldButton1 = 0;
    oldButton2 = 0;
    gear = 0;
}

void InputDevicePorscheRealtimeSim::updateInputState()
{
}

InputDeviceHLRSRealtimeSim::InputDeviceHLRSRealtimeSim()
    : InputDevice()
{
    type = VEHICLE_INPUT_HLRS_REALTIME_SIM;
    name = "HLRS_REALTIME_SIM";

    oldButton1 = 0;
    oldButton2 = 0;
    gear = 0;
}

void InputDeviceHLRSRealtimeSim::updateInputState()
{
    HLRSRealtimeDynamics *dyn = dynamic_cast<HLRSRealtimeDynamics *>(SteeringWheelPlugin::plugin->dynamics);
    if (dyn)
    {
        static int oldButtonState = 0;

        gear = dyn->getGear();
        hornButton = ((dyn->getButtonState() & 1) != 0);
        resetButton = ((dyn->getButtonState() & 2) != 0);

        if ((dyn->getButtonState() & 4) != 0)
        {
        }
        if ((dyn->getButtonState() & 8) != 0)
        {
        }
        if ((dyn->getButtonState() & 16) != 0)
        {
            if (oldButtonState != dyn->getButtonState())
            {
#ifdef USE_CAR_SOUND
                CarSound::instance()->start(CarSound::Ignition);
#else
                anlasserSource->play();
#endif
            }
        }
        oldButtonState = dyn->getButtonState();
    }
}

//PORSCHE_SIM
InputDevicePorscheSim::InputDevicePorscheSim()
    : InputDevice()
{
    type = VEHICLE_INPUT_PORSCHE_SIM;
    name = "PORSCHE_SIMULATOR";

    oldButton1 = 0;
    oldButton2 = 0;
    gear = 0;
}

void InputDevicePorscheSim::updateInputState()
{
    /*
      if(cover->numJoysticks>0)
      {
   //steeringWheelAngle=-cover->axes[0][0]*M_PI;
   steeringWheelAngle=-cover->axes[0][0]*M_PI*2.0;
   pedalB = cover->axes[0][2];
   pedalA = cover->axes[0][1];
   int button1=cover->buttons[0][0];
   int button2=cover->buttons[0][1];

   if(button1==1 && !oldButton1)
   {
   if(gear <5)
   gear++;
   }
   if(button2==1 && !oldButton2)
   {
   if(gear >-1)
   gear--;

   }
   oldButton1 = button1;
   oldButton2 = button2;

   //resetButton = false;
   //pedalC = 0.0;
   //hornButton = false;
    */
    PorscheController *dataController = SteeringWheelPlugin::plugin->dataController;
    if (dataController)
    {
        steeringWheelAngle = dataController->getSteeringWheelAngle() * (-M_PI) * angleRatio;
        pedalA = dataController->getGas();
        pedalB = dataController->getBrake();
        pedalC = dataController->getClutch();

        int button1 = dataController->getGearButtonUp();
        int button2 = dataController->getGearButtonDown();

        int shiftdiff = 0;
        if (button1 == 1 && !oldButton1)
        {
            shiftdiff = 1;
        }
        if (button2 == 1 && !oldButton2)
        {
            shiftdiff = -1;
        }
        oldButton1 = button1;
        oldButton2 = button2;

        if (automatic)
        {
            shiftdirection += shiftdiff;
            if (shiftdirection > 1)
                shiftdirection = 1;
            else if (shiftdirection < -1)
                shiftdirection = -1;
            switch (shiftdirection)
            {
            case 1:
                gear += getAutoGearDiff();
                if (gear < 1)
                    gear = 1;
                break;
            case -1:
                gear = -1;
                break;
            case 0:
                gear = 0;
                break;
            }
        }
        else
        {
            gear += shiftdiff;
        }
        if (gear > 5)
            gear = 5;
        else if (gear < -1)
            gear = -1;

        hornButton = dataController->getHorn();
        resetButton = dataController->getReset();

        mirrorLightLeft = dataController->getMirrorLightLeft();
        mirrorLightRight = dataController->getMirrorLightRight();
    }

    if (velocitySetpoint > 0)
    {
        double pedalAtmp = 0.5 * (velocitySetpoint - SteeringWheelPlugin::plugin->dynamics->getVelocity()) - 0.01 * SteeringWheelPlugin::plugin->dynamics->getAcceleration();
        if (pedalAtmp > 1)
            pedalAtmp = 1;

        if (pedalAtmp < pedalA)
            pedalA = pedalAtmp;

        //std::cerr << "Pedal A: " << pedalA << std::endl;
    }
}

//KEYBOARD
InputDeviceKeyboard::InputDeviceKeyboard()
    : InputDevice()
{
    type = VEHICLE_INPUT_KEYBOARD;
    name = "KEYBOARD";
}

void InputDeviceKeyboard::updateInputState()
{
    Keyboard *keyb = dynamic_cast<Keyboard *>(SteeringWheelPlugin::plugin->sitzkiste);
    if (keyb)
    {
        //std::cerr << "Keyboard in!" << std::endl;
        steeringWheelAngle = keyb->getAngle() * angleRatio;
        pedalC = keyb->getClutch();
        pedalB = keyb->getBrake();
        pedalA = keyb->getGas();

        if (automatic)
        {
            shiftdirection += keyb->getGearDiff();
            if (shiftdirection > 1)
                shiftdirection = 1;
            else if (shiftdirection < -1)
                shiftdirection = -1;
            switch (shiftdirection)
            {
            case 1:
                gear += getAutoGearDiff();
                if (gear < 1)
                    gear = 1;
                break;
            case -1:
                gear = -1;
                break;
            case 0:
                gear = 0;
                break;
            }
        }
        else
        {
            gear += keyb->getGearDiff();
        }
        if (gear > 5)
            gear = 5;
        else if (gear < -1)
            gear = -1;

        hornButton = keyb->getHorn();
        resetButton = keyb->getReset();
    }
}

InputDeviceMotionPlatform::InputDeviceMotionPlatform()
    : InputDevice()
{
    type = VEHICLE_INPUT_MOTIONPLATFORM;
    name = "MOTIONPLATFORM";
    powerShift = true;
    oldParkState = true;
    oldFanButtonState = false;
    ccOn = false;
    ccSpeed = 0;
    ccActive = false;
#ifndef USE_CAR_SOUND
    anlasserSource = NULL;
    Audio *anlasserAudio = new Audio("AnlasserInnen.wav");
    if (player)
    {
        anlasserSource = player->newSource(anlasserAudio);
        if (anlasserSource)
        {
            anlasserSource->setLoop(false);
            anlasserSource->stop();
            anlasserSource->setIntensity(1.0);
        }
    }
#endif

    if (coVRMSController::instance()->isMaster())
    {
#ifdef __XENO__
        //can0 = new XenomaiSocketCan("rtcan0");
        //linMot = new LinearMotorControlTask(*can0);
        //con1 = new CanOpenController("rtcan1");
        //steeringWheel = new XenomaiSteeringWheel(*con1, 1);
        //can3 = new XenomaiSocketCan("rtcan3");

        p_kombi = KI::instance();
        p_klsm = KLSM::instance();
        p_klima = Klima::instance();
        p_beckhoff = Beckhoff::instance();
        //p_brakepedal = BrakePedal::instance();
        p_gaspedal = GasPedal::instance();
        p_ignitionLock = IgnitionLock::instance();
        vehicleUtil = VehicleUtil::instance();

        fprintf(stderr, "\n\n\ninit KI\n");

        p_beckhoff->setDigitalOut(0, 0, false);
        p_beckhoff->setDigitalOut(0, 1, false);
        p_beckhoff->setDigitalOut(0, 2, false);
        p_beckhoff->setDigitalOut(0, 3, false);
        p_beckhoff->setDigitalOut(0, 4, false);
        p_beckhoff->setDigitalOut(0, 5, false);
        p_beckhoff->setDigitalOut(0, 6, false);
        p_beckhoff->setDigitalOut(0, 7, false);

//vehicleUtil->setVehicleState(VehicleUtil::KEYIN);
//vehicleUtil->setVehicleState(VehicleUtil::KEYIN_IGNITED);
//vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTART);
//vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ERUNNING);

/*linMot->setVelocityOne(1000);
      linMot->setVelocityTwo(1000);
      linMot->setVelocityThree(1000);
      linMot->setAccelerationOne(200);
      linMot->setAccelerationTwo(200);
      linMot->setAccelerationThree(200);
      linMot->setPositionOne(linMot->posMiddle);
      linMot->setPositionTwo(linMot->posMiddle);
      linMot->setPositionThree(linMot->posMiddle);
  
      linMot->start();
      linMot->changeStateOne(linMot->statePositioning);
      linMot->changeStateTwo(linMot->statePositioning);
      linMot->changeStateThree(linMot->statePositioning);*/

//p_brakePedal->start();

//steeringWheel->start();

/*p_kombi->initDevice();
      p_klsm->initDevice();
      p_kombi->startDevice();
      p_klsm->startDevice();*/

/*while( //abs(linMot->getPositionOne()-linMot->posMiddle)>10 ||
             abs(linMot->getPositionTwo()-linMot->posMiddle)>10 //||
             abs(linMot->getPositionThree()-linMot->posMiddle)>10 ) {
         usleep(10000);
      }
      linMot->setVelocityOne(5000);
      linMot->setVelocityTwo(5000);
      linMot->setVelocityThree(5000);
      linMot->setAccelerationOne(1000);
      linMot->setAccelerationTwo(1000);
      linMot->setAccelerationThree(1000);*/
#endif
    }

    memset(&sharedState, 0, sizeof(sharedState));

    sharedState.SpoilerState = false;
    sharedState.DamperState = false;
    sharedState.SportMode = false;
    sharedState.PSMState = false;
}

InputDeviceMotionPlatform::~InputDeviceMotionPlatform()
{
    if (coVRMSController::instance()->isMaster())
    {
#ifdef __XENO__
//vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTOP);
//vehicleUtil->setVehicleState(VehicleUtil::KEYOUT);
/*linMot->setVelocityOne(1000);
      linMot->setVelocityTwo(1000);
      linMot->setVelocityThree(1000);
      linMot->setAccelerationOne(200);
      linMot->setAccelerationTwo(200);
      linMot->setAccelerationThree(200);
      linMot->setPositionOne(linMot->posLowerBound);
      linMot->setPositionTwo(linMot->posLowerBound);
      linMot->setPositionThree(linMot->posLowerBound);
      while( //abs(linMot->getPositionOne()-linMot->posLowerBound)>10 ||
             abs(linMot->getPositionTwo()-linMot->posLowerBound)>100 //||
             abs(linMot->getPositionThree()-linMot->posLowerBound)>10 ) {
         usleep(10000);
      }
      linMot->changeStateOne(linMot->stateDisable);
      linMot->changeStateTwo(linMot->stateDisable);
      linMot->changeStateThree(linMot->stateDisable);*/
// Stop the car

//delete can3;
#endif
    }
    //delete anlasserSource;
}

void InputDeviceMotionPlatform::updateInputState()
{
    if (coVRMSController::instance()->isMaster())
    {
#ifdef __XENO__

        float ccGas = 0.0;

        p_klsm->p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_GRA_D = 1;
        if (ccOn)
        {
            if (ccActive)
            {
                float sDiff = ccSpeed - SteeringWheelPlugin::plugin->dynamics->getVelocity();
                sDiff = sDiff / 10.0;
                if (sDiff > 0.8)
                    sDiff = 0.8;
                iDiff += sDiff * 0.01 * cover->frameDuration();
                ccGas = iDiff + sDiff;
                if (ccGas > 0.8)
                    ccGas = 0.8;
            }
            if (ccGas < 0)
                ccGas = 0;
        }
        else
        {
            iDiff = 0;
            p_klsm->p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_GRA_D = 0;
#if 0          
	  fprintf(stderr," GRA_Checksum %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Checksum);
	  fprintf(stderr," GRA_HSchalt %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_HSchalt);
	  fprintf(stderr," GRA_Tip_aus_GRA %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Tip_aus_GRA);
	  fprintf(stderr," GRA_Tip_Verzoegern %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Tip_Verzoegern);
	  fprintf(stderr," GRA_Tip_Beschl %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Tip_Beschl);
	  fprintf(stderr," GRA_Verzoeg %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Verzoeg);
	  fprintf(stderr," GRA_Beschl %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Beschl);
	  fprintf(stderr," GRA_BT_Fehler %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_BT_Fehler);
	  fprintf(stderr," GRA_Tip_Setzen %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Tip_Setzen);
	  fprintf(stderr," GRA_Tip_Wiederauf %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_Tip_Wiederauf);
	  fprintf(stderr," GRA_BZ %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.GRA_BZ);
	  fprintf(stderr," Tip_Down %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.Tip_Down);
	  fprintf(stderr," Tip_Up %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.Tip_Up);
	  fprintf(stderr," ph1 %d ",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.ph1);
	  fprintf(stderr," ph2 %d\n",p_klsm->p_CANProv->LSS_3.values.canmsg.cansignals.ph2);
#endif
        }
        bool ParkState = false;
        EinspurDynamikRealtime *einspur = dynamic_cast<EinspurDynamikRealtime *>(SteeringWheelPlugin::plugin->dynamics);
        FourWheelDynamicsRealtime *fourwheel = dynamic_cast<FourWheelDynamicsRealtime *>(SteeringWheelPlugin::plugin->dynamics);
#ifdef HAVE_CARDYNAMICSCGA
        CarDynamicsCGARealtime *cardynCGA = dynamic_cast<CarDynamicsCGARealtime *>(SteeringWheelPlugin::plugin->dynamics);
#endif

        static bool firstTime = true;
        if (firstTime)
        {
            if (einspur)
            {
                einspur->platformToGround();
            }
            else if (fourwheel)
            {
                fourwheel->platformToGround();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->platformToGround();
            }
#endif
            firstTime = false;
        }

        if (einspur)
        {
            sharedState.steeringWheelAngle = einspur->getSteerWheelAngle();
        }
        else if (fourwheel)
        {
            sharedState.steeringWheelAngle = fourwheel->getSteerWheelAngle();
        }
        if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            float pedal = p_gaspedal->getActualAngle() / 100.0;
            if (ccGas > pedal)
                sharedState.pedalA = ccGas;
            else
                sharedState.pedalA = pedal;
        }
        else
        {
            ParkState = true;
            sharedState.pedalA = 0.0;
        }
        //p_CANProv->GW_D_4.values.canmsg.cansignals.S_Gurt_F_D = 1;

        //std::cerr << "gas: " << sharedState.pedalA << std::endl;
        //sharedState.pedalB = (double)p_brakepedal->getPosition() / (double)p_brakepedal->maxPosition;
        sharedState.pedalB = std::max(0.0, std::min(1.0, ValidateMotionPlatform::instance()->getBrakePedalPosition() * 5.0));
        sharedState.pedalC = 0;

        /*switch(p_beckhoff->getDigitalIn(0)) {
         case 0x83:
          automatic = false;
          p_kombi->set_gwh(4); //M
          default;
         case        
      }*/
        resetButton = p_klsm->getReturnStat();

        /*if(p_beckhoff->getDigitalIn(0,6))
      {
            sharedState.gear = 0;
             ParkState = true;
             p_kombi->set_gwh(0); //P
            automatic = false;
      }*/
        p_kombi->setPetrolLevel(100);
        static int oldState = -1;
        static bool startEngine = false;
        static double startTime = 0.0;

        int currentLeverState = 0;
        if (p_beckhoff->getDigitalIn(0, 0))
        {
            if (p_beckhoff->getDigitalIn(0, 1))
            {
                if (p_beckhoff->getDigitalIn(0, 7))
                {
                    automatic = false;
                    p_kombi->setGearshiftLever(KI::GEAR_M); //M
                    currentLeverState = KI::GEAR_M;
                }
                else if (!p_beckhoff->getDigitalIn(0, 2))
                {
                    automatic = true;
                    p_kombi->setGearshiftLever(KI::GEAR_D); //D
                    currentLeverState = KI::GEAR_D;
                }
            }
            else
            {
                sharedState.gear = -1;
                automatic = false;
                p_kombi->setGearshiftLever(KI::GEAR_R); //R
                currentLeverState = KI::GEAR_R;
            }
        }
        else if (p_beckhoff->getDigitalIn(0, 1))
        {
            sharedState.gear = 0;
            ParkState = true;
            p_kombi->setGearshiftLever(KI::GEAR_P); //P
            currentLeverState = KI::GEAR_P;
            automatic = false;
        }
        else if (p_beckhoff->getDigitalIn(0, 2))
        {
            automatic = false;
            p_kombi->setGearshiftLever(KI::GEAR_N); //N
            currentLeverState = KI::GEAR_N;
            sharedState.gear = 0;
        }

        if (oldState != vehicleUtil->getVehicleState())
        {
            std::cerr << "currentState:" << vehicleUtil->getVehicleState() << std::endl;
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN)
                p_ignitionLock->releaseKey();
        }
        oldState = vehicleUtil->getVehicleState();
        if (p_ignitionLock->getLockState() == IgnitionLock::KEYOUT)
        {
            if (vehicleUtil->getVehicleState() != VehicleUtil::KEYOUT)
            {
                std::cerr << "out" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYOUT);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::ENGINESTOP)
        {
            // key_left (STOP ENGINE)
            std::cerr << "stop" << std::endl;
            vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTOP);
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::IGNITION)
        {
            // key_right1 (IGNITION)
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN)
            {
                std::cerr << "ignite" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN_IGNITED);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::ENGINESTART && currentLeverState == KI::GEAR_P)
        {
            // key_right2 (START ENGINE)
            if (startEngine == false && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_IGNITED)
            {
                startEngine = true;
                startTime = cover->frameTime();
#ifdef USE_CAR_SOUND
                CarSound::instance()->start(CarSound::Ignition);
#else
                anlasserSource->play();
#endif
                std::cerr << "start" << vehicleUtil->getVehicleState() << "  " << VehicleUtil::KEYIN_IGNITED << " " << startEngine << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTART);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::KEYIN)
        {
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYOUT)
            {
                std::cerr << "in" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN);
            }
        }
        if (startEngine && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ESTART && cover->frameTime() > startTime + 0.6)
        {
            vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ERUNNING);
            startEngine = false;
        }

        if (ParkState && sharedState.pedalB > 0.1 && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            p_beckhoff->setDigitalOut(0, 0, true);
        }
        else
        {
            p_beckhoff->setDigitalOut(0, 0, false);
        }
        if (p_klima->getFanButtonStat() && oldFanButtonState == false)
        {
            if (fourwheel)
            {
                fourwheel->centerWheel();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->centerSteeringWheel();
            }
#endif
        }
        oldFanButtonState = p_klima->getFanButtonStat();
        if (ParkState && !oldParkState)
        {
            EinspurDynamikRealtime *einspur = dynamic_cast<EinspurDynamikRealtime *>(SteeringWheelPlugin::plugin->dynamics);
            if (einspur)
            {
                einspur->platformToGround();
            }
            else if (fourwheel)
            {
                fourwheel->platformToGround();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->platformToGround();
            }
#endif
        }
        if (!ParkState && oldParkState)
        {
            EinspurDynamikRealtime *einspur = dynamic_cast<EinspurDynamikRealtime *>(SteeringWheelPlugin::plugin->dynamics);
            if (einspur)
            {
                einspur->platformReturnToAction();
            }
            else if (fourwheel)
            {
                fourwheel->platformReturnToAction();
            }
#ifdef HAVE_CARDYNAMICSCGA
            else if (cardynCGA)
            {
                cardynCGA->platformReturnToAction();
            }
#endif
        }

        oldParkState = ParkState;

        static unsigned char oldButtonState = 0;

        if (p_kombi->getButtonState() & KI::Button_Sport && !(oldButtonState & KI::Button_Sport))
        {
            sharedState.SportMode = (!sharedState.SportMode);
        }
        if (p_kombi->getButtonState() & KI::Button_PSM && !(oldButtonState & KI::Button_PSM))
        {
            sharedState.PSMState = (!sharedState.PSMState);
        }
        if (p_kombi->getButtonState() & KI::Button_Spoiler && !(oldButtonState & KI::Button_Spoiler))
        {
            sharedState.SpoilerState = (!sharedState.SpoilerState);
        }
        if (p_kombi->getButtonState() & KI::Button_Damper && !(oldButtonState & KI::Button_Damper))
        {
            sharedState.DamperState = (!sharedState.DamperState);
            if (fourwheel)
            {
                fourwheel->setSportDamper(sharedState.DamperState);
            }
        }
        oldButtonState = p_kombi->getButtonState();
        unsigned char leds = 0;
        leds |= sharedState.SpoilerState << 0;
        leds |= sharedState.DamperState << 1;
        leds |= sharedState.SportMode << 2;
        leds |= sharedState.PSMState << 3;
        //fprintf(stderr,"leds %d\n" ,leds);
        p_kombi->setLEDState(leds);

        //std::cerr << "p_beckhoff digitil in 0:" << (int)p_beckhoff->getDigitalIn(0) << ", 1: "  << (int)p_beckhoff->getDigitalIn(1) << ", 2: "  << (int)p_beckhoff->getDigitalIn(2) << std::endl;
        if (automatic)
        {
            static double oldShiftTime = 0;
            if (cover->frameTime() - oldShiftTime > 0.2)
            {
                int gearDiff;
                if (sharedState.SportMode)
                {
                    gearDiff = getAutoGearDiff(55, 108);
                }
                else
                {
                    //fprintf(stderr,"gas %f %f \n" ,sharedState.pedalA,SteeringWheelPlugin::plugin->dynamics->getEngineSpeed());

                    gearDiff = getAutoGearDiff(19.0 + (35 * sharedState.pedalA * sharedState.pedalA), 32 + (73 * sharedState.pedalA * sharedState.pedalA));
                }
                sharedState.gear += gearDiff;
                if (sharedState.gear < 0)
                {
                    sharedState.gear = 0;
                }
                else if (sharedState.gear > 5)
                {
                    sharedState.gear = 5;
                }
                if (gearDiff != 0)
                {
                    oldShiftTime = cover->frameTime();
                }
            }
        }
        else
        {
            static int oldStat = 0;
            int stat = p_klsm->getShiftStat();
            if (stat != oldStat)
            {
                sharedState.gear += stat;
                if (sharedState.gear < 0)
                {
                    sharedState.gear = 0;
                }
                else if (sharedState.gear > 5)
                {
                    sharedState.gear = 5;
                }
            }
            oldStat = stat;
        }
        if (p_klsm->getBlinkLeftStat())
        {
            p_kombi->indicator(BlinkerTask::LEFT);
        }
        else if (p_klsm->getBlinkRightStat())
        {
            p_kombi->indicator(BlinkerTask::RIGHT);
        }
        else
        {
            p_kombi->indicator(BlinkerTask::NONE);
        }
        p_kombi->setGear(sharedState.gear);
        /*if(sharedState.gear == -1)
         p_kombi->set_gwh(2);
      else if(sharedState.gear == 0)
         p_kombi->set_gwh(1);
      else
         p_kombi->set_gwh(4); */
        p_kombi->setGear(sharedState.gear);
        p_kombi->setGear(sharedState.gear);
        coVRMSController::instance()->sendSlaves((char *)&sharedState, sizeof(sharedState));
        coVRMSController::instance()->sendSlaves((char *)&resetButton, sizeof(resetButton));

        /*double vehVelTanh = tanh(SteeringWheelPlugin::plugin->dynamics->getVelocity());
      steeringWheel->setRumbleAmplitude(vehVelTanh*SteeringWheelPlugin::plugin->rumbleFactor->getValue()*SteeringWheelPlugin::plugin->dynamics->getRoadType());
      steeringWheel->setDrillElasticity(vehVelTanh);*/

        /*EinspurDynamik* einspur = dynamic_cast<EinspurDynamik*>(SteeringWheelPlugin::plugin->dynamics);
      if(einspur) {
         linMot->setLongitudinalAngle(einspur->getZeta());
         linMot->setLateralAngle(-einspur->getEpsilon());
         //linMot->setLongitudinalAngularVelocity(einspur->getDZeta());
         //linMot->setLateralAngularVelocity(-einspur->getDEpsilon());
         //linMot->setLongitudinalAngularAcceleration(einspur->getDDZeta());
         //linMot->setLateralAngularAcceleration(-einspur->getDDEpsilon());
      }*/
        if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            p_kombi->setSpeed(SteeringWheelPlugin::plugin->dynamics->getVelocity() * 3.6);
            p_kombi->setRevs(SteeringWheelPlugin::plugin->dynamics->getEngineSpeed() * 60);
        }
        else
        {
            p_kombi->setSpeed(0.0);
            p_kombi->setRevs(0.0);
        }
        hornButton = (bool)(p_klsm->getHornStat());
#endif
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&sharedState, sizeof(sharedState));
        coVRMSController::instance()->readMaster((char *)&resetButton, sizeof(resetButton));
    }

    steeringWheelAngle = sharedState.steeringWheelAngle;
    pedalA = sharedState.pedalA;
    pedalB = sharedState.pedalB;
    pedalC = sharedState.pedalC;
    gear = sharedState.gear;
}

InputDevice *InputDevice::instance()
{
    if (inDevice == NULL)
    {
        std::string inDeviceName = coCoviseConfig::getEntry("COVER.Plugin.SteeringWheel.InputDevice");
        if (inDeviceName.empty())
            inDeviceName = "AUTO";
        std::cerr << "inDeviceName: " << inDeviceName << std::endl;
        inDevice = findInputDevice(inDeviceName);
    }
    return inDevice;
}

void InputDevice::destroy()
{
    if (inDevice)
    {
        delete inDevice;
        inDevice = NULL;
    }
}

int InputDevice::autoDetectRetries = 0;
InputDevice *InputDevice::inDevice = NULL;
//Find Input Device with name
InputDevice *InputDevice::findInputDevice(std::string name)
{
    if (name == "SAITEK")
        return (new InputDeviceSaitek());
    else if (name == "THRUSTMASTER")
        return (new InputDeviceThrustmaster());
    else if (name == "MOMO")
        return (new InputDeviceMomo());
    else if (name == "LOGITECH")
        return (new InputDeviceLogitech());
    else if (name == "SITZKISTE")
        return (new InputDeviceSitzkiste());
    else if (name == "PORSCHE_SIM")
        return (new InputDevicePorscheSim());
    else if (name == "PORSCHE_REALTIME_SIM")
        return (new InputDevicePorscheRealtimeSim());
    else if (name == "HLRS_REALTIME_SIM")
        return (new InputDeviceHLRSRealtimeSim());
    else if (name == "KEYBOARD")
        return (new InputDeviceKeyboard());
    else if (name == "MOTIONPLATFORM")
        return (new InputDeviceMotionPlatform());
    else if (name == "AUTO")
        return autodetect();
    else
        return autodetect();
}

//Autodetect routine
InputDevice *InputDevice::autodetect()
{
    static int autoDetectRetries = 0;
    if (autoDetectRetries < 10)
    {
        fprintf(stderr, "cover->numJoysticks %d\n", cover->numJoysticks);
        fprintf(stderr, "cover->number_axes[0] %d\n", cover->number_axes[0]);
        fprintf(stderr, "cover->number_buttons[0] %d\n", cover->number_buttons[0]);
        fprintf(stderr, "cover->number_axes[1] %d\n", cover->number_axes[1]);
        fprintf(stderr, "cover->number_axes[2] %d\n", cover->number_axes[2]);
        autoDetectRetries++;
    }
    if (cover->numJoysticks > 0)
    {
        if ((cover->number_axes[0] == 3) && (cover->number_buttons[0] == 3))
        {
            fprintf(stderr, "PORSCHE_SIMULATOR\n");
            return (new InputDevicePorscheSim());
        }
        /*
      else if((cover->number_axes[0] ==50) && (cover->number_buttons[0] == 3 ))
      {
         inputDevice = VEHICLE_INPUT_PORSCHE_REALTIME_SIM;
         d_inputDevice.set("PORSCHE_REALTIME_SIMULATOR");
         fprintf(stderr,"PORSCHE_REALTIME_SIMULATOR\n");
      }
		*/
        else if (cover->numJoysticks > 1 && cover->number_axes[2] == 1 && SteeringWheelPlugin::plugin->sitzkiste && SteeringWheelPlugin::plugin->sitzkiste->doRun && cover->axes[BRAKE_NUMBER] && cover->axes[GAS_NUMBER])
        {
            //d_inputDevice.set("SITZKISTE");
            return (new InputDeviceSitzkiste());
        }
        else if ((cover->number_axes[0] == 4) && (cover->number_buttons[0] == 11))
        {
            return (new InputDeviceMomo());
        }
        else if ((cover->number_axes[0] == 6) && (cover->number_buttons[0] == 10))
        {
            return (new InputDeviceMomo());
        }
        else if ((cover->number_axes[0] == 2) && (cover->number_buttons[0] == 10))
        {
            return (new InputDeviceMomo());
        }
        else if ((cover->number_axes[0] == 5 || cover->number_axes[0] == 3) && (cover->number_buttons[0] == 10))
        {
            return (new InputDeviceThrustmaster());
        }
        else if (cover->number_axes[0] > 2)
        {
            return (new InputDeviceSaitek());
        }
        else
        {
            return (new InputDeviceLogitech());
        }
    }
    else
    {
        return (new InputDeviceKeyboard());
    }
}

int InputDevice::getAutoGearDiff(float downRPM, float upRPM)
{
    int diff = 0;

    double speed = SteeringWheelPlugin::plugin->dynamics->getEngineSpeed();

    if (gear == 0)
    {
        if (speed < 10)
            diff = -1;
        else if (speed > 20)
            diff = 1;
    }
    else if (gear == 1)
    {
        if (speed < 10)
            diff = -1;
        else if (speed > upRPM)
            diff = 1;
    }
    else
    {
        if (speed < downRPM)
            diff = -1;
        else if (speed > upRPM)
            diff = 1;
    }

    return diff;
}
