/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __KI_h
#define __KI_h

//--------------------------------------------------------------------
// PROJECT        KI                                       © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    This singleton class enables you to control the
//                Porsche kombined instrument using beckhoff and CAN
//                signals
//
// CREATED        15-May-09, S. Franz
// MODIFIED       17-July-09, S. Franz
//                - Application of HLRS style guide
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "HMIDeviceIface.h"
#include "Beckhoff.h"
#include "XenomaiEvent.h"
#include "fasiUpdateManager.h"
#include <math.h>
#include <time.h>

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class VEHICLEUTILEXPORT BlinkerTask : public XenomaiTask
{
public:
    enum blinkState
    {
        NONE = -1,
        LEFT = 0,
        RIGHT = 1,
        BOTH = 2
    };
    BlinkerTask(CANProvider *cp);
    ~BlinkerTask();
    void bstart();
    void bsuspend();
    void bresume();
    void setState(blinkState bs);

protected:
    void run(); // override Xenomai Method
private:
    XenomaiEvent *event;
    CANProvider *pCANprovider;
    blinkState state;
    RTIME changeTime;
    int blinkCounter;
    blinkState currentState;
    bool blink;
};
//--------------------------------------------------------------------

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class VEHICLEUTILEXPORT KI : public HMIDeviceIface, public fasiUpdateable
{
public:
    enum gearshiftLeverState
    {
        GEAR_P = 0,
        GEAR_N = 1,
        GEAR_R = 2,
        GEAR_D = 3,
        GEAR_M = 4
    };
    enum LEDBits
    {
        LED_Spoiler = 1,
        LED_Damper = 2,
        LED_Sport = 4,
        LED_PSM = 8,
    };
    enum ButtonBits
    {
        Button_Spoiler = 1,
        Button_Damper = 2,
        Button_Sport = 4,
        Button_PSM = 8,
    };
    virtual ~KI();

    // public methods to control the KI's analog inputs with Beckhoff
    void toggleWashWaterWarning();
    void toggleBrakePadWarning();
    void setPetrolLevel(int percent);

    // public methods to control the KI by CAN
    void setOilPressure(double oilPbar);
    void setOutsideTemp(double outTempCelcius);
    void setSpeed(double speedKmh);
    void setRevs(double rpmUmin);
    void setWaterTemp(double wTempCelsius);
    void setOiltemp(double oilTCelcius);
    void setTime();
    void doBCSet();
    void doBCReset();
    void doBCUp();
    void doBCDown();
    void activateDriverBeltWarning();
    void activateCoDriverBeltWarning();
    void deactivateDriverBeltWarning();
    void deactivateCoDriverBeltWarning();
    void setGearshiftLever(uint leverpos);
    void setGear(int gearnumber);
    void activateIllum();
    void deactivateIllum();
    void indicator(BlinkerTask::blinkState state);

    // HMIDeviceIface methods
    bool keyInLock(int keynumber);
    bool ignitionOn();
    bool initDevice();
    bool startDevice();
    bool stopDevice();
    bool shutDownDevice();

    static KI *instance(); // singleton

    void setLEDState(unsigned char ledState);
    unsigned char getLEDState();
    unsigned char getButtonState();
    unsigned char getLightState();
    unsigned char getJoystickState();

    // TODO (sebastian): these fcts are not in use at the moment
    // void Clutch();
    // void ParkingAid();
    // void IndicateLeftTip();
    // void IndicateRightTip());
    // void SetTime(int, int, int);
    // void SetDimming(int);

protected:
    KI();

    static KI *p_KI;

private:
    // private methods to control the KI by CAN
    void setTirepressureDiff(double pressurediffBar);
    void setTirepressure(double bar);

    int round(double);
    int calcRawValue(double physvalue, double offset, double factor);

    bool update();

    double m_tstamp_pcmtime;
    BlinkerTask *bt;
};
//--------------------------------------------------------------------

#endif
