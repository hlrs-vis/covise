/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KI.h"
#include <sys/time.h>

// KI ////////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
KI *KI::p_KI = NULL;

// constructor, destructor, instance ---------------------------------
KI::KI()
{
    m_tstamp_pcmtime = 0.0;
    m_state = KI::DEFAULT;
    // if not initialized, there will be a kombined instrument error
    setPetrolLevel(100);
    setOutsideTemp(25.0);
    bt = new BlinkerTask(p_CANProv);
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_Getriebe_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_ACD_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_Verdeck_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_RDK_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_Kombi_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_PCM_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_BC_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_GRA_D = 1;
    p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_ALR_D = 1;
}

KI::~KI()
{
    p_KI = NULL;
    delete bt;
}

// singleton
KI *KI::instance()
{
    if (p_KI == NULL)
    {
        p_KI = new KI();
    }
    return p_KI;
}
//--------------------------------------------------------------------

// public methods to control the KI's analog inputs with Beckhoff ----
void KI::toggleWashWaterWarning()
{
    // Beckhoff
    return;
}

void KI::toggleBrakePadWarning()
{
    // Beckhoff
    return;
}

void KI::setPetrolLevel(int petrollevelpercent)
{
    float petrollevelraw;
    double physmax = 100;
    double physmin = 0;

    if (petrollevelpercent > physmax || petrollevelpercent < physmin)
    {
        std::cerr << "KI::setPetrolLevel: value for petrollevelpercent higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        // 4.9 V ~ leer
        // 9.0 V ~ voll

        // calculate petrol level in raw format
        petrollevelraw = 3 + ((petrollevelpercent / 100) * 6.0);

        // set beckhoff analog for for petrol level
        Beckhoff::instance()->setAnalogOut(0, 0, petrollevelraw);
    }
}
//--------------------------------------------------------------------

// public methods to control the KI by CAN ---------------------------
void KI::setOilPressure(double oilPbar)
{
    int oilpressureraw;
    double offset = 0; // from Porsche DBC
    double factor = 0.04; // from Porsche DBC
    double physmax = 10.16; // from Porsche DBC
    double physmin = 0; // from Porsche DBC

    if (oilPbar > physmax || oilPbar < physmin)
    {
        std::cerr << "KI::setOilPressure: value for oilPbar higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        // calculate oiltemp in raw format and round to int
        oilpressureraw = calcRawValue(oilPbar, offset, factor);

        // set signal for oilpressure
        p_CANProv->MOTOR_4_D.values.canmsg.cansignals.Oeldruck_D = oilpressureraw;
    }
}

void KI::setOutsideTemp(double outTempCelcius)
{
    int atempraw;
    double offset = -50.0; // from Porsche DBC
    double factor = 0.5; // from Porsche DBC
    double physmax = 77.0; // from Porsche DBC
    double physmin = -50.0; // from Porsche DBC

    if (outTempCelcius > physmax || outTempCelcius < physmin)
    {
        std::cerr << "KI::setOutsideTemp: value for outTempCelcius higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        atempraw = calcRawValue(outTempCelcius, offset, factor);
        p_CANProv->GW_D_1.values.canmsg.cansignals.Atemp_ungef_D = atempraw;
    }
}

void KI::setSpeed(double speedKmh)
{
    int speedraw;
    double offset = 0; // from Porsche DBC
    double factor = 0.02; // from Porsche DBC
    double physmax = 655.32; // from Porsche DBC
    double physmin = 0; // from Porsche DBC

    if (fabs(speedKmh) > physmax || fabs(speedKmh) < physmin)
    {
        //std::cerr << "KI::set_speed: value for speedKmh higher than "
        //<< physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        speedraw = calcRawValue(fabs(speedKmh), offset, factor);
        p_CANProv->BREMSE_2_D.values.canmsg.cansignals.RG_VL_D = speedraw;
        p_CANProv->BREMSE_2_D.values.canmsg.cansignals.RG_HL_D = speedraw;
        p_CANProv->BREMSE_2_D.values.canmsg.cansignals.RG_VR_D = speedraw;
        p_CANProv->BREMSE_2_D.values.canmsg.cansignals.RG_HR_D = speedraw;
    }
}

void KI::setRevs(double rpmUmin)
{
    int rpmraw;
    double offset = 0; // from Porsche DBC
    double factor = 0.25; // from Porsche DBC
    double physmax = 16256.0; // from Porsche DBC
    double physmin = 0; // from Porsche DBC

    if (rpmUmin > physmax || rpmUmin < physmin)
    {
        std::cerr << "KI::setRevs: value for rpmUmin higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        rpmraw = calcRawValue(rpmUmin, offset, factor);
        p_CANProv->GW_D_2.values.canmsg.cansignals.nmot_D = rpmraw;
    }
}

void KI::setWaterTemp(double wTempCelsius)
{
    int watertempraw;
    double offset = -48.0; // from Porsche DBC
    double factor = 0.75; // from Porsche DBC
    double physmax = 142.5; // from Porsche DBC
    double physmin = -48.0; // from Porsche DBC

    // convert fahrenheit to degree
    //watertempF = (watertempF - 32 ) * (5 / 9);

    if (wTempCelsius > physmax || wTempCelsius < physmin)
    {
        std::cerr << "KI::setWaterTemp: value for wTempCelsius higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        watertempraw = calcRawValue(wTempCelsius, offset, factor);
        p_CANProv->GW_D_2.values.canmsg.cansignals.Tmot_D = watertempraw;
    }
}

void KI::setOiltemp(double oilTCelcius)
{
    int oiltempraw;
    double offset = -48.0; // from Porsche DBC
    double factor = 0.75; // from Porsche DBC
    double physmax = 142.5; // from Porsche DBC
    double physmin = -48.0; // from Porsche DBC

    // convert fahrenheit to degree
    // oiltempF = (oiltempF - 32 ) * (5 / 9);

    if (oilTCelcius > physmax || oilTCelcius < physmin)
    {
        std::cerr << "KI::setOiltemp: value for oilTCelcius higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        oiltempraw = calcRawValue(oilTCelcius, offset, factor);
        p_CANProv->MOTOR_4_D.values.canmsg.cansignals.Toel_D = oiltempraw;
    }
}

void KI::setTime()
{
    int maxh = 23; // from Porsche DBC
    int minh = 0; // from Porsche DBC
    int maxmin = 59; // from Porsche DBC
    int minmin = 0; // from Porsche DBC
    int maxs = 59; // from Porsche DBC
    int mins = 0; // from Porsche DBC

    time_t rawTime;

    // get raw system (calendar) time
    time(&rawTime);

    // convert calendar time to broken-down time
    struct tm *ptimeStruct = localtime(&rawTime);

    //fprintf(stderr,"KI::set_time\n");

    if ((ptimeStruct->tm_hour > maxh || ptimeStruct->tm_hour < minh)
        && (ptimeStruct->tm_min > maxmin || ptimeStruct->tm_min < minmin)
        && (ptimeStruct->tm_sec > maxs || ptimeStruct->tm_sec < mins))
    {
        std::cerr << "KI::setTime: values for setTime are not within bounds"
                  << std::endl;
    }
    else
    {
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Std = ptimeStruct->tm_hour;
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Min = ptimeStruct->tm_min;
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Sek = ptimeStruct->tm_sec;
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Uebern = 1;

        timeval currentTime;
        gettimeofday(&currentTime, NULL);
        m_tstamp_pcmtime = (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
    }
}

void KI::doBCSet()
{
    // ATTENTION: falling edge is generated by kombined instrument!
    p_CANProv->BC_1_D.values.canmsg.cansignals.BC_Set_D = 1;
}

void KI::doBCReset()
{
    // ATTENTION: falling edge is generated by kombined instrument!
    p_CANProv->BC_1_D.values.canmsg.cansignals.BC_Reset_D = 1;
}

void KI::doBCUp()
{
    // ATTENTION: falling edge is generated by kombined instrument!
    p_CANProv->BC_1_D.values.canmsg.cansignals.BC_Up_Cursor_D = 1;
}

void KI::doBCDown()
{
    // ATTENTION: falling edge is generated by kombined instrument!
    p_CANProv->BC_1_D.values.canmsg.cansignals.BC_Down_Cursor_D = 1;
}

void KI::activateDriverBeltWarning()
{
    p_CANProv->GW_D_4.values.canmsg.cansignals.S_Gurt_F_D = 0;
}

void KI::activateCoDriverBeltWarning()
{
    p_CANProv->GW_D_4.values.canmsg.cansignals.S_Gurt_BF_D = 0;
}

void KI::deactivateDriverBeltWarning()
{
    p_CANProv->GW_D_4.values.canmsg.cansignals.S_Gurt_F_D = 1;
}

void KI::deactivateCoDriverBeltWarning()
{
    p_CANProv->GW_D_4.values.canmsg.cansignals.S_Gurt_BF_D = 1;
}

void KI::setGearshiftLever(uint leverpos)
{
    switch (leverpos)
    {
    case 0: //P
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 8;
        break;
    case 1: //N
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 6;
        break;
    case 2: // R
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 7;
        break;
    case 3: // D
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 5;
        break;
    case 4: // M
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 14;
        break;
    default: //N
        p_CANProv->GW_D_2.values.canmsg.cansignals.poshebel_D = 6;

        std::cerr << "KI::setGearshiftLever: value for leverpos higher than 4 or lower than 0"
                  << std::endl;
    }
}

void KI::setGear(int gearnumber)
{
    switch (gearnumber)
    {
    case -1:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 7;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 7;
        break;
    case 0:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 0;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 0;
        break;
    case 1:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 1;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 1;
        break;
    case 2:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 2;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 2;
        break;
    case 3:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 3;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 3;
        break;
    case 4:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 4;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 4;
        break;
    case 5:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 5;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 5;
        break;
    default:
        p_CANProv->GW_D_6.values.canmsg.cansignals.Eing_Gang_D = 1;
        p_CANProv->GW_D_2.values.canmsg.cansignals.gangauti_D = 0;

        std::cerr << "KI::setGear: value for gearnumber higher than +5 or lower than -1"
                  << std::endl;
    }
}

// (de)activates the KI's illumination
void KI::activateIllum()
{
    p_CANProv->GW_D_1.values.canmsg.cansignals.S_SLicht_D = 1;
}

// (de)activates the KI's illumination
void KI::deactivateIllum()
{
    p_CANProv->GW_D_1.values.canmsg.cansignals.S_SLicht_D = 0;
}

void KI::indicator(BlinkerTask::blinkState state)
{
    bt->setState(state);
}

//--------------------------------------------------------------------

// private methods to control the KI by CAN --------------------------
void KI::setTirepressureDiff(double pressurediffBar)
{
    int pressureraw;
    double offset = -6.3; // from Porsche DBC
    double factor = 0.1; // from Porsche DBC
    double physmax = 6.3; // from Porsche DBC
    double physmin = -6.3; // from Porsche DBC

    if (pressurediffBar > physmax || pressurediffBar < physmin)
    {
        std::cerr << "KI::setTirepressureDiff: value for pressurediffBar higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        pressureraw = calcRawValue(pressurediffBar, offset, factor);
        p_CANProv->RDK_1_D.values.canmsg.cansignals.Diff_Druck_HL_D = pressureraw;
        p_CANProv->RDK_1_D.values.canmsg.cansignals.Diff_Druck_HR_D = pressureraw;
        p_CANProv->RDK_1_D.values.canmsg.cansignals.Diff_Druck_VL_D = pressureraw;
        p_CANProv->RDK_1_D.values.canmsg.cansignals.Diff_Druck_VR_D = pressureraw;
    }
}

void KI::setTirepressure(double pressureBar)
{
    int pressureraw;
    double offset = 0; // from Porsche DBC
    double factor = 0.1; // from Porsche DBC
    double physmax = 6.2; // from Porsche DBC
    double physmin = 0; // from Porsche DBC

    if (pressureBar > physmax || pressureBar < physmin)
    {
        std::cerr << "KI::setTirepressure: value for pressureBar higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        pressureraw = calcRawValue(pressureBar, offset, factor);
        p_CANProv->RDK_4_D.values.canmsg.cansignals.Druck_HL_unkomp_D = pressureraw;
        p_CANProv->RDK_4_D.values.canmsg.cansignals.Druck_HR_unkomp_D = pressureraw;
        p_CANProv->RDK_4_D.values.canmsg.cansignals.Druck_VL_unkomp_D = pressureraw;
        p_CANProv->RDK_4_D.values.canmsg.cansignals.Druck_VR_unkomp_D = pressureraw;
    }
}
//--------------------------------------------------------------------

void KI::setLEDState(unsigned char ledState)
{
    p_CANProv->LEDState.values.canmsg.cansignals.LED = ledState;
}

unsigned char KI::getLEDState()
{
    return p_CANProv->LEDState.values.canmsg.cansignals.LED;
}

unsigned char KI::getButtonState()
{
    return p_CANProv->ButtonState.values.canmsg.cansignals.Button;
}
unsigned char KI::getJoystickState()
{
    return p_CANProv->ButtonState.values.canmsg.cansignals.Joystick;
}
unsigned char KI::getLightState()
{
    return p_CANProv->ButtonState.values.canmsg.cansignals.Light;
}

// HMIDeviceIface methods --------------------------------------------
bool KI::keyInLock(int /* keynumber */)
{
    if (m_state == KI::DEFAULT)
    {
        setPetrolLevel(90);
        // set initial values for gauges
        setOutsideTemp(25.0);

        // set belt warning signals to off
        deactivateDriverBeltWarning();
        deactivateCoDriverBeltWarning();

        // activate the KI's illumination
        activateIllum();

        m_state = KI::KEY;
        return true;
    }
    else
        return false;
}

bool KI::ignitionOn()
{
    if (m_state == KI::KEY)
    {
        setPetrolLevel(90);

        // for tha show...
        setOilPressure(0.0);
        setSpeed(0.0);
        setRevs(0.0);
        setWaterTemp(96.0);
        setOiltemp(95.0);

        // other signals
        setTirepressureDiff(0.0);
        setTirepressure(2.0);
        setTime();
        setGearshiftLever(0);
        setGear(0);

        doBCSet();

        m_state = KI::IGNITED;
        return true;
    }
    else
        return false;
}

bool KI::initDevice()
{
    if (m_state == KI::IGNITED)
    {
        // here's the show...
        setPetrolLevel(90);
        setOilPressure(10.0);
        setWaterTemp(96.0);
        setOiltemp(95.0);
        setSpeed(0.0);
        setRevs(0.0);

        m_state = KI::INITIALIZED;
        return true;
    }
    else
        return false;
}

bool KI::startDevice()
{
    if (m_state == KI::INITIALIZED)
    {
        // end of the show
        setOilPressure(4.0);
        setSpeed(0.0);
        setRevs(0.0);
        setWaterTemp(96.0);
        setOiltemp(95.0);

        m_state = KI::STARTED;
        return true;
    }
    else
        return false;
}

bool KI::stopDevice()
{
    if (m_state == KI::STARTED || m_state == KI::IGNITED)
    {
        m_state = KI::KEY;
        return true;
    }
    else
        return false;
}

bool KI::shutDownDevice()
{
    if (m_state == KI::KEY)
    {
        // set belt warning signals to on
        activateDriverBeltWarning();
        activateCoDriverBeltWarning();

        // deactivate the KI's illumination
        deactivateIllum();

        m_state = KI::DEFAULT;
        return true;
    }
    else
        return false;
}
//--------------------------------------------------------------------

// Private methods ---------------------------------------------------
int KI::round(double d)
{
    return (int)(d < 0 ? d - 0.5f : d + 0.5f);
}

int KI::calcRawValue(double physvalue, double offset, double factor)
{
    return round((physvalue - offset) / factor);
}

bool KI::update()
{
    // generate falling edges for adoption of new time in KI
    timeval currentTime;
    gettimeofday(&currentTime, NULL);
    double cTime = (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
    if (m_tstamp_pcmtime + 2.0 < cTime)
    {
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Uebern = 1;
    }
    else
        p_CANProv->PCM_1.values.canmsg.cansignals.PCM_Uebern = 0;

    return true;
}
//--------------------------------------------------------------------

//BlinkerTask/////////////////////////////////////////////////////////

void BlinkerTask::setState(blinkState bs)
{
    if (bs != state)
    {
        state = bs;
        if (state != NONE) // start immediately but stop only after timeout
            event->signal(1);
    }
}

void BlinkerTask::run()
{
    while (blink)
    {
        static bool on = false;
        static int oldState;
#ifdef MERCURY
        unsigned int rmask;
#else
        unsigned long rmask;
#endif
        event->wait(1, &rmask, EV_ANY, 400000000);
        event->clear(1);
        if (blinkCounter > 4 || (state != NONE))
        {
            currentState = state;
        }
        if (state != NONE && oldState != state)
        {
            blinkCounter = 0;
            on = true;
        }
        if (currentState == LEFT)
        {
            on ? pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 1 : pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 0;
            pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 0;
            blinkCounter++;
            on = !on;
        }
        else if (currentState == RIGHT)
        {
            on ? pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 1 : pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 0;
            pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 0;
            blinkCounter++;
            on = !on;
        }
        else if (currentState == BOTH)
        {
            on ? pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 1 : pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 0;
            on ? pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 1 : pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 0;
            blinkCounter++;
            on = !on;
        }
        else if (currentState == NONE)
        {
            pCANprovider->GW_D_1.values.canmsg.cansignals.BL_r_an_D = 0;
            pCANprovider->GW_D_1.values.canmsg.cansignals.BL_l_an_D = 0;
            blinkCounter = 0;
            on = true;
        }
        oldState = state;
    }
}

// constructor
BlinkerTask::BlinkerTask(CANProvider *cp)
    : XenomaiTask("BlinkerTask")
{
    event = new XenomaiEvent(0);
    pCANprovider = cp;
    //fprintf(stderr,"BlinkerTask::BlinkerTask\n");
    //set_periodic(500);
    blink = true;
    state = NONE;
    currentState = NONE;
    bstart();
}

// destructor
BlinkerTask::~BlinkerTask()
{
    blink = false;
    currentState = NONE;
    setState(NONE);
    event->signal(1);
    delete event;
    //fprintf(stderr,"BlinkerTask::~BlinkerTask\n");
}

// destructor
void BlinkerTask::bstart()
{
    //fprintf(stderr,"SendTask::bstart\n");

    XenomaiTask::start();
}

// destructor
void BlinkerTask::bresume()
{
    //fprintf(stderr,"SendTask::bresume\n");

    XenomaiTask::resume();
}

// destructor
void BlinkerTask::bsuspend()
{
    //fprintf(stderr,"SendTask::bsuspend\n");

    XenomaiTask::suspend();
}
//-------------------------------------------------------------------------------
