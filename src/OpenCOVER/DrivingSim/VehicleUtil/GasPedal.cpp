/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GasPedal.h"

//GasPedal////////////////////////////////////////////////////////////

// set protected static pointer for singleton to NULL
GasPedal *GasPedal::p_gaspedal = NULL;

// constructor, destructor, instance ---------------------------------
GasPedal::GasPedal()
{
    m_gaspedalstate = LOCKED;
}

GasPedal::~GasPedal()
{
    p_gaspedal = NULL;
}

// singleton
GasPedal *GasPedal::instance()
{
    if (p_gaspedal == NULL)
    {
        p_gaspedal = new GasPedal();
    }
    return p_gaspedal;
}
//--------------------------------------------------------------------

// public methods ----------------------------------------------------
int GasPedal::getChecksumStatus()
{
    return p_CANProv->STATUS_FFP.values.canmsg.cansignals.CHKSM_ST_FFP;
}

int GasPedal::getAliveStatus()
{
    return p_CANProv->STATUS_FFP.values.canmsg.cansignals.ALIV_ST_FFP;
}

int GasPedal::getStatusMode()
{
    return p_CANProv->STATUS_FFP.values.canmsg.cansignals.ST_FFP_MOD;
}

int GasPedal::getStatusAck()
{
    return p_CANProv->STATUS_FFP.values.canmsg.cansignals.ST_RCPT_FFP;
}

// gets actual value of angle in %
double GasPedal::getActualAngle()
{
    int angleraw;
    double anglephys;
    double offset = 0.0; // from FFP manual
    double factor = 0.025; // from FFP manual
    double physmax = 100.0; // from FFP manual
    double physmin = 0.0; // from FFP manual

    angleraw = p_CANProv->STATUS_FFP.values.canmsg.cansignals.ANG_AVL_FFP;

    if (angleraw == 0xFFF)
    {
        std::cerr << "GasPedal::getActualAngle: value from gas pedal is invalid"
                  << std::endl;
    }
    else
    {
        anglephys = calcPhys(angleraw, offset, factor);

        if (anglephys > physmax)
        {
            std::cerr << "GasPedal::getActualAngle: value higher than "
                      << physmax << std::endl;
            return physmax;
        }
        if (anglephys < physmin)
        {
            std::cerr << "GasPedal::getActualAngle: value lower than " << physmin << std::endl;
            return physmin;
        }
        else
            return anglephys;
    }
    return 0.0;
}

int GasPedal::getStatusActualAngle()
{
    return p_CANProv->STATUS_FFP.values.canmsg.cansignals.ST_ANG_AVL_FFP;
}

// gets actual value of thermal reserve in °C
double GasPedal::getThermalReserve()
{
    int thermalresraw;
    double thermalresphys;
    //double offset = 0.0;        // from FFP manual
    //double factor = 0.4;        // from FFP manual
    //double physmax = 100.0;     // from FFP manual
    //double physmin = 0.0;       // from FFP manual
    double offset = -40.0; // from Sebastian
    double factor = 0.68; // from Sebastian
    double physmax = +130.0; // from Sebastian
    double physmin = -40.0; // from Sebastian

    thermalresraw = p_CANProv->STATUS_FFP.values.canmsg.cansignals.THR_SPAR_FFP;

    if (thermalresraw == 0xFF)
    {
        std::cerr << "GasPedal::getThermalReserve: value from gas pedal is invalid"
                  << std::endl;
    }
    else
    {
        thermalresphys = calcPhys(thermalresraw, offset, factor);

        if (thermalresphys > physmax)
        {
            std::cerr << "GasPedal::getThermalReserve: value higher than " << physmax << std::endl;
            return physmax;
        }
        if (thermalresphys < physmin)
        {
            std::cerr << "GasPedal::getThermalReserve: value lower than " << physmin << std::endl;
            return physmin;
        }
        else
            return thermalresphys;
    }
    return 0.0;
}

// gets actual value of current in amp
double GasPedal::getActualCurrent()
{
    int currentraw;
    double currentphys;
    //double offset = 0.0;        // from FFP manual
    //double factor = 0.4;        // from FFP manual
    //double physmax = 100.0;     // from FFP manual
    //double physmin = 0.0;       // from FFP manual
    double offset = -15.0; // from Sebastian
    double factor = 0.012; // from Sebastian
    double physmax = +15.0; // from Sebastian
    double physmin = -15.0; // from Sebastian

    currentraw = p_CANProv->STATUS_FFP.values.canmsg.cansignals.STROM_IST_FFP;

    if (currentraw == 0xFF)
    {
        std::cerr << "GasPedal::getActualCurrent: value from gas pedal is invalid"
                  << std::endl;
    }
    else
    {
        currentphys = calcPhys(currentraw, offset, factor);

        if (currentphys < physmin)
        {
            std::cerr << "GasPedal::getActualCurrent: value lower than " << physmin << std::endl;
            return physmin;
        }
        if (currentphys > physmax)
        {
            std::cerr << "GasPedal::getActualCurrent: value higher than "
                      << physmax << std::endl;
            return physmax;
        }
        else
            return currentphys;
    }
    return 0.0;
}

void GasPedal::setChecksumDemand(int cs)
{
    int physmax = 255; // from FFP manual
    int physmin = 0; // from FFP manual

    if ((physmin <= cs) && (cs <= physmax))
    {
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.CHKSM_RQ_ANG_FFP = cs;
    }
    else
    {
        std::cerr << "GasPedal::setChecksumDemand: value for cs higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
}

void GasPedal::setAliveDemand(int ad)
{
    int physmax = 14; // from FFP manual
    int physmin = 0; // from FFP manual

    if ((physmin <= ad) && (ad <= physmax))
    {
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.ALIV_RQ_ANG_FFP = ad;
    }
    else
    {
        std::cerr << "GasPedal::setAliveDemand: value for ad higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
}

// sets target value for pedal angle %
void GasPedal::setTargetValueAngle(double angle)
{
    int angleraw;
    double offset = 0.0; // from FFP manual
    double factor = 0.025; // from FFP manual
    double physmax = 100.0; // from FFP manual
    double physmin = 0.0; // from FFP manual

    if (angle > physmax || angle < physmin)
    {
        std::cerr << "GasPedal::setTargetAngleValue: value for angle higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        angleraw = calcRawValue(angle, offset, factor);
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.ANGLE_FFP = angleraw;
    }
}

// sets target value for max force in %
void GasPedal::setTargetValueMaxTargetForce(double maxforce)
{
    int maxforceraw;
    double offset = 0.0; // from FFP manual
    double factor = 0.4; // from FFP manual
    double physmax = 100.0; // from FFP manual
    double physmin = 0.0; // from FFP manual

    if (maxforce > physmax || maxforce < physmin)
    {
        std::cerr << "GasPedal::setMaxTargetForce: value for maxforce higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        maxforceraw = calcRawValue(maxforce, offset, factor);
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FORCE_MAX_FFP = maxforceraw;
    }
}

// sets target value for min force in %
void GasPedal::setTargetValueMinTargetForce(double minforce)
{
    int minforceraw;
    double offset = 0.0; // from FFP manual
    double factor = 0.4; // from FFP manual
    double physmax = 100.0; // from FFP manual
    double physmin = 0.0; // from FFP manual

    if (minforce > physmax || minforce < physmin)
    {
        std::cerr << "GasPedal::setMinTargetForce: value for minforce higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        minforceraw = calcRawValue(minforce, offset, factor);
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FORCE_MAX_FFP = minforceraw;
    }
}

// sets target value for stiffness in %
void GasPedal::setTargetValueStiffness(double stiffness)
{
    int stiffnessraw;
    double offset = 0.0; // from FFP manual
    double factor = 0.4; // from FFP manual
    double physmax = 100.0; // from FFP manual
    double physmin = 0.0; // from FFP manual

    if (stiffness > physmax || stiffness < physmin)
    {
        std::cerr << "GasPedal::setStiffness: value for minforce higher than "
                  << physmax << " or lower than " << physmin << std::endl;
    }
    else
    {
        stiffnessraw = calcRawValue(stiffness, offset, factor);
        p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.STIFFNESS_FFP = stiffnessraw;
    }
}

void GasPedal::setJitterSignalForm(int form)
{
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP &= ~0x18;
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP |= (form << 3) & 0x18;
    //demandFrame.data[5] = (demandFrame.data[5] & ~0x18) | ((form << 3) & 0x18);
}

void GasPedal::setJitterAmplitude(int amp)
{
    //demandFrame.data[5] = (demandFrame.data[5] & ~0x7) | (amp & 0x7);
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP &= ~0x7;
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP |= amp & 0x7;
}

void GasPedal::setJitterFrequency(int freq)
{
    //demandFrame.data[5] = (demandFrame.data[5] & ~0x60) | ((freq << 5) & 0x60);
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP &= ~0x60;
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_JITTER_FFP |= (freq << 5) & 0x60;
}

void GasPedal::lockGasPedal()
{
    std::cerr << "GasPedal::lockGasPedal" << std::endl;
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_FFP = 0;
}

void GasPedal::unlockGasPedal()
{
    std::cerr << "GasPedal::unlockGasPedal" << std::endl;
    p_CANProv->ANFORDERUNG_FFP.values.canmsg.cansignals.FUNC_FFP = 1;
}

// void GasPedal::incrementAliveDemand()
// {
//    demandFrame.data[1] = (demandFrame.data[1] & ~0xf) | ((((demandFrame.data[1] & 0xf) + 1) % 15) & 0xf);
// }
//--------------------------------------------------------------------

// HMIDeviceIface methods --------------------------------------------
bool GasPedal::keyInLock(int /*keynumber*/)
{
    if (m_gaspedalstate == GasPedal::LOCKED)
    {
        unlockGasPedal();
        m_gaspedalstate = GasPedal::UNLOCKED;
        return true;
    }
    else
        return false;
}

bool GasPedal::shutDownDevice()
{
    if (m_gaspedalstate == GasPedal::UNLOCKED)
    {
        lockGasPedal();
        m_gaspedalstate = GasPedal::LOCKED;
        return true;
    }
    else
        return false;
}
//--------------------------------------------------------------------

// Private methods ---------------------------------------------------

// This function calculates the physical value of a can signal based on
// its raw value (= value transmitted inside the can frame)
double GasPedal::calcPhys(double rawvalue, double offset, double factor)
{
    return (rawvalue * factor) + offset;
}

// This function calculates the raw (= value to transmit inside the
// can frame)value of a can signal based on the given physical value
int GasPedal::calcRawValue(double physvalue, double offset, double factor)
{
    return round((physvalue - offset) / factor);
}

int GasPedal::round(double d)
{
    return (int)(d < 0 ? d - 0.5f : d + 0.5f);
}
//--------------------------------------------------------------------
