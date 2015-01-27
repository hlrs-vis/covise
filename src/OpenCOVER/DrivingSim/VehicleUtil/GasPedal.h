/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GasPedal_h
#define __GasPedal_h

//--------------------------------------------------------------------
// PROJECT        GasPedal                                 © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    Module for sending and receiving message from gas
//                pedal Continental TEMIC HLZ FFP
//
// CREATED        15-May-09, F. Seybold
// MODIFIED       23-July-09, S. Franz
//                - Application of HLRS style guide
//                - Changed to HMIDevice
//                - Changed to singleton
//                30-July-09, S. Franz
//                - Updated comments / documentation
//--------------------------------------------------------------------
//--------------------------------------------------------------------
// $Log$
//--------------------------------------------------------------------
// TAB WIDTH    3
//--------------------------------------------------------------------

#include "HMIDeviceIface.h"
#include "XenomaiSocketCan.h"
#include <iostream>
#include <native/timer.h>

//--------------------------------------------------------------------
// TODO(sebastian): Class comments
class GasPedal : public HMIDeviceIface
{
public:
    virtual ~GasPedal();

    void lockGasPedal();
    void unlockGasPedal();

    int getChecksumStatus();
    int getAliveStatus();
    int getStatusMode();
    int getStatusAck();
    double getActualAngle();
    int getStatusActualAngle();
    double getThermalReserve();
    double getActualCurrent();

    void setChecksumDemand(int cs);
    void setAliveDemand(int ad);
    void setTargetValueAngle(double angle);
    void setTargetValueMaxTargetForce(double maxforce);
    void setTargetValueMinTargetForce(double minforce);
    void setTargetValueStiffness(double stiffness);
    void setJitterSignalForm(int formnumber);

    void setJitterAmplitude(int);
    void setJitterFrequency(int);
    void incrementAliveDemand();

    // HMIDeviceIface methods
    bool keyInLock(int keynumber);
    bool shutDownDevice();

    // Enums
    enum StatusMode
    {
        StatusModeReady = 0,
        StatusModeInit = 1,
        StatusModeError = 2,
        StatusModeInvalid = 3
    };
    enum StatusAck
    {
        StatusAckNoError = 0,
        StatusAckActionError = 1,
        StatusAckTimeOut = 2,
        StatusAckInvalid = 3
    };
    enum StatusPosition
    {
        StatusPositionDemandMet = 0,
        StatusPositionDemandPartlyMet = 1,
        StatusPositionDemandNotMet = 2,
        StatusPositionInvalid = 3
    };
    enum DemandJitterSignalForm
    {
        DemandJitterSignalFormSine = 0,
        DemandJitterSignalFormSawTooth = 1,
        DemandJitterSignalFormSquare = 2
    };

    // constants
    static const int MaxActualPositionValue = 4000;
    static const int MaxThermalReserve = 250;
    static const int MaxActualCurrent = 250;
    static const int MaxChecksumDemand = 255;
    static const int MaxAliveDemand = 14;
    static const int MaxTargetPositionValue = 4000;
    static const int MaxTargetForce = 250;
    static const int MaxJitterAmplitude = 7;
    static const int MaxJitterFrequency = 3;
    static const int MaxStiffness = 250;

    static GasPedal *instance(); // singleton

protected:
    GasPedal();

    static GasPedal *p_gaspedal;

private:
    enum m_gaspedalstates
    {
        UNLOCKED = 0,
        LOCKED = 1,
    };
    int m_gaspedalstate;

    double calcPhys(double rawvalue, double offset, double factor);
    int calcRawValue(double physvalue, double offset, double factor);
    int round(double d);
};
//--------------------------------------------------------------------

#endif
