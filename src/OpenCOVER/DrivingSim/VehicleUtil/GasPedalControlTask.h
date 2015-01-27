/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GasPedalControlTask_h
#define __GasPedalControlTask_h

//--------------------------------------------------------------------
// PROJECT        GasPedal                                 © 2009 HLRS
// $Workfile$
// $Revision$
// DESCRIPTION    Module for sending and receiving message from gas
//                pedal: Continental TEMIC HLZ FFP
//
// CREATED        15-May-09, F. Seybold
// MODIFIED       23-July-09, S. Franz
//                Application of HLRS style guide
//                Changed to HMIDevice
//                Changed to singleton
//
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

    void lock();
    void unlock();

    int getChecksumStatus();
    int getAliveStatus();
    int getStatusMode();
    int getStatusAck();
    int getActualPositionValue();
    int getStatusActualPosition();
    int getThermalReserve();
    int getActualCurrent();

    void setChecksumDemand(int);
    void setAliveDemand(int);
    void incrementAliveDemand();
    void setTargetPositionValue(int);
    void setMaxTargetForce(int);
    void setMinTargetForce(int);
    void setJitterAmplitude(int);
    void setJitterSignalForm(int);
    void setJitterFrequency(int);
    void setStiffness(int);

    //---Status---
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
    static const int MaxActualPositionValue = 4000;
    enum StatusPosition
    {
        StatusPositionDemandMet = 0,
        StatusPositionDemandPartlyMet = 1,
        StatusPositionDemandNotMet = 2,
        StatusPositionInvalid = 3
    };
    static const int MaxThermalReserve = 250;
    static const int MaxActualCurrent = 250;

    //---Demand---
    static const int MaxChecksumDemand = 255;
    static const int MaxAliveDemand = 14;
    static const int MaxTargetPositionValue = 4000;
    static const int MaxTargetForce = 250;
    static const int MaxJitterAmplitude = 7;
    enum DemandJitterSignalForm
    {
        DemandJitterSignalFormSine = 0,
        DemandJitterSignalFormSawTooth = 1,
        DemandJitterSignalFormSquare = 2
    };
    static const int MaxJitterFrequency = 3;
    static const int MaxStiffness = 250;

protected:
    GasPedal();

    static GasPedal *p_gaspedal;

private:
    static const can_id_t statusCanId = 0x162;
    static const can_id_t demandCanId = 0x15f;

    can_frame statusFrame;
    can_frame demandFrame;
};
//--------------------------------------------------------------------

#endif
