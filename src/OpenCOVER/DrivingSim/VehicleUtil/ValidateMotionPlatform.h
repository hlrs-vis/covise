/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __ValidateMotionPlatform_h
#define __ValidateMotionPlatform_h

#include "XenomaiTask.h"
#include "XenomaiSocketCan.h"
#include "XenomaiMutex.h"
#include <vector>
#include <map>
#include <deque>
#include <cmath>
#include <unistd.h>
#include <osg/Matrix>

class VEHICLEUTILEXPORT ValidateMotionPlatform : public XenomaiTask, public XenomaiSocketCan
{
public:
    static ValidateMotionPlatform *instance()
    {
        if (instancePointer)
        {
            return instancePointer;
        }
        else
        {
            return (instancePointer = new ValidateMotionPlatform());
        }
    }

    ~ValidateMotionPlatform();

    void setPositionSetpoint(const unsigned int &, double);
    void setVelocitySetpoint(const unsigned int &, double);
    void setAccelerationSetpoint(const unsigned int &, double);

    void setLongitudinalAngleSetpoint(double);
    void setLateralAngleSetpoint(double);

    template <uint16_t controlMode>
    void switchToMode(const unsigned int &mot);
    template <uint16_t controlMode>
    void switchToMode();

    bool isGrounded();
    bool isMiddleLifted();
    bool isLifted();
	osg::Matrix motionPlatformPosition();
    bool isInitialized()
    {
        return initialized;
    };

    double getPosition(const unsigned int &mot) const
    {
        return (double)((((*((int32_t *)(stateFrameVector[mot].data))) & 0xffff) << 16) + ((*((int32_t *)(stateFrameVector[mot].data + 2))) & 0xffff)) / posFactorIncPerSI;
    }
    double getVelocity(const unsigned int &mot) const
    {
        return (double)((((*((int32_t *)(stateFrameVector[mot].data + 4))) & 0xffff) << 16) + ((*((int32_t *)(stateFrameVector[mot].data + 6))) & 0xffff)) / velFactorIncPerSI;
    }
    XenomaiMutex &getSendMutex()
    {
        return sendMutex;
    }

    double getBrakePedalForce()
    {
        return brakeForce;
    }
    double getBrakeForce()
    {
        //return std::max(-brakeForce - 5.0, 0.0);
        return std::max(-brakeForce - 30.0, 0.0);
    }
    double getBrakePedalPosition()
    {
        return ((double)(*(int16_t *)stateFrameVector[brakeMot].data)) * posFactorSIPerMM;
    }
    double getBrakePedalVelocity()
    {
        return ((double)(*(int32_t *)stateFrameVector[brakeMot].data + 2)) * velFactorSIPerMM;
    }
    void setBrakePedalForce(const double &bf)
    {
        brakeForce = bf;
    }

    //Number of linear motors
    static constexpr unsigned int numLinMots = 3;

    //Absolute limits: SI-Units, Attention: Only provide positive (absolute) valued!
    static constexpr double posMin = 0.02;
    static constexpr double posMax = 0.38;
    static constexpr double posFactorIncPerSI = 1e6;

    static constexpr double velMin = 0.0;
    static constexpr double velMax = 1.0;
    //static const double velMax = 0.1;
    static constexpr double velFactorIncPerSI = 1e4;

    static constexpr double accMin = 0.1;
    static constexpr double accMax = 10.0; //As of 2009-09-17: 20.0 also o.k.!
    //static const double accMax = 1.0;	//As of 2009-09-17: 20.0 also o.k.!
    static constexpr double accFactorIncPerSI = 1e3;

    static constexpr double posMiddle = 0.2;
    static constexpr double sideMotDist = 0.65;
    static constexpr double rearMotDist = 1.1;

    //Control modes
    enum Constants
{
    controlResetBit = 0x01,
    controlPositioning = (9 << 5) | (4 << 10), //FCB 9, Instance 4: Positioning
    controlToGround = (9 << 5) | (3 << 10), //FCB 9, Instance 3: Local Positioning, local vel/acc, setpoint = 0
    controlInterpolatedPositioning = (10 << 5) | (0 << 10), //FCB 10, Instance 0: Interpolated Positioning
    controlDisabled = (1 << 5) | (0 << 10), //FCB 1, Instance 0
    controlReset = (13 << 5) | (0 << 10) | controlResetBit, //FCB 13, Instance 0, Reset-Bit
    controlMiddleLift = (9 << 5) | (2 << 10), //FCB 9, Instance 2: Positioning, dummy for lift to middle position
    controlReference = (12 << 5) | (0 << 10), //FCB 12, Instance 0: Referencing
    controlTorque = (8 << 5) | (0 << 10), //FCB 8, Instance 0: Interpolated torque control
        brakeMot = 3,
    numRotMots = 1,
};
    //const uint16_t controlResetBit = 0x01;
    //const uint16_t controlPositioning = (9 << 5) | (4 << 10); //FCB 9, Instance 4: Positioning
    //const uint16_t controlToGround = (9 << 5) | (3 << 10); //FCB 9, Instance 3: Local Positioning, local vel/acc, setpoint = 0
    //const uint16_t controlInterpolatedPositioning = (10 << 5) | (0 << 10); //FCB 10, Instance 0: Interpolated Positioning
    //const uint16_t controlDisabled = (1 << 5) | (0 << 10); //FCB 1, Instance 0
    //const uint16_t controlReset = (13 << 5) | (0 << 10) | controlResetBit; //FCB 13, Instance 0, Reset-Bit
    //const uint16_t controlMiddleLift = (9 << 5) | (2 << 10); //FCB 9, Instance 2: Positioning, dummy for lift to middle position
    //const uint16_t controlReference = (12 << 5) | (0 << 10); //FCB 12, Instance 0: Referencing
    //const uint16_t controlTorque = (8 << 5) | (0 << 10); //FCB 8, Instance 0: Interpolated torque control

    //Number of rotary motors
    //static const unsigned int brakeMot = 3;
    //const unsigned int numRotMots = 1;
    const double forceFactorPercentPerSI = 1000.0; // [0.001N/N], 3 digits behind floating point
    const double posFactorSIPerMM = 0.00001; // [m/0.01mm], 2 digits behind floating point
    const double velFactorSIPerMM = 0.000001; // [m/0.001mm], 3 digits behind floating point
    const double accFactorSIPerDegree = 2.112; // [m/0.01degree], 2 digits behind floating point

    //Interval-Time
    const RTIME h_ns = 1000000;
    const double h = 1e-3;

protected:
#ifdef MERCURY
    ValidateMotionPlatform(const std::string & = "can0", int = 0, int = 99, int = 0, const can_id_t & = 0x0,
#else
    ValidateMotionPlatform(const std::string & = "rtcan0", int = 0, int = 99, int = T_FPU | T_CPU(4), const can_id_t & = 0x0,
#endif
                           const can_id_t & = 0x11, const can_id_t & = 0x181, const can_id_t & = 0x19,
                           const can_id_t & = 0x12, const can_id_t & = 0x281, const can_id_t & = 0x1a,
                           const can_id_t & = 0x13, const can_id_t & = 0x381, const can_id_t & = 0x1b,
                           const can_id_t & = 0x14, const can_id_t & = 0x481, const can_id_t & = 0x1c);
    static ValidateMotionPlatform *instancePointer;

    void applyPositionSetpoint(const unsigned int &, double);
    void run();

    template <uint16_t controlMode>
    void setControlMode(const unsigned int &mot)
    {
        *(uint16_t *)(controlFrameVector[mot].data) = controlMode;
        sendControlFrame = true;
    }

    template <uint16_t controlMode>
    void setLinMotsControlMode()
    {
        for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
        {
            setControlMode<controlMode>(motIt);
        }
    }

    uint16_t getControlMode(const unsigned int &mot)
    {
        return *(uint16_t *)(controlFrameVector[mot].data);
    }

    uint8_t *getState(const unsigned int &mot)
    {
        return stateFrameVector[stateIdIndexMap[mot]].data;
    }

    //send frames
    std::vector<can_frame> processFrameVector;
    std::vector<can_frame> controlFrameVector;

    //receive frames
    std::map<can_id_t, unsigned int> stateIdIndexMap;
    std::vector<can_frame> stateFrameVector;

    can_frame syncFrame;

    bool sendControlFrame;

    //Setpoint
    std::vector<double> posSetVector;
    double brakeForce;

    //Trajectory
    std::vector<std::deque<double> > trajectoryDequeVector; //[mot][0=n, 1=n-1, ...]

    //Trajector error
    std::vector<std::deque<double> > setpointDequeVector; //[mot][0=n, 1=n-1, ...]

    //Task control
    volatile bool runTask;
    volatile bool taskFinished;
    unsigned long overruns;
    unsigned long updateMisses;
    XenomaiMutex sendMutex;
    bool initialized;
};

inline void ValidateMotionPlatform::setPositionSetpoint(const unsigned int &mot, double sp)
{
    switch (getControlMode(mot))
    {
    case controlInterpolatedPositioning:
        //if(posSetVector[mot] != sp)
        //{
        posSetVector[mot] = sp;
        updateMisses = 0;
        //}
        break;

    default:
        if (sp < posMin)
        {
            sp = posMin;
        }
        else if (sp > posMax)
        {
            sp = posMax;
        }

        (*((uint32_t *)(processFrameVector[mot].data))) = (uint32_t)(posFactorIncPerSI * sp);
    }
}

inline void ValidateMotionPlatform::setVelocitySetpoint(const unsigned int &mot, double sp)
{
    if (sp < velMin)
    {
        sp = velMin;
    }
    else if (sp > velMax)
    {
        sp = velMax;
    }
    (*((uint16_t *)(processFrameVector[mot].data + 4))) = (uint16_t)(velFactorIncPerSI * sp);
}

inline void ValidateMotionPlatform::setAccelerationSetpoint(const unsigned int &mot, double sp)
{
    if (sp < accMin)
    {
        sp = accMin;
    }
    else if (sp > accMax)
    {
        sp = accMax;
    }
    (*((uint16_t *)(processFrameVector[mot].data + 6))) = (uint16_t)(accFactorIncPerSI * sp);
}

inline void ValidateMotionPlatform::setLongitudinalAngleSetpoint(double alpha)
{
    setPositionSetpoint(2, posMiddle + rearMotDist * tan(alpha));
}

inline void ValidateMotionPlatform::setLateralAngleSetpoint(double beta)
{
    double sidePosOff = sideMotDist * tan(beta);
    double positionRight = posMiddle - sidePosOff;
    double positionLeft = posMiddle + sidePosOff;
    setPositionSetpoint(0, positionRight);
    setPositionSetpoint(1, positionLeft);
}

template <uint16_t controlMode>
inline void ValidateMotionPlatform::switchToMode(const unsigned int &mot)
{
    setControlMode<controlMode>(mot);
}

template <>
inline void ValidateMotionPlatform::switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>(const unsigned int &mot)
{
    double pos_0 = getPosition(mot);

    if (pos_0 >= posMin && pos_0 <= posMax)
    {
        double pos_m1 = pos_0 - getVelocity(mot) * h;

        if (pos_0 < posMin)
        {
            pos_0 = posMin;
        }
        else if (pos_0 > posMax)
        {
            pos_0 = posMax;
        }
        if (pos_m1 < posMin)
        {
            pos_m1 = posMin;
        }
        else if (pos_m1 > posMax)
        {
            pos_m1 = posMax;
        }

        trajectoryDequeVector[mot].pop_back();
        trajectoryDequeVector[mot].push_front(pos_m1);
        trajectoryDequeVector[mot].pop_back();
        trajectoryDequeVector[mot].push_front(pos_0);
        setpointDequeVector[mot].pop_back();
        setpointDequeVector[mot].push_front(pos_m1);
        setpointDequeVector[mot].pop_back();
        setpointDequeVector[mot].push_front(pos_0);

        posSetVector[mot] = pos_0;
        updateMisses = 0;

        applyPositionSetpoint(mot, pos_0);

        setControlMode<controlInterpolatedPositioning>(mot);
    }
}

template <>
inline void ValidateMotionPlatform::switchToMode<ValidateMotionPlatform::controlMiddleLift>(const unsigned int &mot)
{
    setControlMode<controlPositioning>(mot);

    setVelocitySetpoint(mot, 0.1);
    setAccelerationSetpoint(mot, 0.2);
    setPositionSetpoint(mot, posMiddle);
}

template <uint16_t controlMode>
inline void ValidateMotionPlatform::switchToMode()
{
    for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
    {
        switchToMode<controlMode>(motIt);
    }
}

inline bool ValidateMotionPlatform::isGrounded()
{
    bool grounded = true;
    for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
    {
        if (getPosition(motIt) > 0.001)
        {
            grounded = false;
            break;
        }
    }
    return grounded;
}

inline bool ValidateMotionPlatform::isMiddleLifted()
{
    bool lifted = true;
    for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
    {
        if (fabs(getPosition(motIt) - posMiddle) > 0.0001)
        {
            lifted = false;
            break;
        }
    }
    return lifted;
}

inline bool ValidateMotionPlatform::isLifted()
{
    bool lifted = true;
    for (unsigned int motIt = 0; motIt < numLinMots; ++motIt)
    {
        double sp = (double)(*((uint32_t *)(processFrameVector[motIt].data))) / posFactorIncPerSI;
        if (fabs(getPosition(motIt) - sp) > 0.0001)
        {
            lifted = false;
            break;
        }
    }
    return lifted;
}

inline osg::Matrix ValidateMotionPlatform::motionPlatformPosition()
{
    osg::Matrix outMatrix;
	
	//double rz = getPosition(0);
	//double lz = getPosition(1);
	//double bz = getPosition(2);
	
	
	return outMatrix;
}
#endif
