/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PORSCHESTEERINGWHEEL_H
#define PORSCHESTEERINGWHEEL_H

#ifdef HAVE_PCAN
#include <iostream>
#include "ServoStar.h"
#ifndef WIN32
#include "sched.h"
#include "errno.h"
#endif

#define PI 3.14159265
#define BUFFERLENGTH 10

/// Class providing functions especially for the Porsche Steering Wheel of the VIS department of HLRS, Stuttgart.
/**
	This class inherits from the ServoStar class.
	Focus is on a realtime get angle and set torque mechanism, as well as a steering wheel homing function.
**/
class PorscheSteeringWheel : public ServoStar
{

public:
    ///Constructor.
    /**
		\param bus Pointer to a CanOpenBus object.
		\param moduleID Node id of the ServoStar.
	**/
    PorscheSteeringWheel(CanOpenBus *bus, unsigned char moduleID);

    ///Set triggers in the ServoStar to provide a get angle and set torque mechanism using PDOs.
    /**
		\param tpdott Transmission type of the angle and velocity tpdo:
							- tpdott=0x01: PDO will be sent after every SYNC signal.
							- tpdott=0xff: PDO will be sent asynchronously (event driven).
		\param guardTime Guarding time for the node life guard in milliseconds.

		\return Returns true if setup has been successful.

		Sets up TPDO3 of the ServoStar for reading angle and velocity values and sets up RPDO3 for writing current setpoints to the ServoStar.	These values can be accessed with the getAngleVelocity and setTorque functions.
	**/
    bool setupAngleTorqueMode(unsigned char tpdott, unsigned short guardTime);

    /// Start PDO traffic of get angle and set torqe mechanism
    bool startAngleTorqueMode();

    /// Stop life guarding of node, shutdown wheel and stop PDO traffic of get angle and set torqe mechanism.
    bool stopAngleTorqueMode();

    /// Check wheel for output stage enabled.
    /**
		\param sleeptime Sleeptime between check cycles in microseconds.
		\param cycles Number of cylces before exit.

		\return True, if output stage enabled. False, if output stage not enabled during the check cycles.

		The functions returns when the output stage is enabled.
	**/
    bool checkWheel(int sleeptime, int cycles);

    /// Reset Wheel.
    bool resetWheel();

    ///Home Porsche Steering Wheel.
    /**
		\return Returns true when homing of the steering wheel is succesfully completed
	**/
    bool homeWheel();

    ///Get angle and velocity of steering wheel.
    /**
		\param angle Angle of the steering wheel (call by reference). Normally in increments: 1 turn = 2^20 increments
		\param velocity Velocity of the steering wheel (call by reference). Resolution: 1bit = 125/1048576 revs/sec.

		\return Returns false if not the expected PDO message has arrived. In this case, the values angle and velocity aren't set. You probably should just call this function again. If you are using SYNCs to get the PDOs, also send a SYNC to the bus again.

		This function blocks until a message arrives.
		Use porscheSetupAngleTorqueMode to setup the ServoStar and start transmission with startNode.
		This function uses a simple "mean-of-last-ten-values"-smoothing for the velocity.
	**/
    bool getAngleVelocity(int &angle, int &velocity);

    ///Set torque of steering wheel.
    /**
		\param torque Torque setpoint for the steering wheel. Value range is -3280 to 3280. If given value exceeds this value range, the value is set to the according maximum value (-3280 or 3280).

		\return Returns true if sending setpoint has been successful (the reception is unconfirmed).

		Use porscheSetupAngleTorqueMode to setup the ServoStar and start transmission with startNode.
	**/
    bool setTorque(int torque);

    ///Get angle of steering wheel in [radians] and velocity in [radians/second].
    /**
		\param angleFloat Angle of the steering wheel in radians(call by reference).
		\param velFloat Velocity of the steering wheel (call by reference).

		\return Returns false if not the expected PDO message has arrived. In this case, the values angle and velocity aren't set. You probably should just call this function again. If you are using SYNCs to get the PDOs, also send a SYNC to the bus again.

		This function blocks until a message arrives.
		Use porscheSetupAngleTorqueMode to setup the ServoStar and start transmission with startNode.
	**/
    bool getAngleVelocityInRadians(double &angleFloat, double &velFloat);

    ///Set torque of steering wheel in [Nm].
    /**
		\param torque Torque setpoint for the steering wheel in [Nm]. If value is over maximum torque, it is set to maximum.

		\return Returns true if sending setpoint has been successful (the reception is unconfirmed).

		Use porscheSetupAngleTorqueMode to setup the ServoStar and start transmission with startNode.
	**/
    bool setTorqueInNm(double torque);

    /// Set FIFO scheduler and higher priority for this process.
    bool setScheduler();

    /// Nice this process.
    /**
		\param incr Nice increment.

		To nice a process means to lower its schedule priority.
	**/
    bool niceProcess(int incr);

    /// Set processor affinity of current process/thread.
    /**
		\param cpu Cpu to set affinity to.
	**/
    bool setAffinity(int cpu);

protected:
    double scaleAngle; //[increments to radians]
    double scaleVel; //[revs/min to radians/second]
    double scaleTorque; //[torque current increments to Nm] linear function assumed
    unsigned char torqueData[6];

    int velBuffer[BUFFERLENGTH];
    int angBuffer[BUFFERLENGTH];
    unsigned char bufferIndex;
};
#endif
#endif
