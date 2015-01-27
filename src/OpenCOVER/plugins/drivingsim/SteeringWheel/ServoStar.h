/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SERVOSTAR_H
#define SERVOSTAR_H

#ifdef HAVE_PCAN
#ifndef WIN32
#include "sys/time.h"
#include "unistd.h"
#include "sched.h"
#else
#ifdef _WIN32
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX 512
#endif
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include "CanOpenBus.h"

///Class providing functions to handle a Kollmorgen ServoStar 600 over a CanOpen bus.
class ServoStar
{
public:
    ///Constructor.
    /**
		\param bus Pointer to a CanOpenBus object.
		\param moduleID Node id of the ServoStar.
	**/
    ServoStar(CanOpenBus *bus, unsigned char moduleID);

    /// Read a object from the dictionary of the ServoStar.
    /**
		\param index Index of the dictionary entry.
		\param subindex Subindex of the dictionary entry.
		\param length Length of the returned object content array (call by reference).
		\param data Pointer to an array the object content will be saved in (call by reference).

		\return True if reading has been successful.
	**/
    bool readObject(unsigned short index, unsigned char subindex, unsigned char &length, void *&data);

    /// Write to an object in the dictionary of a node.
    /**
		\param index Index of the dictionary entry.
		\param subindex Subindex of the dictionary entry.
		\param length Length of the data array (in byte). (call by reference -> will be set the length of the Object)
		\param data Pointer that will be set to the data array of the dictionary entry. (call by reference)

		\return True if writing has been successful.

		The length of the data array must not exceed four bytes.
	**/
    bool writeObject(unsigned short index, unsigned char subindex, unsigned char length, unsigned char *data);

    ///Start node by sending a NMT message.
    bool startNode();

    ///Stop node by sending a NMT message.
    bool stopNode();

    /// Reset node by sending a NMT message.
    bool resetNode();

    /// Set state of the status machine via control word: Enable Operation.
    bool enableOp();

    /// Set state of the status machine via control word: Disable Operation.
    bool disableOp();

    /// Set state of the status machine via control word: Enable Homing.
    bool enableHoming();

    /// Set state of the status machine via control word: Disable Homing (same state as Enable Operation).
    bool disableHoming();

    /// Set state of the status machine via control word: New Setpoint (same state as Enable Homing).
    bool newSetpoint();

    /// Set state of the status machine via control word: Absolute Positioning (pp mode).
    bool absolutePos();

    /// Set state of the status machine via control word: Shutdown.
    bool shutdown();

    /// Set mode of operation.
    /**
		\param opMode Mode of operation:
			- opMode=0xf7: Electrical Gearing
			- opMode=0xf8: Jogging
			- opMode=0xf9: Homing
			- opMode=0xfa: Trajectory
			- opMode=0xfb: Analog current
			- opMode=0xfc: Analog speed
			- opMode=0xfd: Digital current
			- opMode=0xfe: Digital speed
			- opMode=0xff: Position
			- opMode=0x1: Positioning (pp)
			- opMode=0x3: Speed (pv)
			- opMode=0x6: Homing (hm)
	**/
    bool setOpMode(unsigned char opMode);

    /// Set TPDO of the ServoStar.
    /**
		\param tpdonum Number of the TPDO to be set. (tpdonum=1..4)
		\param pdo Number of PDO the TPDO will be set to. (e.g. PDO33: incremental actual position)
	**/
    bool setTPDO(unsigned char tpdonum, unsigned char pdo);

    /// Set RPDO of the ServoStar.
    /**
		\param rpdonum Number of the RPDO to be set. (tpdonum=1..4)
		\param pdo Number of PDO the RPDO will be set to. (e.g. PDO22: current/speed setpoint)
	**/
    bool setRPDO(unsigned char rpdonum, unsigned char pdo);

    /// Set communication parameter of a TPDO.
    /**
		\param tpdonum Number of TPDO to be configured.
		\param tt Transmission time of TPDO:
						- tt=0x01: Transmit after every SYNC signal
						- tt=0x02: Transmit after every second SYNC signal
						- tt=0xff: Transmit asynchronous (event driven)
		\param it Inhibit time of TPDO
	**/
    bool setTPDOCom(unsigned char tpdonum, unsigned char tt, unsigned char it);

    /// Set communication parameter of a RPDO.
    /**
		\param rpdonum Number of RPDO to be configured.
		\param tt Transmission time of RPDO.
						- tt=0x01: Receive after every SYNC signal
						- tt=0x02: Receive after every second SYNC signal
						- tt=0xff: Receive asynchronous
		\param it Inhibit time of RPDO
	**/
    bool setRPDOCom(unsigned char rpdonum, unsigned char tt, unsigned char it);

    /// Set TPDO mapping for one object
    /**
		\param tpdonum Number of TPDO to configure mapping.
		\param index Dictionary index of the object to be mapped onto the TPDO.
		\param subindex Subindex of the object to be mapped onto the TPDO.
		\param datatype Datatype of the object to be mapped onto the TPDO (Number of bits).

		A TPDO is only mapable if it is set to PDO 37, 38, 39 or 40.
	**/
    bool setTPDOMap(unsigned char tpdonum, unsigned short index, unsigned char subindex, unsigned char datatype);

    /// Set TPDO mapping for two objects
    /**
		\param tpdonum Number of TPDO to configure mapping.
		\param index1 Dictionary index of the first object to be mapped onto the TPDO.
		\param subindex1 Subindex of the first object to be mapped onto the TPDO.
		\param datatype1 Datatype of the first object to be mapped onto the TPDO (Number of bits).
		\param index2 Dictionary index of the second object to be mapped onto the TPDO.
		\param subindex2 Subindex of the second object to be mapped onto the TPDO.
		\param datatype2 Datatype of the second object to be mapped onto the TPDO (Number of bits).

		A TPDO is only mapable if it is set to PDO 37, 38, 39 or 40.
	**/
    bool setTPDOMap(unsigned char tpdonum,
                    unsigned short index1, unsigned char subindex1, unsigned char datatype1,
                    unsigned short index2, unsigned char subindex2, unsigned char datatype2);

    /// Set RPDO mapping for one object
    /**
		\param rpdonum Number of RPDO to configure mapping.
		\param index Dictionary index of the object to be mapped onto the RPDO.
		\param subindex Subindex of the object to be mapped onto the RPDO.
		\param datatype Datatype of the object to be mapped onto the RPDO (Number of bits).

		A RPDO is only mapable if it is set to PDO 37, 38, 39 or 40.
	**/
    bool setRPDOMap(unsigned char rpdonum, unsigned short index, unsigned char subindex, unsigned char datatype);

    /// Set RPDO mapping for two objects
    /**
		\param rpdonum Number of RPDO to configure mapping.
		\param index1 Dictionary index of the first object to be mapped onto the RPDO.
		\param subindex1 Subindex of the first object to be mapped onto the RPDO.
		\param datatype1 Datatype of the first object to be mapped onto the RPDO (Number of bits).
		\param index2 Dictionary index of the second object to be mapped onto the RPDO.
		\param subindex2 Subindex of the second object to be mapped onto the RPDO.
		\param datatype2 Datatype of the second object to be mapped onto the RPDO (Number of bits).

		A RPDO is only mapable if it is set to PDO 37, 38, 39 or 40.
	**/
    bool setRPDOMap(unsigned char rpdonum,
                    unsigned short index1, unsigned char subindex1, unsigned char datatype1,
                    unsigned short index2, unsigned char subindex2, unsigned char datatype2);

    /// Set homing parameters for operation mode "Homing (hm)".
    /**
		\param offset Difference between the zero position for the application and the zero point of the machine.
		\param type Type of homing:
			- type=-3: move to mechanical stop, with zeroing
			- type=-2: set reference point at present position, allowing for lag/following error
			- type=-1: homing within a single turn
			- type=1: ...
			- type=2: ...
			- type=8: ...
			- type=12: ...
			- type=17: ...
			- type=18: ...
			- type=24: ...
			- type=28: ...
			- type=33: homing within a single turn, negative direction of rotation
			- type=34: ...
			- type=35: ...
		\param vel Homing speed. Speed during search for switch.
		\param acc Homing acceleration.

			See CanOpen Servostar 400/600 Communication Profile documentation for homing types.
	**/
    bool setupHoming(int offset, char type, unsigned int vel, unsigned int acc);

    /// Set parameters for operation mode "Profile Position Mode (pp)"
    /**
		\param target Target position.
		\param vel Target velocity during run for target position.
		\param acc Acceleration ramp.
	**/
    bool setupPositioning(int target, unsigned int vel, unsigned int acc);

    /// Sleep a given time in microseconds.
    /**
		\param sleeptime Time to sleep in microseconds.
	
		This functions uses gettimeofday from sys/time.h.
	**/

    /// Activate Life Guarding of the ServoStar and set guard time.
    /**
		\param guardTime Guarding time in milliseconds
	**/
    bool setupLifeGuarding(unsigned short guardTime);

    /// Stop Life Guarding of the ServoStar.
    bool stopLifeGuarding();

    /// Send a guard message (RTR message);
    bool sendGuardMessage();

    /// Wait the indicated time.
    /**
		\param waittime Time to wait in microseconds.
		This function uses full resources due to polling the actual time. It's more accurate than the microsleep-function.
	**/
    void microwait(unsigned long waittime);

    /// Sleep the indicated time.
    /**
		\param sleeptime Time to sleep in microseconds.
	**/
    void microsleep(unsigned long sleeptime);

    /// Gets the elapsed seconds since the Epoch (something around 1.1.1970).
    /**
		\return The elapsed time in seconds.
	**/
    unsigned long getTimeInSeconds();

    /// Gets the elapsed microseconds since the Epoch (something around 1.1.1970).
    /**
		\return The elapsed time in microseconds.
	**/
    unsigned long getTimeInMicroseconds();

    /// Checks if the output stage of the node is enabled.
    /**
		\return True, if enabled. Returns false, if disabled or an error occured.
	**/
    bool checkOutputStageEnabled();

protected:
    unsigned char id;

    CanOpenBus *bus;
};
#endif

#endif
