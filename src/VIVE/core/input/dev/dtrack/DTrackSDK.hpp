/* DTrackSDK in C++: DTrackSDK.hpp
 *
 * Functions to receive and process DTrack UDP packets (ASCII protocol), as
 * well as to exchange DTrack2/DTRACK3 TCP command strings.
 *
 * Copyright (c) 2007-2022 Advanced Realtime Tracking GmbH & Co. KG
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Version v2.8.0
 *
 * Purpose:
 *  - receives DTrack UDP packets (ASCII protocol) and converts them into easier to handle data
 *  - sends and receives DTrack2/DTRACK3 commands (TCP)
 *  - DTrack network protocol according to:
 *    'DTrack2 User Manual, Technical Appendix' or 'DTRACK3 Programmer's Guide'
 */

/*! \mainpage DTrackSDK Overview
 *
 * \section intro_sec Introduction
 *
 * The DTrackSDK provides an interface to control a DTrack1/DTrack2/DTRACK3 server and to receive tracking data.
 * Command and data exchange is done through an ASCII protocol.
 *
 * \section content_sec Contents
 *
 * This package consists of one main class: DTrackSDK. For new applications please use this class. For access
 * to tracking data refer to its ancestor class DTrackParser.
 *
 * Classes in folder 'Compatibility' provide legacy support for older SDK versions (DTracklib, DTrack, DTrack2).
 *
 * - DTrackDataTypes class provides type definitions
 * - DTrackNet class provides basic UDP/TCP functionality
 * - DTrackParser class provides string parsing
 */

#ifndef _ART_DTRACKSDK_HPP_
#define _ART_DTRACKSDK_HPP_

#include "DTrackDataTypes.hpp"
#include "DTrackNet.hpp"
#include "DTrackParser.hpp"

#include <string>
#include <vector>

//! Max message size; DEPRECATED
#define DTRACK_PROT_MAXLEN  DTrackSDK::DTRACK2_PROT_MAXLEN

/**
 * \brief DTrack SDK main class derived from DTrackParser.
 *
 * All methods to access tracking data are located in DTrackParser.
 */
class DTrackSDK : public DTrackParser
{
public:

	static const int DTRACK2_PROT_MAXLEN = 200;  //!< max. length of 'dtrack2' command

	//! Compatibility modes for older DTrack systems
	typedef enum {
		SYS_DTRACK_UNKNOWN = 0,  //!< Unknown system
		SYS_DTRACK,              //!< DTrack1 system
		SYS_DTRACK_2             //!< DTrack2/DTRACK3 system
	} RemoteSystemType;

	//! Error codes
	typedef enum {
		ERR_NONE = 0,  //!< No error
		ERR_TIMEOUT,   //!< Timeout occured
		ERR_NET,       //!< Network error
		ERR_PARSE      //!< Error while parsing command
	} Errors;

	/**
	 * \brief Universal constructor. Can be used for any mode. Recommended for new applications.
	 *
	 * Refer to other constructors for details. Communicating mode just for DTrack2/DTRACK3.
	 *
	 * Examples for connection string:
	 * - "5000" : Port number (UDP), use for pure listening mode.
	 * - "224.0.1.0:5000" : Multicast IP and port number (UDP), use for multicast listening mode.
	 * - "atc-301422002:5000" : Hostname of Controller and port number (UDP), use for communicating mode.
	 * - "192.168.0.1:5000" : IP address of Controller and port number (UDP), use for communicating mode.
	 *
	 * @param[in] connection Connection string ("<data port>" or "<ip/host>:<data port>")
	 */
	DTrackSDK( const std::string& connection );

	/**
	 * \brief Constructor. Use for pure listening mode.
	 *
	 * Using this constructor, only a UDP receiver to get tracking data from
	 * the Controller will be established. Please start measurement manually.
	 *
	 * @param[in] data_port Port number (UDP) to receive tracking data from DTrack (0 if to be chosen by SDK)
	 */
	DTrackSDK(unsigned short data_port);

	/**
	 * \brief Constructor. Use for communicating mode with DTrack2/DTRACK3.
	 *
	 * Using this constructor, a UDP receiver to get tracking data from the
	 * Controller as well as a TCP connection with the Controller will be
	 * established. Automatically starts and stops measurement.
	 *
	 * Can also be used for multicast listening mode. In this case only a
	 * UDP receiver to get tracking data from the Controller will be
	 * established. Please start measurement manually.
	 *
	 * @param[in] server_host Hostname or IP address of Controller, or multicast IP address
	 * @param[in] data_port   Port number (UDP) to receive tracking data from DTrack (0 if to be chosen by SDK)
	 */
	DTrackSDK(const std::string& server_host, unsigned short data_port);

	/**
	 * \brief Constructor. Use for communicating mode with DTrack1.
	 *
	 * @param[in] server_host Hostname or IP address of DTrack1 PC
	 * @param[in] server_port Port number (UDP) of DTrack1 PC to send commands to
	 * @param[in] data_port   Port number (UDP) to receive tracking data from DTrack1 (0 if to be chosen by SDK)
	 */
	DTrackSDK(const std::string& server_host, unsigned short server_port, unsigned short data_port);

	/**
	 * \brief General constructor. DEPRECATED.
	 *
	 * @param[in] server_host     Hostname/IP of Controller/DTrack1 PC, or multicast IP (empty string if not used)
	 * @param[in] server_port     Port number (UDP) of DTrack1 PC to send commands to (0 if not used)
	 * @param[in] data_port       Port number (UDP) to receive tracking data from DTrack (0 if to be chosen)
	 * @param[in] remote_type     Type of system to connect to
	 * @param[in] data_bufsize    Buffer size for receiving tracking data in bytes; 0 to set default (32768) 
	 * @param[in] data_timeout_us Timeout for receiving tracking data in us; 0 to set default (1.0 s)
	 * @param[in] srv_timeout_us  Timeout for reply of Controller in us; 0 to set default (10.0 s)
	 */
	DTrackSDK(const std::string& server_host,
	          unsigned short server_port,
	          unsigned short data_port,
	          RemoteSystemType remote_type,
	          int data_bufsize = 0,
	          int data_timeout_us = 0,
	          int srv_timeout_us = 0
	);

	/**
	 * \brief Destructor.
	 */
	~DTrackSDK();


	/**
	 * \brief Returns if UDP socket is open to receive tracking data on local machine.
	 *
	 * Needed to receive DTrack UDP data, but does not guarantee this.
	 * Especially in case no data is sent to this port.
	 *
	 * @return Socket is open?
	 */
	bool isDataInterfaceValid() const;

	/**
	 * \brief Alias for isDataInterfaceValid(). DEPRECATED.
	 *
	 * Due to compatibility to DTrackSDK v2.0.0.
	 *
	 * @return Socket is open?
	 */
	bool isUDPValid() const  { return isDataInterfaceValid(); }

	/**
	 * \brief Alias for isDataInterfaceValid(). DEPRECATED.
	 *
	 * Due to compatibility to DTrackSDK v2.0.0 - v2.5.0.
	 *
	 * @return Socket is open?
	 */
	bool isLocalDataPortValid() const  { return isDataInterfaceValid(); }

	/**
	 * \brief Get UDP data port where tracking data is received.
	 *
	 * @return Data port.
	 */
	unsigned short getDataPort() const;

	/**
	 * \brief Returns if TCP connection for DTrack2/DTRACK3 commands is active.
	 *
	 * @return Command interface is active?
	 */
	bool isCommandInterfaceValid() const;

	/**
	 * \brief Alias for isCommandInterfaceValid(). DEPRECATED.
	 *
	 * Due to compatibility to DTrackSDK v2.0.0.
	 *
	 * @return Command interface is active?
	 */
	bool isTCPValid() const  { return isCommandInterfaceValid(); }

	/**
	 * \brief Returns if TCP connection has full access for DTrack2/DTRACK3 commands.
	 *
	 * @return Got full access to command interface?
	 */
	bool isCommandInterfaceFullAccess();

	/**
	 * \brief Get current remote system type (e.g. DTrack1, DTrack2/DTRACK3).
	 *
	 * @return Type of remote system
	 */
	RemoteSystemType getRemoteSystemType() const;

	/**
	 * \brief Set UDP timeout for receiving tracking data.
	 *
	 * @param[in] timeout Timeout for receiving tracking data in us; 0 to set default (1.0 s)
	 * @return            Success? (i.e. valid timeout)
	 */
	bool setDataTimeoutUS( int timeout );

	/**
	 * \brief Set TCP timeout for exchanging commands with Controller.
	 *
	 * @param[in] timeout Timeout for reply of Controller in us; 0 to set default (10.0 s)
	 * @return            Success? (i.e. valid timeout)
	 */
	bool setCommandTimeoutUS( int timeout );

	/**
	 * \brief Alias for setCommandTimeoutUS(). DEPRECATED.
	 *
	 * Due to compatibility to DTrackSDK v2.0.0 - v2.5.0.
	 *
	 * @param[in] timeout Timeout for reply of Controller in us; 0 to set default (10.0 s)
	 * @return            Success? (i.e. valid timeout)
	 */
	bool setControllerTimeoutUS( int timeout )  { return setCommandTimeoutUS( timeout ); };

	/**
	 * \brief Set UDP buffer size for receiving tracking data.
	 *
	 * @param[in] bufSize Buffer size for receiving tracking data in bytes; 0 to set default (32768)
	 * @return            Success? (i.e. valid size)
	 */
	bool setDataBufferSize( int bufSize );


	/**
	 * \brief Receive and process one tracking data packet.
	 *
	 * This method waits until a data packet becomes available, but no longer
	 * than the timeout. Updates internal data structures.
	 *
	 * @return Receive succeeded?
	 */
	bool receive();

	/**
	 * \brief Process one tracking packet manually.
	 *
	 * This requires no connection to a Controller. Updates internal data structures.
	 *
	 * @param[in] data Data packet to be processed
	 * @return         Processing succeeded?
	 */
	bool processPacket( const std::string& data );

	/**
	 * \brief Get content of the UDP buffer.
	 * 
	 * @return Content of buffer as string
	 */
	std::string getBuf() const;


	/**
	 * \brief Get last error at receiving tracking data (data transmission).
	 *
	 * @return Error code (success = 0)
	 */
	Errors getLastDataError() const;

	/**
	 * \brief Get last error at exchanging commands with Controller (command transmission).
	 *
	 * @return Error code (success = 0)
	 */
	Errors getLastServerError() const;

	/**
	 * \brief Get last DTrack2/DTRACK3 command error code.
	 *
	 * @return Error code
	 */
	int getLastDTrackError() const;

	/**
	 * \brief Get last DTrack2/DTRACK3 command error description.
	 *
	 * @return Error description
	 */
	std::string getLastDTrackErrorDescription() const;


	/**
	 * \brief Start measurement.
	 *
	 * Ensure via DTrack frontend that tracking data is sent to correct UDP data port.
	 *
	 * @return Is command successful? If measurement is already running the return value is false.
	 */
	bool startMeasurement();

	/**
	 * \brief Stop measurement.
	 *
	 * @return Is command successful? If measurement is not running return value is true.
	 */
	bool stopMeasurement();


	/**
	 * \brief Send DTrack1 command via UDP.
	 *
	 * Answer is not received and therefore not processed.
	 *
	 * @param[in] command Command string
	 * @return            Sending command succeeded? If not, a DTrack error is available
	 */
	bool sendDTrack1Command( const std::string& command );

	/**
	 * \brief Alias for sendDTrack1Command(). DEPRECATED.
	 *
	 * Due to compatibility to DTrackSDK v2.0.0 - v2.6.0.
	 *
	 * @param[in] command Command string
	 * @return            Sending command succeeded? If not, a DTrack error is available
	 */
	bool sendCommand( const std::string& command )  { return sendDTrack1Command( command ); }

	/**
	 * \brief Send DTrack2/DTRACK3 command to DTrack and receive answer (TCP command interface).
	 *
	 * Answers like "dtrack2 ok" and "dtrack2 err .." are processed. Both cases are reflected in
	 * the return value. getLastDTrackError() and getLastDTrackErrorDescription() will return more information.
	 *
	 * @param[in]  command DTrack2 command string
	 * @param[out] answer  Buffer for answer; NULL if specific answer is not needed
	 * @return 0   Specific answer, needs to be parsed
	 * @return 1   Answer is "dtrack2 ok"
	 * @return 2   Answer is "dtrack2 err ..". Refer to getLastDTrackError() and getLastDTrackErrorDescription().
	 * @return <0  if error occured (-1 receive timeout, -2 wrong system type, -3 command too long,
	 *                               -9 broken tcp connection, -10 tcp connection invalid, -11 send command failed)
	 */
	int sendDTrack2Command(const std::string& command, std::string* answer = NULL);

	/**
	 * \brief Set DTrack2/DTRACK3 parameter.
	 *
	 * @param[in] category Parameter category
	 * @param[in] name     Parameter name
	 * @param[in] value    Parameter value
	 * @return             Success? (if not, a DTrack error message is available)
	 */
	bool setParam(const std::string& category, const std::string& name, const std::string& value);

	/**
	 * \brief Set DTrack2/DTRACK3 parameter using a string containing parameter category, name and new value.
	 *
	 * @param[in] parameter Complete parameter string without starting "dtrack set "
	 * @return              Success? (if not, a DTrack error message is available)
	 */
	bool setParam(const std::string& parameter);

	/**
	 * \brief Get DTrack2/DTRACK3 parameter.
	 *
	 * @param[in]  category Parameter category
	 * @param[in]  name     Parameter name
	 * @param[out] value    Parameter value
	 * @return              Success? (if not, a DTrack error message is available)
	 */
	bool getParam(const std::string& category, const std::string& name, std::string& value);

	/**
	 * \brief Get DTrack2/DTRACK3 parameter using a string containing parameter category and name.
	 *
	 * @param[in]  parameter Complete parameter string without starting "dtrack get "
	 * @param[out] value     Parameter value
	 * @return               Success? (if not, a DTrack error message is available)
	 */
	bool getParam(const std::string& parameter, std::string& value);


	/**
	 * \brief Get DTrack2/DTRACK3 event message from the Controller.
	 *
	 * Updates internal message structures. Use the appropriate methods to get the contents of the
	 * message.
	 *
	 * @return Message available?
	 */
	bool getMessage();

	/**
	 * \brief Get frame counter of last DTrack2/DTRACK3 event message.
	 *
	 * @return Frame counter
	 */
	unsigned int getMessageFrameNr() const;

	/**
	 * \brief Get error id of last DTrack2/DTRACK3 event message.
	 *
	 * @return Error id
	 */
	unsigned int getMessageErrorId() const;

	/**
	 * \brief Get origin of last DTrack2/DTRACK3 event message.
	 *
	 * @return Origin
	 */
	std::string getMessageOrigin() const;

	/**
	 * \brief Get status of last DTrack2/DTRACK3 event message.
	 *
	 * @return Status
	 */
	std::string getMessageStatus() const;

	/**
	 * \brief Get message text of last DTrack2/DTRACK3 event message.
	 *
	 * @return Message text
	 */
	std::string getMessageMsg() const;


	/**
	 * \brief Send tactile FINGERTRACKING command to set feedback on a specific finger of a specific hand.
	 *
	 * Has to be repeated at least every second; otherwise a timeout mechanism will turn off any feedback.
	 *
	 * Sends command to the sender IP address of the latest received UDP data, if no hostname or IP address
	 * of a Controller is defined.
	 *
	 * @param[in] handId   Hand id, range 0 ..
	 * @param[in] fingerId Finger id, range 0 ..
	 * @param[in] strength Strength of feedback, between 0.0 and 1.0
	 * @return             Success? (if not, a DTrack error message is available)
	 */
	bool tactileFinger( int handId, int fingerId, double strength );

	/**
	 * \brief Send tactile FINGERTRACKING command to set tactile feedback on all fingers of a specific hand.
	 *
	 * Has to be repeated at least every second; otherwise a timeout mechanism will turn off any feedback.
	 *
	 * Sends command to the sender IP address of the latest received UDP data, if no hostname or IP address
	 * of a Controller is defined.
	 *
	 * @param[in] handId   Hand id, range 0 ..
	 * @param[in] strength Strength of feedback on all fingers, between 0.0 and 1.0
	 * @return             Success? (if not, a DTrack error message is available)
	 */
	bool tactileHand( int handId, const std::vector< double >& strength );

	/**
	 * \brief Send tactile FINGERTRACKING command to turn off tactile feedback on all fingers of a specific hand.
	 *
	 * Sends command to the sender IP address of the latest received UDP data, if no hostname or IP address
	 * of a Controller is defined.
	 *
	 * @param[in] handId    Hand id, range 0 ..
	 * @param[in] numFinger Number of fingers
	 * @return              Success? (if not, a DTrack error message is available)
	 */
	bool tactileHandOff( int handId, int numFinger );


	/**
	 * \brief Send Flystick feedback command to start a beep on a specific Flystick.
	 *
	 * Sends command to the sender IP address of the latest received UDP data, if no hostname or IP address
	 * of a Controller is defined.
	 *
	 * @param[in] flystickId  Flystick id, range 0 ..
	 * @param[in] durationMs  Time duration of the beep (in milliseconds)
	 * @param[in] frequencyHz Frequency of the beep (in Hertz)
	 * @return                Success? (if not, a DTrack error message is available)
	 */
	bool flystickBeep( int flystickId, double durationMs, double frequencyHz );

	/**
	 * \brief Send Flystick feedback command to start a vibration pattern on a specific Flystick.
	 *
	 * Sends command to the sender IP address of the latest received UDP data, if no hostname or IP address
	 * of a Controller is defined.
	 *
	 * @param[in] flystickId       Flystick id, range 0 ..
	 * @param[in] vibrationPattern Vibration pattern id, range 1 ..
	 * @return                     Success? (if not, a DTrack error message is available)
	 */
	bool flystickVibration( int flystickId, int vibrationPattern );


private:

	static const unsigned short DTRACK2_PORT_COMMAND = 50105;  //!< Controller port number (TCP) for 'dtrack2' commands
	static const unsigned short DTRACK2_PORT_FEEDBACK = 50110;  //!< Controller port number (UDP) for feedback commands

	static const int DEFAULT_TCP_TIMEOUT = 10000000;  //!< default TCP timeout (in us)
	static const int DEFAULT_UDP_TIMEOUT = 1000000;   //!< default UDP timeout (in us)
	static const int DEFAULT_UDP_BUFSIZE = 32768;     //!< default UDP buffer size (in bytes)

	/**
	 * \brief Set last DTrack2/DTRACK3 command error.
	 *
	 * @param[in] newError       New error code for last operation; default is 0 (success)
	 * @param[in] newErrorString Corresponding error string if exists (optional)
	 */
	void setLastDTrackError(int newError = 0, const std::string& newErrorString = "");

	/**
	 * \brief Private init called by constructor.
	 *
	 * @param[in] server_host Hostname/IP of Controller/DTrack1 PC, or multicast IP (empty string if not used)
	 * @param[in] server_port Port number (UDP) of DTrack1 PC to send commands to (0 if not used)
	 * @param[in] data_port   Port number (UDP) to receive tracking data from DTrack (0 if to be chosen)
	 * @param[in] remote_type Type of system to connect to
	 */
	void init( const std::string& server_host, unsigned short server_port, unsigned short data_port,
	           RemoteSystemType remote_type );

	/**
	 * \brief Send feedback command via UDP.
	 *
	 * @param[in] command Command string
	 * @return            Sending command succeeded? If not, a DTrack error is available
	 */
	bool sendFeedbackCommand( const std::string& command );

	RemoteSystemType rsType;            //!< Remote system type
	Errors lastDataError;               //!< last transmission error (tracking data)
	Errors lastServerError;             //!< last transmission error (commands)

	int lastDTrackError;                //!< last DTrack error: as code
	std::string lastDTrackErrorString;  //!< last DTrack error: as string

	DTrackNet::TCP* d_tcp;              //!< socket for TCP
	int d_tcptimeout_us;                //!< timeout for receiving and sending TCP data

	DTrackNet::UDP* d_udp;              //!< socket for UDP
	unsigned int d_remoteIp;            //!< IP address of Controller/DTrack1 PC (0 if unknown)
	unsigned short d_remoteDT1Port;     //!< Port number (UDP) of DTrack1 PC to send commands to (0 if unknown)
	int d_udptimeout_us;                //!< timeout for receiving UDP data

	int d_udpbufsize;                   //!< size of UDP buffer
	char* d_udpbuf;                     //!< UDP buffer

	std::string d_message_origin;       //!< last DTrack2 message: origin of message
	std::string d_message_status;       //!< last DTrack2 message: status of message
	unsigned int d_message_framenr;     //!< last DTrack2 message: frame counter
	unsigned int d_message_errorid;     //!< last DTrack2 message: error id
	std::string d_message_msg;          //!< last DTrack2 message: message string
};


#endif  // _ART_DTRACKSDK_HPP_

