/* DTrackSDK: C++ header file, A.R.T. GmbH
 *
 * DTrackSDK: functions to receive and process DTrack UDP packets (ASCII protocol), as
 * well as to exchange DTrack2 TCP command strings.
 *
 * Copyright (c) 2007-2017, Advanced Realtime Tracking GmbH
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
 * Version v2.5.0
 *
 * Purpose:
 *  - receives DTrack UDP packets (ASCII protocol) and converts them into easier to handle data
 *  - sends and receives DTrack2 commands (TCP)
 *  - DTrack2 network protocol due to: 'Technical Appendix DTrack v2.0'
 *  - for ARTtrack Controller versions v0.2 (and compatible versions)
 *
 */

/*! \mainpage DTrackSDK Overview
 *
 * \section intro_sec Introduction
 *
 * The DTrackSDK provides an interface to control a DTrack/DTrack2 server and to receive tracking data.
 * Command and data exchange is done through an ASCII protocol.
 *
 * \section content_sec Contents
 *
 * This package consists of one main class: DTrackSDK. For new applications please use this class. For access
 * to tracking data refer to its ancestor class DTrackParser.
 *
 * Classes in folder 'Compatibility' provide legacy support for older SDK versions (DTracklib, DTrack, DTrack2).
 *
 * DTrackDataTypes class provides type definitions
 * DTrackNet class provides basic UDP/TCP functionality
 * DTrackParser class provides string parsing
 */

#ifndef _ART_DTRACKSDK_HPP_
#define _ART_DTRACKSDK_HPP_

#include "DTrackDataTypes.h"
#include "DTrackNet.h"
#include "DTrackParser.hpp"

#include <string>

//! Max message size
#define DTRACK_PROT_MAXLEN 200

/**
 * 	\brief DTrack SDK main class derived from DTrackParser.
 *
 * 	All methods to access tracking data are located in DTrackParser.
 */
class DTrackSDK : public DTrackParser
{
public:
	
	//! Compatibility modes for older SDKs
	typedef enum {
		SYS_DTRACK_UNKNOWN = 0,	//!< unknown system
		SYS_DTRACK,				//!< DTrack system
		SYS_DTRACK_2			//!< DTrack2 system
	} RemoteSystemType;
	
	//! Error codes
	typedef enum {
		ERR_NONE = 0,	//!< no error
		ERR_TIMEOUT,	//!< timeout occured
		ERR_NET,		//!< network error
		ERR_PARSE		//!< error while parsing command
	} Errors;
	
	/**
	 * 	\brief	Constructor. Use for listening mode.
	 *
	 *	@param[in]	data_port		port number to receive tracking data from DTrack
	 */
	DTrackSDK(unsigned short data_port);
	
	/**
	 * 	\brief	Constructor. Use for DTrack2.
	 *
	 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
	 *	@param[in]	data_port		port number to receive tracking data from DTrack
	 */
	DTrackSDK(const std::string& server_host, unsigned short data_port);
	
	/**
	 * 	\brief	Constructor. Use for DTrack.
	 *
	 * 	This constructor can also be used for DTrack2. In this case server_port must be 50105.
	 *
	 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
	 *	@param[in]	server_port		port number of DTrack
	 *	@param[in]	data_port		port number to receive tracking data from DTrack
	 */
	DTrackSDK(const std::string& server_host, unsigned short server_port, unsigned short data_port);
	
	/**
	 * 	\brief	General constructor.
	 *
	 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
	 *	@param[in]	server_port		port number of ARTtrack Controller
	 *	@param[in]	data_port		port number to receive tracking data from ARTtrack Controller (0 if to be chosen)
	 *	@param[in]	remote_type		type of system to connect to
	 *	@param[in]	data_bufsize	size of buffer for UDP packets in bytes; default is 32kb (32768 bytes)
	 *	@param[in]	data_timeout_us	timeout for receiving tracking data in us; default is 1s (1,000,000 us)
	 *	@param[in]	srv_timeout_us	timeout for reply of ARTtrack Controller in us; default is 10s (10,000,000 us)
	 */
	DTrackSDK(const std::string& server_host,
	          unsigned short server_port,
	          unsigned short data_port,
	          RemoteSystemType remote_type,
	          int data_bufsize = 32768,
	          int data_timeout_us = 1000000,
	          int srv_timeout_us = 10000000
	);
	
	/**
	 * 	\brief Destructor.
	 */
	~DTrackSDK();
	
	/**
	 *	\brief	Receive and process one tracking data packet.
	 *
	 *	Updates internal data structures.
	 *	@return	receive succeeded?
	 */
	bool receive();
	
	/**
	 * 	\brief Start measurement.
	 *
	 *	Ensure via DTrack frontend that data is sent to the local data port.
	 *	@return 	Is command successful? If measurement is already running the return value is false.
	 */
	bool startMeasurement();
	
	/**
	 * 	\brief Stop measurement.
	 *
	 * 	@return 	Is command successful? If measurement is not running return value is true.
	 */
	bool stopMeasurement();
	
	/**
	 * 	\brief Get data port where tracking data is received.
	 *
	 *	@return Data port.
	 */
	unsigned short getDataPort() const;
	
	/**
	 *	\brief Is UDP socket open to receive tracking data on local machine?
	 *	An open socket is needed to receive data, but does not guarantee this.
	 *	Especially in case no data is sent to this port.
	 *
	 *	Replaces valid() in older SDKs.
	 *	@return	socket open?
	 */
	bool isLocalDataPortValid() const;
	
	/**
	 * 	\brief Alias for isLocalDataPortValid().
	 * 	Due to compatibility to DTrackSDK v2.0.0.
	 */
	bool isUDPValid() const  { return isLocalDataPortValid(); }
	
	/**
	 * 	\brief Is TCP connection for DTrack2 commands active?
	 *
	 *	On DTrack systems this function returns always false.
	 *	@return	command interface active?
	 */
	bool isCommandInterfaceValid() const;
	
	/**
	 * 	\brief Alias for isCommandInterfaceValid().
	 * 	Due to compatibility to DTrackSDK v2.0.0.
	 */
	bool isTCPValid() const  { return isCommandInterfaceValid(); }
	
	/**
	 * 	\brief	Get current remote system type (e.g. DTrack, DTrack2).
	 *
	 * 	@return	type of remote system
	 */
	RemoteSystemType getRemoteSystemType() const;
	
	/**
	 * 	\brief	Set timeout for receiving tracking data.
	 *
	 * 	@param[in]	timeout		timeout for receiving tracking data in us; default is 1s (1,000,000 us)
	 * 	@return		Success? (i.e. valid timeout)
	 */
	bool setDataTimeoutUS(int timeout);
	
	/**
	 * 	\brief	Set timeout for reply of ARTtrack Controller.
	 *
	 * 	@param[in]	timeout		timeout for reply of ARTtrack Controller in us; default is 1s (1,000,000 us)
	 * 	@return		Success? (i.e. valid timeout)
	 */
	bool setControllerTimeoutUS(int timeout);
	
	/**
	 * 	\brief	Get last error as error code (data transmission).
	 *
	 * 	@return error code (success = 0)
	 */
	Errors getLastDataError() const;
	
	/**
	 * 	\brief	Get last error as error code (command transmission).
	 *
	 * 	@return error code (success = 0)
	 */
	Errors getLastServerError() const;
	
	/**
	 * 	\brief	Get last DTrack error code.
	 *
	 * 	@return Error code.
	 */
	int getLastDTrackError() const;
	
	/**
	 * 	\brief	Get last DTrack error description.
	 *
	 * 	@return Error description.
	 */
	std::string getLastDTrackErrorDescription() const;
	
	/**
	 *	\brief	Send DTrack command via UDP.
	 *
	 *	Answer is not received and therefore not processed.
	 *	@param[in]	command		command string
	 *	@return		sending command succeeded? if not, a DTrack error is available
	 */
	bool sendCommand(const std::string& command);
	
	/**
	 * 	\brief Send DTrack2 command to DTrack and receive answer (TCP command interface).
	 *
	 *	Answers like "dtrack2 ok" and "dtrack2 err .." are processed. Both cases are reflected in
	 *	the return value. getLastDTrackError() and getLastDTrackErrorDescription() will return more information.
	 *
	 * 	@param[in]	command	DTrack2 command string
	 * 	@param[out]	answer	buffer for answer; NULL if specific answer is not needed
	 * 	@return	0	specific answer, needs to be parsed
	 *  @return 1   answer is "dtrack2 ok"
	 *  @return 2   answer is "dtrack2 err ..". Refer to getLastDTrackError() and getLastDTrackErrorDescription().
	 * 	@return <0 if error occured (-1 receive timeout, -2 wrong system type, -3 command too long,
	 *  -9 broken tcp connection, -10 tcp connection invalid, -11 send command failed)
	 */
	int sendDTrack2Command(const std::string& command, std::string* answer = NULL);
	
	/**
	 * 	\brief	Set DTrack2 parameter.
	 *
	 *	@param[in] 	category	parameter category
	 *	@param[in] 	name		parameter name
	 *	@param[in] 	value		parameter value
	 *	@return		success? (if not, a DTrack error message is available)
	 */
	bool setParam(const std::string& category, const std::string& name, const std::string& value);
	
	/**
	 * 	\brief	Set DTrack2 parameter.
	 *
	 * 	@param[in]	parameter	 complete parameter string without starting "dtrack set "
	 *	@return		success? (if not, a DTrack error message is available)
	 */
	bool setParam(const std::string& parameter);
	
	/**
	 * 	\brief	Get DTrack2 parameter.
	 *
	 *	@param[in] 	category	parameter category
	 *	@param[in] 	name		parameter name
	 *	@param[out]	value		parameter value
	 *	@return		success? (if not, a DTrack error message is available)
	 */
	bool getParam(const std::string& category, const std::string& name, std::string& value);
	
	/**
	 * 	\brief	Get DTrack2 parameter.
	 *
	 *	@param[in] 	parameter	complete parameter string without starting "dtrack get "
	 *	@param[out]	value		parameter value
	 *	@return		success? (if not, a DTrack error message is available)
	 */
	bool getParam(const std::string& parameter, std::string& value);
	
	/**
	 *	\brief	Get DTrack2 message.
	 *
	 *	Updates internal message structures
	 *	@return message available?
	 */
	bool getMessage();
	
	/**
	 * 	\brief Get origin of last DTrack2 message.
	 *
	 *	@return origin
	 */
	std::string getMessageOrigin() const;
	
	/**
	 * 	\brief Get status of last DTrack2 message.
	 *
	 *	@return status
	 */
	std::string getMessageStatus() const;
	
	/**
	 * 	\brief Get frame counter of last DTrack2 message.
	 *
	 *	@return frame counter
	 */
	unsigned int getMessageFrameNr() const;
	
	/**
	 * 	\brief Get error id of last DTrack2 message.
	 *
	 *	@return error id
	 */
	unsigned int getMessageErrorId() const;
	
	/**
	 * 	\brief Get message string of last DTrack2 message.
	 *
	 *	@return mesage string
	 */
	std::string getMessageMsg() const;
	
protected:
	/**
	 * 	\brief Set last dtrack error.
	 *
	 * 	@param[in]  newError        new error code for last operation; default is 0 (success)
	 * 	@param[in]  newErrorString  corresponding error string if exists (optional)
	 */
	void setLastDTrackError(int newError = 0, const std::string& newErrorString = "");
	
	/**
	 * 	\brief	Private init called by constructor.
	 *
	 *	@param[in]  server_host     hostname or IP address of ARTtrack Controller (empty string if not used)
	 *	@param[in]  server_port     port number of ARTtrack Controller
	 *	@param[in]  data_port       port number to receive tracking data from ARTtrack Controller (0 if to be chosen)
	 *	@param[in]  remote_type     type of system to connect to
	 *	@param[in]  data_bufsize    size of buffer for UDP packets in bytes; default is 32kb (32768 bytes)
	 *	@param[in]  data_timeout_us timeout for receiving tracking data in us; default is 1s (1,000,000 us)
	 *	@param[in]  srv_timeout_us  timeout for reply of ARTtrack Controller in us; default is 10s (10,000,000 us)
	 */
	void init(const std::string& server_host, unsigned short server_port, unsigned short data_port,
	          RemoteSystemType remote_type, int data_bufsize = 32768, int data_timeout_us = 1000000,
	          int srv_timeout_us = 10000000
	);
	
	RemoteSystemType rsType;            //!< Remote system type
	Errors lastDataError;               //!< last transmission error (tracking data)
	Errors lastServerError;             //!< last transmission error (commands)
	
	int lastDTrackError;                //!< last DTrack error: as code
	std::string lastDTrackErrorString;  //!< last DTrack error: as string
	
	void* d_tcpsock;                    //!< socket number for TCP
	int d_tcptimeout_us;                //!< timeout for receiving and sending TCP data
	
	void* d_udpsock;                    //!< socket number for UDP
	unsigned int d_remote_ip;          //!< IP address for remote access
	unsigned short d_udpport;          //!< port number for UDP
	unsigned short d_remoteport;       //!< port number for UDP (remote) / TCP
	int d_udptimeout_us;                //!< timeout for receiving UDP data
	
	int d_udpbufsize;                   //!< size of UDP buffer
	char* d_udpbuf;                     //!< UDP buffer
	
	std::string d_message_origin;       //!< last DTrack2 message: origin of message
	std::string d_message_status;       //!< last DTrack2 message: status of message
	unsigned int d_message_framenr;     //!< last DTrack2 message: frame counter
	unsigned int d_message_errorid;     //!< last DTrack2 message: error id
	std::string d_message_msg;          //!< last DTrack2 message: message string
};


#endif /* _ART_DTRACKSDK_HPP_ */
