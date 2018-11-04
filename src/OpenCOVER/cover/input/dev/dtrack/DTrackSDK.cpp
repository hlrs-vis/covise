/* DTrackSDK: C++ source file, A.R.T. GmbH
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

#include "DTrackSDK.hpp"
#include "DTrackParse.hpp"

#include <cstring>
#include <cstdlib>
#include <clocale>

// use Visual Studio specific method to avoid warnings
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64)
	#define strdup _strdup
#endif

using namespace DTrackSDK_Net;
using namespace DTrackSDK_Parse;

/**
 * 	\brief	Constructor. Use for listening mode.
 *
 *	@param[in]	data_port		port number to receive tracking data from DTrack
 */
DTrackSDK::DTrackSDK(unsigned short data_port)
    : DTrackParser()
{
	init("", 0, data_port, SYS_DTRACK_UNKNOWN);
}


/**
 * 	\brief	Constructor. Use for DTrack2.
 *
 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
 *	@param[in]	data_port		port number to receive tracking data from DTrack
 */
DTrackSDK::DTrackSDK(const std::string& server_host, unsigned short data_port)
    : DTrackParser()
{
	init(server_host, 50105, data_port, SYS_DTRACK_2);
}


/**
 * 	\brief	Constructor. Use for DTrack.
 *
 * 	This constructor can also be used for DTrack2. In this case server_port must be 50105.
 *
 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
 *	@param[in]	server_port		port number of DTrack
 *	@param[in]	data_port		port number to receive tracking data from DTrack
 */
DTrackSDK::DTrackSDK(const std::string& server_host, unsigned short server_port, unsigned short data_port)
    : DTrackParser()
{
	init(server_host, server_port, data_port, SYS_DTRACK_UNKNOWN);
}


/**	std::vector<DTrack_Monitor2D_Type> act_2d;        //!< array with marker data for each camera
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
DTrackSDK::DTrackSDK(const std::string& server_host, unsigned short server_port, unsigned short data_port,
                     RemoteSystemType remote_type, int data_bufsize, int data_timeout_us, int srv_timeout_us)
    : DTrackParser()
{
	init(server_host, server_port, data_port, remote_type, data_bufsize, data_timeout_us, srv_timeout_us);
}


/**
 * 	\brief	Private init called by constructor.
 *
 *	@param[in]	server_host		hostname or IP address of ARTtrack Controller (empty string if not used)
 *	@param[in]	server_port		port number of ARTtrack Controller
 *	@param[in]	data_port		port number to receive tracking data from ARTtrack Controller (0 if to be chosen)
 *	@param[in]	remote_type		type of system to connect to
 *	@param[in]	data_bufsize	size of buffer for UDP packets in bytes; default is 32kb (32768 bytes)
 *	@param[in]	data_timeout_us	timeout for receiving tracking data in us; default is 1s (1,000,000 us)
 *	@param[in]	srv_timeout_us	timeout for reply of ARTtrack Controller in us; default is 10s (10,000,000 us)
 */
void DTrackSDK::init(const std::string& server_host, unsigned short server_port, unsigned short data_port,
                     RemoteSystemType remote_type, int data_bufsize, int data_timeout_us, int srv_timeout_us)
{
	setlocale( LC_NUMERIC, "C" );
	
	rsType = remote_type;
	int err;
	
	d_udpsock = NULL;
	d_tcpsock = NULL;
	d_udpbuf = NULL;
	
	lastDataError = ERR_NONE;
	lastServerError = ERR_NONE;
	setLastDTrackError();
	
	d_udptimeout_us = data_timeout_us;
	d_tcptimeout_us = srv_timeout_us;
	d_remoteport = server_port;
	
	net_init();
	
	// parse remote address if available
	d_remote_ip = 0;
	if (!server_host.empty()) {
		d_remote_ip = ip_name2ip(server_host.c_str());
	}
	
	// create UDP socket:
	d_udpport = data_port;
	
	if ((d_remote_ip != 0) && (server_port == 0)) { // listen to multicast case
		err = udp_init(&d_udpsock, &d_udpport, d_remote_ip);
	} else { // normal case
		err = udp_init(&d_udpsock, &d_udpport);
	}
	if (err) {
		d_udpsock = NULL;
		d_udpport = 0;
		return;
	}
	
	// create UDP buffer:
	d_udpbufsize = data_bufsize;
	d_udpbuf = (char *)malloc(data_bufsize);
	if (!d_udpbuf) {
		udp_exit(d_udpsock);
		d_udpsock = NULL;
		d_udpport = 0;
		return;
	}
	
	if (d_remote_ip) {
		if (server_port == 0) { // multicast
			d_remoteport = 0;
		} else {
			if (rsType != SYS_DTRACK) {
				err = tcp_client_init(&d_tcpsock, d_remote_ip, server_port);
				if (err) {  // no connection to DTrack2 server
					// on error assuming DTrack if system is unknown
					if (rsType == SYS_DTRACK_UNKNOWN)
					{
						rsType = SYS_DTRACK;
						// DTrack will not listen to tcp port 50105 -> ignore tcp connection
					}
				} else {
					// TCP connection up, should be DTrack2
					rsType = SYS_DTRACK_2;
				}
			}
		}
	}
	// reset actual DTrack data:
	act_framecounter = 0;
	act_timestamp = -1;
	
	act_num_body = act_num_flystick = act_num_meatool = act_num_mearef = act_num_hand = act_num_human = 0;
	act_num_inertial = 0;
	act_num_marker = 0;
	
	d_message_origin = "";
	d_message_status = "";
	d_message_framenr = 0;
	d_message_errorid = 0;
	d_message_msg = "";
}


/**
 * 	\brief Destructor.
 */
DTrackSDK::~DTrackSDK()
{
	// release buffer
	free(d_udpbuf);
	
	// release sockets & net
	if ((d_remote_ip != 0) && (d_remoteport == 0)) {
		udp_exit(d_udpsock, d_remote_ip);
	} else {
		udp_exit(d_udpsock);
	}
	tcp_exit(d_tcpsock);
	net_exit();
}


/**
 * 	\brief	Set timeout for receiving tracking data.
 *
 * 	@param[in]	timeout		timeout for receiving tracking data in us; default is 1s (1,000,000 us)
 * 	@return		Success? (i.e. valid timeout)
 */
bool DTrackSDK::setDataTimeoutUS(int timeout) {
	if (timeout < 1)
		return false;
	d_udptimeout_us = timeout;
	return true;
}


/**
 * 	\brief	Set timeout for reply of ARTtrack Controller.
 *
 * 	@param[in]	timeout		timeout for reply of ARTtrack Controller in us; default is 1s (1,000,000 us)
 * 	@return		Success? (i.e. valid timeout)
 */
bool DTrackSDK::setControllerTimeoutUS(int timeout) {
	if (timeout < 1)
		return false;
	d_tcptimeout_us = timeout;
	return true;
}


/**
 * 	\brief	Get current remote system type (e.g. DTrack, DTrack2).
 *
 * 	@return	type of remote system
 */
DTrackSDK::RemoteSystemType DTrackSDK::getRemoteSystemType() const
{
	return rsType;
}

/**
 * 	\brief	Get last error as error code (data transmission).
 *
 * 	@return error code (success = 0)
 */
DTrackSDK::Errors DTrackSDK::getLastDataError() const
{
	return lastDataError;
}


/**
 * 	\brief	Get last error as error code (command transmission).
 *
 * 	@return error code (success = 0)
 */
DTrackSDK::Errors DTrackSDK::getLastServerError() const
{
	return lastServerError;
}


/**
 * 	\brief Set last dtrack error.
 *
 * 	@param[in]	newError		new error code for last operation; default is 0 (success)
 * 	@param[in]	newErrorString	corresponding error string if exists (optional)
 */
void DTrackSDK::setLastDTrackError(int newError, const std::string& newErrorString)
{
	lastDTrackError = newError;
	lastDTrackErrorString = newErrorString;
}


/**
 * 	\brief	Get last DTrack error code.
 *
 * 	@return Error code.
 */
int DTrackSDK::getLastDTrackError() const
{
	return lastDTrackError;
}


/**
 * 	\brief	Get last DTrack error description.
 *
 * 	@return Error description.
 */
std::string DTrackSDK::getLastDTrackErrorDescription() const
{
	return lastDTrackErrorString;
}


/**
 *	\brief Is UDP socket open to receive tracking data on local machine?
 *	An open socket is needed to receive data, but does not guarantee this.
 *	Especially in case no data is sent to this port.
 *
 *	Replaces valid() in older SDKs.
 *	@return	socket open?
 */
bool DTrackSDK::isLocalDataPortValid() const
{
	return (d_udpsock != NULL);
}


/**
 * 	\brief Is TCP connection for DTrack2 commands active?
 *
 *	On DTrack systems this function returns always false.
 *	@return	command interface active?
 */
bool DTrackSDK::isCommandInterfaceValid() const
{
	return (d_tcpsock != NULL);
}


/**
 *	\brief	Receive and process one tracking data packet.
 *
 *	Updates internal data structures.
 *	@return	receive succeeded?
 */
bool DTrackSDK::receive()
{
	char* s;
	int len;
	
	lastDataError = ERR_NONE;
	lastServerError = ERR_NONE;
	
	if (!isLocalDataPortValid()) {
		lastDataError = ERR_NET;
		return false;
	}
	
	// defaults:
	startFrame();
	
	// receive UDP packet:
	len = udp_receive(d_udpsock, d_udpbuf, d_udpbufsize-1, d_udptimeout_us);
	if (len == -1) {
		lastDataError = ERR_TIMEOUT;
		return false;
	}
	
	if (len <= 0) {
		lastDataError = ERR_NET;
		return false;
	}
	
	s = d_udpbuf;
	s[len] = '\0';
	
	// process lines:
	lastDataError = ERR_PARSE;
	
	do {
		if (!parseLine(&s))
			return false;
	} while((s = string_nextline(d_udpbuf, s, d_udpbufsize)));
	
	endFrame();
	
	lastDataError = ERR_NONE;
	return true;
}


/**
 *	\brief	Send DTrack command via UDP.
 *
 *	Answer is not received and therefore not processed.
 *	@param[in]	command		command string
 *	@return		sending command succeeded? if not, a DTrack error is available
 */
bool DTrackSDK::sendCommand(const std::string& command)
{
	if (!isLocalDataPortValid())
		return false;
	lastDataError = ERR_NONE;
	// dest is dtrack2
	if (rsType == SYS_DTRACK_2)	{
		// command style is dtrack?
		if (0 == strncmp(command.c_str(), "dtrack ", 7)) {
			std::string c = command.substr(7);
			// start measurement
			if (0 == strncmp(c.c_str(), "10 3",4)) {
				return startMeasurement();
			}
			// stop measurement
			if (	(0 == strncmp(c.c_str(), "10 0",4))
			        ||	(0 == strncmp(c.c_str(), "10 1",4)))
			{
				return stopMeasurement();
			}
			// simulate success of other old commands
			return true;
		}
	}
	if (udp_send(d_udpsock, (void*)command.c_str(), (unsigned int)command.length() + 1, d_remote_ip, d_remoteport, d_udptimeout_us))
	{
		lastDataError = ERR_NET;
		return false;
	}
	if (strcmp(command.c_str(), "dtrack 10 3") == 0) {
#ifdef OS_UNIX
		sleep(1);     // some delay (actually only necessary for older DTrack versions...)
#endif
#ifdef OS_WIN
		Sleep(1000);  // some delay (actually only necessary for older DTrack versions...)
#endif
	}
	return true;
}


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
int DTrackSDK::sendDTrack2Command(const std::string& command, std::string* answer)
{
	// Params via TCP are not supported in DTrack
	if (rsType != SYS_DTRACK_2)
		return -2;
	
	// reset dtrack error
	setLastDTrackError();
	
	// command too long?
	if (command.length() > DTRACK_PROT_MAXLEN) {
		lastServerError = ERR_NET;
		return -3;
	}
	
	// connection invalid
	if (!isCommandInterfaceValid()) {
		lastServerError = ERR_NET;
		return -10;
	}
	
	// send TCP command string:
	if ((tcp_send(d_tcpsock, command.c_str(), command.length() + 1, d_tcptimeout_us))) {
		lastServerError = ERR_NET;
		return -11;
	}
	
	// receive TCP response string:
	char ans[DTRACK_PROT_MAXLEN];
	int err;
	if ((err = tcp_receive(d_tcpsock, ans, DTRACK_PROT_MAXLEN, d_tcptimeout_us)) < 0) {
		
		if (err == -1) {	// timeout
			lastServerError = ERR_TIMEOUT;
		}
		else
			if (err == -9) {	// broken connection
				tcp_exit(d_tcpsock);
				d_tcpsock = NULL;
			}
			else
				lastServerError = ERR_NET;	// network error
		
		if (answer)
			*answer = "";
		
		return err;
	}
	
	// parse answer:
	
	// check for "dtrack2 ok" / no error
	if (0 == strcmp(ans, "dtrack2 ok"))
		return 1;
	
	// got error msg?
	if (0 == strncmp(ans, "dtrack2 err ", 12)) {
		char *s = ans + 12;
		int i;
		
		// parse error code
		if (!(s = string_get_i((char *)s, &i))) {
			setLastDTrackError(-1100, "SDK error -1100");
			lastServerError = ERR_PARSE;
			return -1100;
		}
		lastDTrackError = i;
		
		// parse error string
		if (!(s = string_get_quoted_text((char *)s, lastDTrackErrorString))) {
			setLastDTrackError(-1100, "SDK error -1100");
			lastServerError = ERR_PARSE;
			return -1101;
		}
		
		return 2;
	}
	
	// not 'dtrack2 ok'/'dtrack2 err ..' -> return msg
	if (answer)
		*answer = ans;
	
	lastServerError = ERR_NONE;
	return 0;
}


/**
 * 	\brief	Set DTrack2 parameter.
 *
 *	@param[in] 	category	parameter category
 *	@param[in] 	name		parameter name
 *	@param[in] 	value		parameter value
 *	@return		success? (if not, a DTrack error message is available)
 */
bool DTrackSDK::setParam(const std::string& category, const std::string& name, const std::string& value)
{
	return setParam(category + " " + name + " " + value);
}


/**
 * 	\brief	Set DTrack2 parameter.
 *
 * 	@param[in]	parameter	 complete parameter string without starting "dtrack set "
 *	@return		success? (if not, a DTrack error message is available)
 */
bool DTrackSDK::setParam(const std::string& parameter)
{
	// send command, 1 means answer "dtrack2 ok"
	return (1 == sendDTrack2Command("dtrack2 set " + parameter));
}


/**
 * 	\brief	Get DTrack2 parameter.
 *
 *	@param[in] 	category	parameter category
 *	@param[in] 	name		parameter name
 *	@param[out]	value		parameter value
 *	@return		success? (if not, a DTrack error message is available)
 */
bool DTrackSDK::getParam(const std::string& category, const std::string& name, std::string& value)
{
	return getParam(category + " " + name, value);
}


/**
 * 	\brief	Get DTrack2 parameter.
 *
 *	@param[in] 	parameter	complete parameter string without starting "dtrack get "
 *	@param[out]	value		parameter value
 *	@return		success? (if not, a DTrack error message is available)
 */
bool DTrackSDK::getParam(const std::string& parameter, std::string& value)
{
	// Params via TCP are not supported in DTrack
	if (rsType != SYS_DTRACK_2)
		return false;
	
	std::string res;
	// expected answer is "dtrack2 set" -> return value 0
	if (0 != sendDTrack2Command("dtrack2 get " + parameter, &res))
		return false;
	
	// parse parameter from answer
	if (0 == strncmp(res.c_str(), "dtrack2 set ", 12)) {
		char *str = strdup(res.c_str() + 12);
		char *s = str;
		if (!(s = string_cmp_parameter(s, parameter.c_str()))) {
			free(str);
			lastServerError = ERR_PARSE;
			return false;
		}
		
		// assign result
		value = s;
		free(str);
		return true;
	}
	
	return false;
}


/**
 *	\brief	Get DTrack2 message.
 *
 *	Updates internal message structures
 *	@return message available?
 */
bool DTrackSDK::getMessage()
{
	// Messages via TCP are not supported in DTrack
	if (rsType != SYS_DTRACK_2)
		return false;
	
	// send request
	std::string res;
	if (0 != sendDTrack2Command("dtrack2 getmsg", &res))
		return false;
	
	// check answer
	if (0 != strncmp(res.c_str(), "dtrack2 msg ", 12))
		return false;
	
	// reset values
	d_message_origin = d_message_msg = d_message_status = "";
	d_message_framenr = d_message_errorid = 0;
	
	// parse message
	const char* s = res.c_str() + 12;
	// get 'origin'
	if (!(s = string_get_word((char *)s, d_message_origin)))
		return false;
	
	// get 'status'
	if (!(s = string_get_word((char *)s, d_message_status)))
		return false;
	
	unsigned int ui;
	// get 'frame counter'
	if(!(s = string_get_ui((char *)s, &ui)))
		return false;
	d_message_framenr = ui;
	
	// get 'error id'
	if(!(s = string_get_ui((char *)s, &ui)))
		return false;
	d_message_errorid = ui;
	
	// get 'message'
	if(!(s = string_get_quoted_text((char *)s, d_message_msg)))
		return false;
	
	return true;
}


/**
 * 	\brief Get data port where tracking data is received.
 *
 *	@return Data port.
 */
unsigned short DTrackSDK::getDataPort() const
{
	return d_udpport;
}


/**
 *	\brief Get origin of last DTrack2 message.
 *
 *	@return origin
 */
std::string DTrackSDK::getMessageOrigin() const
{
	return d_message_origin;
}

/**
 *	\brief Get status of last DTrack2 message.
 *
 *	@return status
 */
std::string DTrackSDK::getMessageStatus() const
{
	return d_message_status;
}


/**
 * 	\brief Get frame counter of last DTrack2 message.
 *
 *	@return frame counter
 */
unsigned int DTrackSDK::getMessageFrameNr() const
{
	return d_message_framenr;
}


/**
 * 	\brief Get error id of last DTrack2 message.
 *
 *	@return error id
 */
unsigned int DTrackSDK::getMessageErrorId() const
{
	return d_message_errorid;
}


/**
 * 	\brief Get message string of last DTrack2 message.
 *
 *	@return mesage string
 */
std::string DTrackSDK::getMessageMsg() const
{
	return d_message_msg;
}

/**
 * 	\brief Start measurement.
 *
 *	Ensure via DTrack frontend that data is sent to the local data port.
 *	@return 	Is command successful? If measurement is already running the return value is false.
 */
bool DTrackSDK::startMeasurement()
{
	// Check for special DTrack handling
	if (rsType == SYS_DTRACK) {
		return (sendCommand("dtrack 10 3")) && (sendCommand("dtrack 31"));
	}
	
	// start tracking, 1 means answer "dtrack2 ok"
	return (1 == sendDTrack2Command("dtrack2 tracking start"));
}


/**
 * 	\brief Stop measurement.
 *
 * 	@return 	Is command successful? If measurement is not running return value is true.
 */
bool DTrackSDK::stopMeasurement()
{
	// Check for special DTrack handling
	if (rsType == SYS_DTRACK) {
		return (sendCommand("dtrack 32")) && (sendCommand("dtrack 10 0"));
	}
	
	// stop tracking, 1 means answer "dtrack2 ok"
	return (1 == sendDTrack2Command("dtrack2 tracking stop"));
}
