/* DTrackNet: C header file, A.R.T. GmbH
 *
 * Functions for receiving and sending UDP/TCP packets
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
 */

#ifndef _ART_DTRACKNET_H_
#define _ART_DTRACKNET_H_

namespace DTrackSDK_Net {

/**
 * 	\brief	Initialize network ressources
 */
void net_init(void);

/**
 * 	\brief	Free network ressources
 */
void net_exit(void);

/**
 * 	\brief	Convert string to IP address.
 *
 *	@param[in]	name	IPv4 dotted decimal address or hostname
 *	@return 	IP address, 0 if error occured
 */
unsigned int ip_name2ip(const char* name);

/**
 * 	\brief	Initialize UDP socket.
 *
 *	@param[out]		sock	socket number
 *	@param[in,out]	port	port number, 0 if to be chosen by the OS
 *	@param[in]		ip		multicast ip to listen
 *	@return 0 if ok, < 0 if error occured
 */
int udp_init(void** sock, unsigned short* port, unsigned int ip = 0);

/**
 * 	\brief	Deinitialize UDP socket.
 *
 *	@param[in]	sock	socket number
 *	@param[in]	ip		multicast ip to drop
 *	@return	0 ok, -1 error
 */
int udp_exit(void* sock, unsigned int ip = 0);

/**
 *	\brief	Receive UDP data.
 *
 *	Tries to receive one packet, as long as data is available.
 *	@param[in]	sock	socket number
 *	@param[out] buffer 	buffer for UDP data
 *	@param[in] 	maxlen	length of buffer
 *	@param[in]  tout_us timeout in us (micro sec)
 *	@return	number of received bytes, <0 if error/timeout occured
 */
int udp_receive(const void* sock, void *buffer, int maxlen, int tout_us);

/**
 *	\brief	Send UDP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[in] 	buffer	buffer for UDP data
 *	@param[in] 	len		length of buffer
 *	@param[in] 	ipaddr	IPv4 address to send data to
 *	@param[in] 	port	port number to send data to
 *	@param[in] 	tout_us	timeout in us (micro sec)
 *	@return	0 if ok, <0 if error/timeout occured
 */
int udp_send(const void* sock, void* buffer, int len, unsigned int ipaddr, unsigned short port, int tout_us);

/**
 *	\brief	Initialize client TCP socket.
 *
 *	@param[out] sock	socket number
 *	@param[in] 	ip		ip address of TCP server
 *	@param[in] 	port	port number of TCP server
 *	@return		0 if ok, <0 if error occured
 */
int tcp_client_init(void** sock, unsigned int ip, unsigned short port);

/**
 * 	\brief	Deinitialize TCP socket
 *
 *	@param[in]	sock	socket number
 *	@return		0 ok, -1 error
 */
int tcp_exit(void* sock);

/**
 * 	\brief	Receive TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[out] buffer	buffer for TCP data
 *	@param[in]	maxlen	length of buffer
 *	@param[in]	tout_us	timeout in us (micro sec)
 *	@return		number of received bytes, <0 if error/timeout occured, -9 broken connection
 */
int tcp_receive(const void* sock, void *buffer, int maxlen, int tout_us);

/**
 * 	\brief	Send TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[in] 	buffer	buffer for TCP data
 *	@param[in] 	len		length of buffer
 *	@param[in] 	tout_us	timeout in us (micro sec)
 *	@return	0 if ok, <0 if error/timeout occured
 */
int tcp_send(const void* sock, const void* buffer, int len, int tout_us);

}

#endif // _ART_DTRACKNET_H_
