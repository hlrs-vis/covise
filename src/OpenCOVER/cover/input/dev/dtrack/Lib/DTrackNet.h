/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* DTrackNet: C header file, A.R.T. GmbH 3.5.07-17.6.13
 *
 * Functions for receiving and sending UDP/TCP packets
 * Copyright (C) 2007-2013, Advanced Realtime Tracking GmbH
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Version v2.4.0
 *
 */

#ifndef _ART_DTRACKNET_H_
#define _ART_DTRACKNET_H_

// usually the following should work; otherwise define OS_* manually:
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64)
#define OS_WIN // for MS Windows (2000, XP, Vista, 7)
#else
#define OS_UNIX // for Unix (Linux, Irix)
#endif

#ifdef OS_UNIX
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif
#ifdef OS_WIN
#include <ws2tcpip.h>
#include <winsock2.h>
#include <windows.h>
#endif

namespace DTrackSDK_Net
{

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
unsigned int ip_name2ip(const char *name);

/**
 * 	\brief	Initialize UDP socket.
 *
 *	@param[out]		sock	socket number
 *	@param[in,out]	port	port number, 0 if to be chosen by the OS
 *	@param[in]		ip		multicast ip to listen
 *	@return 0 if ok, < 0 if error occured
 */
int udp_init(void **sock, unsigned short *port, unsigned int ip = 0);

/**
 * 	\brief	Deinitialize UDP socket.
 *
 *	@param[in]	sock	socket number
 *	@param[in]	ip		multicast ip to drop
 *	@return	0 ok, -1 error
 */
int udp_exit(void *sock, unsigned int ip = 0);

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
int udp_receive(const void *sock, void *buffer, int maxlen, int tout_us);

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
int udp_send(const void *sock, void *buffer, int len, unsigned int ipaddr, unsigned short port, int tout_us);

/**
 *	\brief	Initialize client TCP socket.
 *
 *	@param[out] sock	socket number
 *	@param[in] 	ip		ip address of TCP server
 *	@param[in] 	port	port number of TCP server
 *	@return		0 if ok, <0 if error occured
 */
int tcp_client_init(void **sock, unsigned int ip, unsigned short port);

/**
 * 	\brief	Deinitialize TCP socket
 *
 *	@param[in]	sock	socket number
 *	@return		0 ok, -1 error
 */
int tcp_exit(void *sock);

/**
 * 	\brief	Receive TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[out] buffer	buffer for TCP data
 *	@param[in]	maxlen	length of buffer
 *	@param[in]	tout_us	timeout in us (micro sec)
 *	@return		number of received bytes, <0 if error/timeout occured, -9 broken connection
 */
int tcp_receive(const void *sock, void *buffer, int maxlen, int tout_us);

/**
 * 	\brief	Send TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[in] 	buffer	buffer for TCP data
 *	@param[in] 	len		length of buffer
 *	@param[in] 	tout_us	timeout in us (micro sec)
 *	@return	0 if ok, <0 if error/timeout occured
 */
int tcp_send(const void *sock, const void *buffer, int len, int tout_us);
}

#endif // _ART_DTRACKNET_H_
