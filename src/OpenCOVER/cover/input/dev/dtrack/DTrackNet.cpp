/* DTrackNet: C/C++ source file, A.R.T. GmbH
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

#include "DTrackNet.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

// usually the following should work; otherwise define OS_* manually:
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64)
	#define OS_WIN   // for MS Windows (2000, XP, Vista, 7, 8)
#else
	#define OS_UNIX  // for Unix (Linux, Irix)
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

// internal socket type
struct _ip_socket_struct {
#ifdef OS_UNIX
	int ossock;		// Unix socket
#endif
#ifdef OS_WIN
	SOCKET ossock;	// Windows Socket
#endif
};

namespace DTrackSDK_Net {

/**
 * 	\brief	Initialize network ressources
 */
void net_init(void)
{
#ifdef OS_WIN
	// initialize socket dll (only Windows):
	WORD vreq;
	WSADATA wsa;// internal socket type
	vreq = MAKEWORD(2, 0);
	if (WSAStartup(vreq, &wsa) != 0)
	{
		return;
	}
#endif
}


/**
 * 	\brief	Free network ressources
 */
void net_exit(void)
{
#ifdef OS_WIN
	WSACleanup();
#endif
}


/**
 * 	\brief	Convert string to IP address.
 *
 *	@param[in]	name	IPv4 dotted decimal address or hostname
 *	@return 	IP address, 0 if error occured
 */
unsigned int ip_name2ip(const char* name)
{
	unsigned int ip;
	struct addrinfo hints, *servinfo;
	struct sockaddr_in *resulting_ipaddr;
	// try if string contains IP address:
	if (inet_pton(AF_UNSPEC, name, &ip) == 1)
	{
		return ntohl(ip);
	}
	// try if string contains hostname:
	memset(&hints, 0, sizeof(hints));
	hints.ai_flags = AI_CANONNAME;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	if (getaddrinfo(name, NULL, &hints, &servinfo) != 0)
	{
		return 0;
	}
	resulting_ipaddr = (struct sockaddr_in*)servinfo->ai_addr;
	return ntohl(((struct in_addr)(resulting_ipaddr->sin_addr)).s_addr);
}


// ---------------------------------------------------------------------------------------------------
// Handling UDP data:
// ---------------------------------------------------------------------------------------------------

/**
 * 	\brief	Initialize UDP socket.
 *
 *	@param[out]		sock	socket number
 *	@param[in,out]	port	port number, 0 if to be chosen by the OS
 *	@param[in]		ip		multicast ip to listen
 *	@return 0 if ok, < 0 if error occured
 */
int udp_init(void** sock, unsigned short* port, unsigned int ip)
{
	struct _ip_socket_struct* s;
	struct sockaddr_in addr;
#ifdef OS_UNIX
	socklen_t addrlen;
#endif
#ifdef OS_WIN
	int addrlen;
#endif
	s = (struct _ip_socket_struct *)malloc(sizeof(struct _ip_socket_struct));
	if (s == NULL)
	{
		return -11;
	}
	// create socket:
#ifdef OS_UNIX
	s->ossock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s->ossock < 0)
	{
		free(s);
		return -2;
	}
#endif
#ifdef OS_WIN
	s->ossock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s->ossock == INVALID_SOCKET)
	{
		free(s);
		return -2;
	}
#endif
	if (ip != 0) {
		// set reuse port to on to allow multiple binds per host
		int flag_on = 1;
		if ((setsockopt(s->ossock, SOL_SOCKET, SO_REUSEADDR, (char*)&flag_on, sizeof(flag_on))) < 0)
		{
			perror("setsockopt() failed1");
			udp_exit(s);
			return -4;
		}
	}
	
	// name socket:
	addr.sin_family = AF_INET;
	addr.sin_port = htons(*port);
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addrlen = sizeof(addr);
	if (bind(s->ossock, (struct sockaddr *)&addr, addrlen) < 0)
	{
		udp_exit(s);
		return -3;
	}
	if (*port == 0)
	{
		// port number was chosen by the OS
		if (getsockname(s->ossock, (struct sockaddr *)&addr, &addrlen))
		{
			udp_exit(s);
			return -3;
		}
		*port = ntohs(addr.sin_port);
	}
	
	if (ip != 0) {
		// construct an IGMP join request structure
		struct ip_mreq ipmreq;
		ipmreq.imr_multiaddr.s_addr = htonl(ip);
		ipmreq.imr_interface.s_addr = htonl(INADDR_ANY);
		// send an ADD MEMBERSHIP message via setsockopt
		if ((setsockopt(s->ossock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&ipmreq, sizeof(ipmreq))) < 0)
		{
			perror("setsockopt() failed2");
			udp_exit(s);
			return -5;
		}
	}
	
	*sock = s;
	return 0;	
}


/**
 * 	\brief	Deinitialize UDP socket.
 *
 *	@param[in]	sock	socket number
 *	@param[in]	ip		multicast ip to drop
 *	@return	0 ok, -1 error
 */
int udp_exit(void* sock, unsigned int ip)
{
	int err;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	if (sock == NULL)
	{
		return 0;
	}
	
	if (ip != 0) {
		struct ip_mreq ipmreq;
		ipmreq.imr_multiaddr.s_addr = htonl(ip);
		ipmreq.imr_interface.s_addr = htonl(INADDR_ANY);
		// send a DROP MEMBERSHIP message via setsockopt
		if ((setsockopt(s->ossock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char*)&ipmreq, sizeof(ipmreq))) < 0)
		{
			perror("setsockopt() failed3");
		}
	}
#ifdef OS_UNIX
	err = close(s->ossock);
#endif
#ifdef OS_WIN
	err = closesocket(s->ossock);
	WSACleanup();
#endif
	free(sock);
	if(err < 0)
	{
		return -1;
	}
	return 0;
}


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
int udp_receive(const void* sock, void *buffer, int maxlen, int tout_us)
{
	int nbytes, err;
	fd_set set;
	struct timeval tout;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	// waiting for data:
	FD_ZERO(&set);
	FD_SET(s->ossock, &set);
	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;
	switch ((err = select(FD_SETSIZE, &set, NULL, NULL, &tout)))
	{
		case 1:
			break;        // data available
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}
	// receiving packet:
	while (1)
	{	// receive one packet:
		nbytes = recv(s->ossock, (char *)buffer, maxlen, 0);
		if (nbytes < 0)
		{	// receive error
			return -3;
		}
		// check, if more data available: if so, receive another packet
		FD_ZERO(&set);
		FD_SET(s->ossock, &set);
		
		tout.tv_sec = 0;   // no timeout
		tout.tv_usec = 0;
		if (select(FD_SETSIZE, &set, NULL, NULL, &tout) != 1)
		{
			// no more data available: check length of received packet and return
			if (nbytes >= maxlen)
			{   // buffer overflow
				return -4;
			}
			return nbytes;
		}
	}
}


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
int udp_send(const void* sock, void* buffer, int len, unsigned int ipaddr, unsigned short port, int tout_us)
{
	fd_set set;
	struct timeval tout;
	int nbytes, err;
	struct sockaddr_in addr;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	// building address:
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(ipaddr);
	addr.sin_port = htons(port);
	// waiting to send data:
	FD_ZERO(&set);
	FD_SET(s->ossock, &set);
	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;
	switch ((err = select(FD_SETSIZE, NULL, &set, NULL, &tout)))
	{
		case 1:
			break;
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}
	// sending data:
	nbytes = sendto(s->ossock, (const char* )buffer, len, 0, (struct sockaddr* )&addr,
	                (size_t )sizeof(struct sockaddr_in));
	if(nbytes < len)
	{	// send error
		return -3;
	}
	return 0;
}

// ---------------------------------------------------------------------------------------------------
// Handling TCP data:
// ---------------------------------------------------------------------------------------------------

/**
 *	\brief	Initialize client TCP socket.
 *
 *	@param[out] sock	socket number
 *	@param[in] 	ip		ip address of TCP server
 *	@param[in] 	port	port number of TCP server
 *	@return		0 if ok, <0 if error occured
 */
int tcp_client_init(void** sock, unsigned int ip, unsigned short port)
{
	struct _ip_socket_struct* s;
	struct sockaddr_in addr;
	s = (struct _ip_socket_struct *)malloc(sizeof(struct _ip_socket_struct));
	if(s == NULL)
	{
		return -11;
	}
	// initialize socket dll (only Windows):
#ifdef OS_WIN
	{
		WORD vreq;
		WSADATA wsa;
		vreq = MAKEWORD(2, 0);
		if (WSAStartup(vreq, &wsa) != 0)
		{
			free(s);
			return -1;
		}
	}
#endif
	// create socket:
#ifdef OS_UNIX
	s->ossock = socket(PF_INET, SOCK_STREAM, 0);
	if (s->ossock < 0)
	{
		free(s);
		return -2;
	}
#endif
#ifdef OS_WIN
	s->ossock = socket(PF_INET, SOCK_STREAM, 0);
	if (s->ossock == INVALID_SOCKET)
	{
		WSACleanup();
		free(s);
		return -2;
	}
#endif
	// connect with server:
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(ip);
	addr.sin_port = htons(port);
	if ((connect(s->ossock, (struct sockaddr *)&addr, (size_t )sizeof(addr))))
	{
		tcp_exit(s);
		return -3;
	}
	*sock = s;
	return 0;
}


/**
 * 	\brief	Deinitialize TCP socket
 *
 *	@param[in]	sock	socket number
 *	@return		0 ok, -1 error
 */
int tcp_exit(void* sock)
{
	int err;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	if (sock == NULL)
	{
		return 0;
	}
#ifdef OS_UNIX
	err = close(s->ossock);
#endif
#ifdef OS_WIN
	err = closesocket(s->ossock);
	WSACleanup();
#endif
	free(sock);
	if (err < 0)
	{
		return -1;
	}
	return 0;
}


/**
 * 	\brief	Receive TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[out] buffer	buffer for TCP data
 *	@param[in]	maxlen	length of buffer
 *	@param[in]	tout_us	timeout in us (micro sec)
 *	@return		number of received bytes, <0 if error/timeout occured, -9 broken connection
 */
int tcp_receive(const void* sock, void *buffer, int maxlen, int tout_us)
{
	int nbytes, err;
	fd_set set;
	struct timeval tout;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	// waiting for data:
	FD_ZERO(&set);
	FD_SET(s->ossock, &set);
	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;
	switch ((err = select(FD_SETSIZE, &set, NULL, NULL, &tout)))
	{
		case 1:
			break;        // data available
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}
	// receiving packet:
	nbytes = recv(s->ossock, (char *)buffer, maxlen, 0);
	if (nbytes == 0)
	{	// broken connection
		return -9;
	}
	if (nbytes < 0)
	{	// receive error
		return -3;
	}
	if (nbytes >= maxlen)
	{	// buffer overflow
		return -4;
	}
	return nbytes;
}


/**
 * 	\brief	Send TCP data.
 *
 *	@param[in] 	sock	socket number
 *	@param[in] 	buffer	buffer for TCP data
 *	@param[in] 	len		length of buffer
 *	@param[in] 	tout_us	timeout in us (micro sec)
 *	@return	0 if ok, <0 if error/timeout occured
 */
int tcp_send(const void* sock, const void* buffer, int len, int tout_us)
{
	fd_set set;
	struct timeval tout;
	int nbytes, err;
	struct _ip_socket_struct* s = (struct _ip_socket_struct *)sock;
	// waiting to send data:
	FD_ZERO(&set);
	FD_SET(s->ossock, &set);
	tout.tv_sec = tout_us / 1000000;
	tout.tv_usec = tout_us % 1000000;
	switch ((err = select(FD_SETSIZE, NULL, &set, NULL, &tout)))
	{
		case 1:
			break;
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}
	// sending data:
	nbytes = send(s->ossock, (const char* )buffer, len, 0);
	if (nbytes < len)
	{	// send error
		return -3;
	}
	return 0;
}

} // end namespace
