/* DTrackNet: C++ source file
 *
 * DTrackSDK: functions for receiving and sending UDP/TCP packets.
 *
 * Copyright 2007-2021, Advanced Realtime Tracking GmbH & Co. KG
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
 * Version v2.7.0
 * 
 */

#include "DTrackNet.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>

// usually the following should work; otherwise define OS_* manually:
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64)
	#define OS_WIN   // for MS Windows (2000, XP, Vista, 7, 8, 10)
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

namespace DTrackNet {

/**
 * \brief Internal socket type.
 */
struct _ip_socket_struct {
#ifdef OS_UNIX
	int ossock;  // Unix socket
#endif
#ifdef OS_WIN
	SOCKET ossock;  // Windows socket
#endif
};


/*
 * Initialize network ressources.
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


/*
 * Free network ressources.
 */
void net_exit(void)
{
#ifdef OS_WIN
	WSACleanup();
#endif
}


/*
 * Convert string to IP address (IPv4 only).
 */
unsigned int ip_name2ip( const char* name )
{
	int err;
	struct addrinfo hints, *res;

	memset( &hints, 0, sizeof( hints ) );
	hints.ai_family = AF_INET;  // only IPv4 supported
	hints.ai_socktype = SOCK_STREAM;

	res = NULL;
	err = getaddrinfo( name, NULL, &hints, &res );
	if ( err != 0 || res == NULL )
		return 0;

	unsigned int ip;
	struct sockaddr_in sin;
	memcpy( &sin, res->ai_addr, sizeof( sin ) );  // casting is causing warnings (-Wcast-align) by some compilers
	ip = ntohl( ( (struct in_addr )( sin.sin_addr ) ).s_addr );

	freeaddrinfo( res );
	return ip;
}


// ---------------------------------------------------------------------------------------------------
// Handling UDP data:
// ---------------------------------------------------------------------------------------------------

/*
 * Initialize UDP socket.
 */
UDP::UDP( unsigned short port, unsigned int multicastIp )
	: m_isValid( false ), m_socket( NULL ), m_port( port ), m_multicastIp( 0 ), m_remoteIp( 0 )
{
	struct _ip_socket_struct* s;
	struct sockaddr_in addr;
#ifdef OS_UNIX
	socklen_t addrlen;
#endif
#ifdef OS_WIN
	int addrlen;
#endif

	s = new struct _ip_socket_struct();

	// create socket:
#ifdef OS_UNIX
	s->ossock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s->ossock < 0)
	{
		delete s;
		return;
	}
#endif
#ifdef OS_WIN
	s->ossock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (s->ossock == INVALID_SOCKET)
	{
		delete s;
		return;
	}
#endif
	m_socket = s;

	if ( multicastIp != 0 )
	{
		// set reuse port to on to allow multiple binds per host
		int flag_on = 1;
		if ( setsockopt( m_socket->ossock, SOL_SOCKET, SO_REUSEADDR, (char* )&flag_on, sizeof( flag_on ) ) < 0 )
		{
			perror("setsockopt() failed1");
			return;
		}
	}
	
	// name socket:
	addr.sin_family = AF_INET;
	addr.sin_port = htons( m_port );
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addrlen = sizeof(addr);
	if ( bind( m_socket->ossock, (struct sockaddr *)&addr, addrlen ) < 0 )
		return;

	if ( m_port == 0 )
	{
		// port number was chosen by the OS
		if ( getsockname( m_socket->ossock, (struct sockaddr *)&addr, &addrlen ) )
			return;

		m_port = ntohs( addr.sin_port );
	}
	
	if ( multicastIp != 0 )
	{
		// construct an IGMP join request structure
		struct ip_mreq ipmreq;
		ipmreq.imr_multiaddr.s_addr = htonl( multicastIp );
		ipmreq.imr_interface.s_addr = htonl(INADDR_ANY);
		// send an ADD MEMBERSHIP message via setsockopt
		if ( setsockopt( m_socket->ossock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char* )&ipmreq, sizeof( ipmreq ) ) < 0 )
		{
			perror("setsockopt() failed2");
			return;
		}
		m_multicastIp = multicastIp;
	}
	
	m_isValid = true;	
}


/*
 * Deinitialize UDP socket.
 */
UDP::~UDP()
{
	if ( m_socket == NULL )  return;

	if ( m_multicastIp != 0 )
	{
		struct ip_mreq ipmreq;
		ipmreq.imr_multiaddr.s_addr = htonl( m_multicastIp );
		ipmreq.imr_interface.s_addr = htonl(INADDR_ANY);
		// send a DROP MEMBERSHIP message via setsockopt
		if ( setsockopt( m_socket->ossock, IPPROTO_IP, IP_DROP_MEMBERSHIP, (char* )&ipmreq, sizeof( ipmreq ) ) < 0 )
		{
			perror("setsockopt() failed3");
		}
	}

#ifdef OS_UNIX
	close( m_socket->ossock );
#endif
#ifdef OS_WIN
	closesocket( m_socket->ossock );
#endif
	delete m_socket;
}


/*
 * Returns if UDP socket is open to receive data.
 */
bool UDP::isValid()
{
	return m_isValid;
}


/*
 * Get UDP data port where data is received.
 */
unsigned short UDP::getPort()
{
	return m_port;
}


/*
 * Get IP address of the sender of the latest received data.
 */
unsigned int UDP::getRemoteIp()
{
	return m_remoteIp;
}


/*
 * Receive UDP data.
 */
int UDP::receive( void *buffer, int maxLen, int toutUs )
{
	int err;
	fd_set set;
	struct timeval tout;

	// waiting for data:
	FD_ZERO(&set);
	FD_SET( m_socket->ossock, &set );
	tout.tv_sec = toutUs / 1000000;
	tout.tv_usec = toutUs % 1000000;

	err = select( FD_SETSIZE, &set, NULL, NULL, &tout );
	switch ( err )
	{
		case 1:
			break;        // data available
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// receiving packet:
	while ( true )
	{
		struct sockaddr_in addr;
#ifdef OS_UNIX
		socklen_t addrlen;
#endif
#ifdef OS_WIN
		int addrlen;
#endif
		addrlen = sizeof( struct sockaddr_in );

		int nbytes = static_cast< int >( recvfrom( m_socket->ossock, ( char* )buffer, maxLen, 0,
		                                           ( struct sockaddr* )&addr, &addrlen ) );  // receive one packet
		if (nbytes < 0)
		{	// receive error
			return -3;
		}

		if ( addr.sin_family == AF_INET )  // only IPv4 supported
		{
			m_remoteIp = ntohl( addr.sin_addr.s_addr );
		}

		// check, if more data available: if so, receive another packet
		FD_ZERO(&set);
		FD_SET( m_socket->ossock, &set );

		tout.tv_sec = 0;   // no timeout
		tout.tv_usec = 0;
		if (select(FD_SETSIZE, &set, NULL, NULL, &tout) != 1)
		{
			// no more data available: check length of received packet and return
			if ( nbytes >= maxLen )
			{   // buffer overflow
				return -4;
			}
			return nbytes;
		}
	}
}


/*
 * Send UDP data.
 */
int UDP::send( const void* buffer, int len, unsigned int ip, unsigned short port, int toutUs )
{
	fd_set set;
	struct timeval tout;
	int err;
	struct sockaddr_in addr;

	// building address:
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl( ip );
	addr.sin_port = htons(port);

	// waiting to send data:
	FD_ZERO(&set);
	FD_SET( m_socket->ossock, &set );
	tout.tv_sec = toutUs / 1000000;
	tout.tv_usec = toutUs % 1000000;

	err = select( FD_SETSIZE, NULL, &set, NULL, &tout );
	switch ( err )
	{
		case 1:
			break;
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// sending data:
	int nbytes = static_cast< int >( sendto( m_socket->ossock, (const char* )buffer, len, 0, (struct sockaddr* )&addr,
	                                         (size_t )sizeof( struct sockaddr_in ) ) );
	if(nbytes < len)
	{	// send error
		return -3;
	}
	return 0;
}


// ---------------------------------------------------------------------------------------------------
// Handling TCP data:
// ---------------------------------------------------------------------------------------------------

/*
 * Initialize client TCP socket.
 */
TCP::TCP( unsigned int ip, unsigned short port )
	: m_isValid( false ), m_socket( NULL )
{
	struct _ip_socket_struct* s;
	struct sockaddr_in addr;

	s = new struct _ip_socket_struct();

	// create socket:
#ifdef OS_UNIX
	s->ossock = socket(PF_INET, SOCK_STREAM, 0);
	if (s->ossock < 0)
	{
		delete s;
		return;
	}
#endif
#ifdef OS_WIN
	s->ossock = socket(PF_INET, SOCK_STREAM, 0);
	if (s->ossock == INVALID_SOCKET)
	{
		delete s;
		return;
	}
#endif
	m_socket = s;

	// connect with server:
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(ip);
	addr.sin_port = htons(port);
	if ( connect( m_socket->ossock, (struct sockaddr *)&addr, (size_t )sizeof( addr ) ) != 0 )
		return;

	m_isValid = true;
}


/*
 * Deinitialize TCP socket.
 */
TCP::~TCP()
{
	if ( m_socket == NULL )  return;

#ifdef OS_UNIX
	close( m_socket->ossock );
#endif
#ifdef OS_WIN
	closesocket( m_socket->ossock );
#endif
	delete m_socket;
}


/*
 * Returns if TCP connection is active.
 */
bool TCP::isValid()
{
	return m_isValid;
}


/*
 * Receive TCP data.
 */
int TCP::receive( void *buffer, int maxLen, int toutUs )
{
	int err;
	fd_set set;
	struct timeval tout;

	// waiting for data:
	FD_ZERO(&set);
	FD_SET( m_socket->ossock, &set );
	tout.tv_sec = toutUs / 1000000;
	tout.tv_usec = toutUs % 1000000;

	err = select( FD_SETSIZE, &set, NULL, NULL, &tout );
	switch ( err )
	{
		case 1:
			break;        // data available
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// receiving packet:
	int nbytes = static_cast< int >( recv( m_socket->ossock, (char *)buffer, maxLen, 0 ) );
	if (nbytes == 0)
	{	// broken connection
		return -9;
	}
	if (nbytes < 0)
	{	// receive error
		return -3;
	}
	if (nbytes >= maxLen)
	{	// buffer overflow
		return -4;
	}
	return nbytes;
}


/*
 * Send TCP data.
 */
int TCP::send( const void* buffer, int len, int toutUs )
{
	fd_set set;
	struct timeval tout;
	int err;

	// waiting to send data:
	FD_ZERO(&set);
	FD_SET( m_socket->ossock, &set );
	tout.tv_sec = toutUs / 1000000;
	tout.tv_usec = toutUs % 1000000;

	err = select( FD_SETSIZE, NULL, &set, NULL, &tout );
	switch ( err )
	{
		case 1:
			break;
		case 0:
			return -1;    // timeout
		default:
			return -2;    // error
	}

	// sending data:
	int nbytes = static_cast< int >( sendto( m_socket->ossock, (const char* )buffer, len, 0, NULL, 0 ) );
	if (nbytes < len)
	{	// send error
		return -3;
	}
	return 0;
}


}  // namespace DTrackNet

