/* DTrackNet: C++ header file
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

#ifndef _ART_DTRACKNET_H_
#define _ART_DTRACKNET_H_

namespace DTrackNet {

struct _ip_socket_struct;  // forward declaration

/**
 * \brief Initialize network ressources.
 */
void net_init(void);

/**
 * \brief Free network ressources.
 */
void net_exit(void);


/**
 * \brief Convert string to IP address.
 *
 * @param[in] name Ipv4 dotted decimal address or hostname
 * @return         IP address, 0 if error occured
 */
unsigned int ip_name2ip(const char* name);


/**
 * \brief Handling UDP data.
 */
class UDP
{
public:

	/**
	 * \brief Initialize UDP socket.
	 *
	 * @param[in] port        Port number, 0 if to be chosen by the OS
	 * @param[in] multicastIp Multicast IP to listen (optional)
	 */
	UDP( unsigned short port, unsigned int multicastIp = 0 );

	/**
	 * \brief Deinitialize UDP socket.
	 */
	~UDP();

	/**
	 * \brief Returns if UDP socket is open to receive data.
	 *
	 * @return Socket is valid
	 */
	bool isValid();

	/**
	 * \brief Get UDP data port where data is received.
	 *
	 * @return Port number
	 */
	unsigned short getPort();

	/**
	 * \brief Get IP address of sender of latest received data.
	 *
	 * @return IPv4 address
	 */
	unsigned int getRemoteIp();

	/**
	 * \brief Receive UDP data.
	 *
	 * Tries to receive one packet, as long as data is available.
	 *
	 * @param[out] buffer Buffer for UDP data
	 * @param[in]  maxLen Length of buffer
	 * @param[in]  toutUs Timeout in us (micro seconds)
	 * @return            Number of received bytes, <0 if error/timeout occured
	 */
	int receive( void *buffer, int maxLen, int toutUs );

	/**
 	* \brief Send UDP data.
 	*
 	* @param[in] buffer Buffer for UDP data
 	* @param[in] len    Length of buffer
 	* @param[in] ip     IPv4 address to send data to
 	* @param[in] port   Port number to send data to
 	* @param[in] toutUs Timeout in us (micro sec)
 	* @return           0 if ok, <0 if error/timeout occured
 	*/
	int send( const void* buffer, int len, unsigned int ip, unsigned short port, int toutUs );

private:

	bool m_isValid;
	struct _ip_socket_struct* m_socket;
	unsigned short m_port;
	unsigned int m_multicastIp;
	unsigned int m_remoteIp;
};


/**
 * \brief Handling TCP data.
 */
class TCP
{
public:

	/**
	 * \brief Initialize client TCP socket.
	 *
	 * @param[in] ip   IP address of TCP server
	 * @param[in] port Port number of TCP server
	 */
	TCP( unsigned int ip, unsigned short port );

	/**
	 * \brief Deinitialize TCP socket.
	 */
	~TCP();

	/**
	 * \brief Returns if TCP connection is active.
	 */
	bool isValid();

	/**
	 * \brief Receive TCP data.
	 *
	 * @param[out] buffer Buffer for TCP data
	 * @param[in]  maxLen Length of buffer
	 * @param[in]  toutUs Timeout in us (micro seconds)
	 * @return            Number of received bytes, <0 if error/timeout occured, -9 broken connection
	 */
	int receive( void *buffer, int maxLen, int toutUs );

	/**
	 * \brief Send TCP data.
	 *
	 * @param[in] buffer Buffer for TCP data
	 * @param[in] len    Length of buffer
	 * @param[in] toutUs Timeout in us (micro seconds)
	 * @return           0 if ok, <0 if error/timeout occured
	 */
	int send( const void* buffer, int len, int toutUs );

private:

	bool m_isValid;
	struct _ip_socket_struct* m_socket;
};


}  // namespace DTrackNet

#endif  // _ART_DTRACKNET_H_

