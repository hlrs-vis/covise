/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//****************************************************************************
// Filename:            rel_mcast.cpp
// Project Affiliation: none
// Funcionality:        Provides reliable multicast udp sockets
// Author:              Michael Poehnl
// Institution:         University of Stuttgart, HPC Center
// Operating Systems:   Linux
// Creation Date:       03-09-23
//****************************************************************************
#include "rel_mcast.h"

using std::cerr;
using std::endl;
using namespace opencover;

//----------------------------------------------------------------------------
/** Constructor for mcast server
 @param portnumber   port number
 @param number_clients   number of multicast clients
 @param addr   multicast address
 @param interfaceName  outgoing interface for multicast

*/
Rel_Mcast::Rel_Mcast(int portnumber, int number_clients, const char *addr, const char *interfaceName)
    : port(portnumber)
    , num_clients(number_clients)
    , mcast_addr(addr)
    , if_name(interfaceName)
{
    host_addrlen = sizeof(host_addr);
    sockfd = -1;
    debug_level = 1;
    sock_buffsize = 65535;
    loopback = 0;
    is_server = true;
    max_seg_size = 544;
    read_timeout = 50;
    seq_num = 0;
    msg_buff_size = 100;
    num_retry = 100;
    tail = 0;
    sync_seq_num = 0;
}

//----------------------------------------------------------------------------
/** Constructor for mcast client
 @param portnumber   port number
 @param addr   multicast address
 @param interfaceName   incoming multicast interface
*/
Rel_Mcast::Rel_Mcast(int portnumber, const char *addr, const char *interfaceName)
    : port(portnumber)
    , mcast_addr(addr)
    , if_name(interfaceName)
{
    host_addrlen = sizeof(host_addr);
    sockfd = -1;
    debug_level = 1;
    sock_buffsize = 65535;
    loopback = 0;
    is_server = false;
    max_seg_size = 544;
    read_timeout = 10;
    seq_num = 0;
    msg_buff_size = 0;
    num_retry = 3;
    tail = 0;
    sync_seq_num = 0;
}

//----------------------------------------------------------------------------
/// Destructor
Rel_Mcast::~Rel_Mcast()
{
    delete[] buffer;
    if (sockfd >= 0)
#ifdef _WIN32
        closesocket(sockfd);
#else
        if (close(sockfd))
            if (errno == EWOULDBLOCK)
                RM_TRACE(1, debug_level, "Linger time expires");
#endif
}

//----------------------------------------------------------------------------
/** Initializes a multicast server or client. Calls init_client() for multicast client
and init_server for multicast server.
*/
Rel_Mcast::RM_Error_Type Rel_Mcast::init()
{
    if (is_server)
        return init_server();
    else
        return init_client();
}

//----------------------------------------------------------------------------
/** Initializes a multicast server
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::init_server()
{
    RM_TRACE(3, debug_level, "entering init_server()");
#ifndef WIN32
    struct ifreq ifreq;
#endif
    struct in_addr mcast_if;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        RM_ERRNO(1, debug_level, "Error: socket()");
        return RM_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sock_buffsize, sizeof(sock_buffsize)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
                   (char *)&sock_buffsize, sizeof(sock_buffsize)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    memset((char *)&host_addr, 0, host_addrlen);
    host_addr.sin_family = AF_INET;
#ifdef WIN32
    host_addr.sin_addr.s_addr = inet_addr(mcast_addr);
#else
    inet_pton(AF_INET, mcast_addr, &host_addr.sin_addr);
#endif
    host_addr.sin_port = htons((unsigned short int)port);
#ifndef WIN32
    if (if_name == 0)
    {
#endif
        mcast_if.s_addr = INADDR_ANY;
#ifndef WIN32
    }
    else
    {
        strncpy(ifreq.ifr_name, if_name, IFNAMSIZ);
        if (ioctl(sockfd, SIOCGIFADDR, &ifreq))
        {
            RM_ERRNO(1, debug_level, "Error: ioctl()");
            return RM_SOCK_ERROR;
        }
        memcpy(&mcast_if, &((struct sockaddr_in *)&ifreq.ifr_addr)->sin_addr,
               sizeof(struct in_addr));
    }
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_IF,
                   &mcast_if, sizeof(mcast_if)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_LOOP,
                   &loopback, sizeof(loopback)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
#endif
    RM_TRACE(3, debug_level, "leaving init_server()");
    buffer = new char[msg_buff_size * max_seg_size];
    if (!buffer)
        return RM_ALLOC_ERROR;
    return RM_OK;
}

//----------------------------------------------------------------------------
/** Initializes a multicast client
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::init_client()
{
    RM_TRACE(3, debug_level, "entering init_client()");
#ifndef WIN32
    struct ifreq ifreq;
#endif
    struct ip_mreq mreq;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        RM_ERRNO(1, debug_level, "Error: socket()");
        return RM_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF,
                   (char *)&sock_buffsize, sizeof(sock_buffsize)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF,
                   (char *)&sock_buffsize, sizeof(sock_buffsize)))
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    memset((char *)&host_addr, 0, host_addrlen);
    host_addr.sin_family = AF_INET;
#ifdef WIN32
    host_addr.sin_addr.s_addr = inet_addr(mcast_addr);
#else
    inet_pton(AF_INET, mcast_addr, &host_addr.sin_addr);
#endif
    host_addr.sin_port = htons((unsigned short int)port);
    if (bind(sockfd, (struct sockaddr *)&host_addr, host_addrlen))
    {
        RM_ERRNO(1, debug_level, "Error: bind()");
        return RM_SOCK_ERROR;
    }
    memcpy(&mreq.imr_multiaddr, &host_addr.sin_addr, sizeof(struct in_addr));
#ifndef WIN32
    if (if_name == 0)
    {
#endif
        mreq.imr_interface.s_addr = INADDR_ANY;
#ifndef WIN32
    }
    else
    {
        strncpy(ifreq.ifr_name, if_name, IFNAMSIZ);
        if (ioctl(sockfd, SIOCGIFADDR, &ifreq))
        {
            RM_ERRNO(1, debug_level, "Error: ioctl()");
            return RM_SOCK_ERROR;
        }
        memcpy(&mreq.imr_interface,
               &((struct sockaddr_in *)&ifreq.ifr_addr)->sin_addr,
               sizeof(struct in_addr));
    }
#endif
#ifdef WIN32
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                   (const char *)&mreq, sizeof(mreq)))
#else
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                   &mreq, sizeof(mreq)))
#endif
    {
        RM_ERRNO(1, debug_level, "Error: setsockopt()");
        return RM_SOCK_ERROR;
    }
    RM_TRACE(3, debug_level, "leaving init_client()");
    buffer = new char[max_seg_size];
    return RM_OK;
}

//----------------------------------------------------------------------------
/** Sends a multicast message to all clients,
 @param data   pointer to the data to send
 @param len   number of bytes to send
 @param sync  true for synchronization after send
*/
Rel_Mcast::RM_Error_Type Rel_Mcast::write_mcast(const void *vdata, int len, bool sync)
{
    const char *data = reinterpret_cast<const char *>(vdata);
    RM_TRACE(3, debug_level, "entering write_mcast()");
    uint send_sn;
#ifdef __APPLE__
    char ssn;
#else
    uint ssn;
#endif
    struct msghdr msg_send;
    struct iovec iov_send[2];
    int left;
    int towrite;
    int message_size;

    if (!is_server)
    {
        RM_TRACE(1, debug_level, "mcast client can't send multicast messages");
        return RM_WRITE_ERROR;
    }
    send_sn = seq_num;
    left = len;
    message_size = max_seg_size + 4;
    memset(&msg_send, 0, sizeof(msg_send));
    msg_send.msg_name = (caddr_t)&host_addr;
    msg_send.msg_namelen = host_addrlen;
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 2;
    iov_send[0].iov_base = &ssn;
    iov_send[0].iov_len = 4;
    iov_send[1].iov_len = max_seg_size;
    while ((left > 0) || (send_sn != seq_num) || sync)
    {
        if ((((seq_num + 1) % msg_buff_size == tail) || (send_sn == 0xFFFFFFFF)) && (send_sn == seq_num))
        {
            if (synchronize(send_sn))
                return RM_SYNC_ERROR;
            if (send_sn == 0xFFFFFFFF)
            {
                send_sn = 0;
                seq_num = 0;
                sync_seq_num = 0;
            }
        }
        if (send_sn == seq_num)
        {
            if (left > 0)
            {
                seq_num++;
                send_sn++;
                if (left >= max_seg_size)
                    towrite = max_seg_size;
                else
                    towrite = left;
                memcpy(buffer + max_seg_size * (send_sn % msg_buff_size), data, towrite);
                iov_send[1].iov_base = (buffer + max_seg_size * (send_sn % msg_buff_size));
                ssn = htonl(send_sn);
                if (write_msg(&msg_send, message_size))
                {
                    RM_ERRNO(1, debug_level, "Error: write_msg()");
                    return RM_WRITE_ERROR;
                }
                data += towrite;
                left -= towrite;

                RM_TRACE(3, debug_level, "Sending sequenz number: " << send_sn);
            }
            else if (sync)
            {
                if (synchronize(send_sn))
                    return RM_SYNC_ERROR;
                if (send_sn == seq_num)
                {
                    seq_num++;
                    send_sn++;
                    iov_send[1].iov_base = (buffer + max_seg_size * (send_sn % msg_buff_size));
                    ssn = htonl(send_sn);
                    if (write_msg(&msg_send, message_size))
                    {
                        RM_ERRNO(1, debug_level, "Error: write_msg()");
                        return RM_WRITE_ERROR;
                    }
                    RM_TRACE(3, debug_level, "Sending sync release. SeqNum: " << send_sn);
                    RM_TRACE(3, debug_level, "leaving write_mcast()");
                    return RM_OK;
                }
            }
        }
        else
        {
            send_sn++;
            iov_send[1].iov_base = (buffer + max_seg_size * (send_sn % msg_buff_size));
            ssn = htonl(send_sn);
            if (write_msg(&msg_send, message_size))
            {
                RM_ERRNO(1, debug_level, "Error: write_msg()");
                return RM_WRITE_ERROR;
            }
            RM_TRACE(3, debug_level, "Sending sequenz number: " << send_sn);
        }
    }
    RM_TRACE(3, debug_level, "leaving write_mcast()");
    return RM_OK;
}

//----------------------------------------------------------------------------
/** Receives a multicast message from the server
 @param data   pointer to where the received data is written
 @param len   number of bytes to receive
 @param sync  true for synchronization after receive
*/
Rel_Mcast::RM_Error_Type Rel_Mcast::read_mcast(void *vdata, int len, bool sync)
{
    char *data = reinterpret_cast<char *>(vdata);
    RM_TRACE(3, debug_level, "entering read_mcast()");
    uint read_sn;
#ifdef __APPLE__
    char rsn;
    char ssn;
    char bar;
#else
    uint rsn;
    uint ssn;
    uint bar;
#endif
    struct msghdr msg_recv, msg_send;
    struct iovec iov_recv[2], iov_send[2];
    int left;
    int toread;
    int message_size;

    if (is_server)
    {
        RM_TRACE(1, debug_level, "mcast server can't receive multicast messages");
        return RM_READ_ERROR;
    }
    memset(&msg_recv, 0, sizeof(msg_recv));
    memset(&msg_send, 0, sizeof(msg_recv));
    left = len;
    message_size = max_seg_size + 4;
    msg_recv.msg_name = (caddr_t)&host_addr;
    msg_recv.msg_namelen = host_addrlen;
    msg_recv.msg_iov = iov_recv;
    msg_recv.msg_iovlen = 2;
    iov_recv[0].iov_base = &rsn;
    iov_recv[0].iov_len = 4;
    iov_recv[1].iov_base = buffer;
    iov_recv[1].iov_len = max_seg_size;
    msg_send.msg_name = (caddr_t)&host_addr;
    msg_send.msg_namelen = host_addrlen;
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 2;
    iov_send[0].iov_base = &ssn;
    iov_send[0].iov_len = 4;
    iov_send[1].iov_base = &bar;
    iov_send[1].iov_len = 4;

    while ((left > 0) || sync)
    {
        if (left >= max_seg_size)
            toread = max_seg_size;
        else
            toread = left;
        if (readable_timeo())
        {
            if (read_msg(&msg_recv, message_size))
            {
                RM_ERRNO(1, debug_level, "Error: recvfrom()");
                return RM_READ_ERROR;
            }
            read_sn = ntohl(rsn);
            RM_TRACE(3, debug_level, "Received sequenz number: " << read_sn);
            if ((read_sn == 1) && (seq_num == 0xFFFFFFFF))
                seq_num = 0;
            if (read_sn == 0)
            {
                memcpy(&bar, buffer, 4);
                if (bar)
                {
                    RM_TRACE(3, debug_level, "Synchronization");
                    ssn = htonl(seq_num);
                    if (write_msg(&msg_send, 8))
                    {
                        RM_ERRNO(1, debug_level, "Error: write_msg()");
                        return RM_WRITE_ERROR;
                    }
                }
                else
                {
                    RM_TRACE(1, debug_level, "Received abort");
                    return RM_SYNC_ERROR;
                }
            }
            else if (read_sn == seq_num + 1)
            {
                seq_num++;
                if (left == 0)
                {
                    RM_TRACE(3, debug_level, "Received sync release");
                    RM_TRACE(3, debug_level, "leaving read_mcast()");
                    return RM_OK;
                }
                memcpy(data, buffer, toread);
                left -= toread;
                data += toread;
            }
            else if (read_sn > seq_num + 1)
            {
                RM_TRACE(2, debug_level, "packet loss detected (read). SeqNum: " << seq_num + 1);
            }
        }
        else
        {
            if (left == 0)
            {
                seq_num++;
                RM_TRACE(3, debug_level, "leaving read_mcast()");
                return RM_OK;
            }
            RM_TRACE(1, debug_level, "read_msg() timed out");
            return RM_TIMEOUT_ERROR;
        }
    }
    RM_TRACE(3, debug_level, "leaving read_mcast()");
    return RM_OK;
}

//----------------------------------------------------------------------------
/** function for synchronizing the clients
 @param send_sn   current sending sequence number
*/
int Rel_Mcast::synchronize(uint &send_sn)
{
    RM_TRACE(3, debug_level, "entering synchronize()");
    int size = max_seg_size + 4;
    int i;
    bool timeout;
    uint read_sn;
#ifdef __APPLE__
    char rsn;
    char sssn;
    char rssn;
    char bar = 0;
#else
    uint rsn;
    uint sssn;
    uint rssn;
    uint bar = 0;
#endif
    uint send_sync_sn = 0;
    uint read_sync_sn;
    struct msghdr msg_send, msg_recv;
    struct iovec iov_send[3], iov_recv[2];
    int attempts = 0;
    struct sockaddr c_addr;

    memset((char *)&msg_recv, 0, sizeof(msg_recv));
    msg_recv.msg_name = (caddr_t)&c_addr;
    msg_recv.msg_namelen = host_addrlen;
    msg_recv.msg_iov = iov_recv;
    msg_recv.msg_iovlen = 2;
    iov_recv[0].iov_base = &rsn;
    iov_recv[0].iov_len = 4;
    iov_recv[1].iov_base = &rssn;
    iov_recv[1].iov_len = 4;
    memset((char *)&msg_send, 0, sizeof(msg_recv));
    msg_send.msg_name = (caddr_t)&host_addr;
    msg_send.msg_namelen = host_addrlen;
    msg_send.msg_iov = iov_send;
    msg_send.msg_iovlen = 3;
    iov_send[0].iov_base = &bar;
    iov_send[0].iov_len = 4;
    iov_send[1].iov_base = &sssn;
    iov_send[1].iov_len = 4;
    iov_send[2].iov_base = buffer;
    iov_send[2].iov_len = max_seg_size - 4;

    while (attempts <= num_retry)
    {
        timeout = false;
        send_sync_sn++;
        sssn = htonl(send_sync_sn);
        if (write_msg(&msg_send, size))
        {
            RM_ERRNO(1, debug_level, "Error: write_msg()");
            return -1;
        }
        i = 0;
        while (i < num_clients)
        {
            if (readable_timeo())
            {
                if (read_msg(&msg_recv, 8))
                {
                    RM_ERRNO(1, debug_level, "Error: read_msg()");
                    return -1;
                }
                read_sync_sn = ntohl(rssn);
                if (read_sync_sn == send_sync_sn)
                {
                    read_sn = ntohl(rsn);
                    RM_TRACE(3, debug_level, "sync answer received with SeqNum: " << read_sn);
                    if ((read_sn < sync_seq_num) || (read_sn > seq_num))
                    {
                        RM_TRACE(1, debug_level, "Error: a not synced client appeared with SeqNum: " << read_sn);
                        return -1;
                    }
                    if (read_sn < send_sn)
                    {
                        send_sn = read_sn;
                        RM_TRACE(2, debug_level, "packet loss detected (sync). SeqNum: " << read_sn + 1);
                    }
                    i++;
                }
            }
            else
            {
                timeout = true;
                attempts++;
                RM_TRACE(2, debug_level, "not all clients responding. SeqNum: " << send_sn);
                break;
            }
        }
        if (!timeout)
        {
            if (data_waiting())
            {
                RM_TRACE(1, debug_level, "Error: More clients than configured??");
                return -1;
            }
            tail = (send_sn + 1) % msg_buff_size;
            sync_seq_num = send_sn;
            RM_TRACE(3, debug_level, "leaving synchronize(). synced SeqNum: " << sync_seq_num);
            return 0;
        }
    }
    RM_TRACE(1, debug_level, "Error: synchronize()");
    return -1;
}

//----------------------------------------------------------------------------
/** Writes a single multicast message
 @param msg   pointer to message to write
 @param msg_size   message size in bytes to write
*/
int Rel_Mcast::write_msg(const struct msghdr *msg, int msg_size)
{
    bool intr;
    int ret;

    do
    {
        intr = false;
        if ((ret = sendmsg(sockfd, msg, 0)) != msg_size)
        {
            if ((ret < 0) && (errno == EINTR))
            {
                RM_TRACE(3, debug_level, "Interrupted write");
                intr = true;
            }
            else
                return -1;
        }
    } while (intr);
    return 0;
}

//----------------------------------------------------------------------------
/** Reads a single multicast message
 @param msg   pointer to where received message is written
 @param msg_size   message size in bytes to read
*/
int Rel_Mcast::read_msg(struct msghdr *msg, int msg_size)
{
    bool intr;
    int ret;

    do
    {
        intr = false;
        if ((ret = recvmsg(sockfd, msg, 0)) != msg_size)
        {
            if ((ret < 0) && (errno == EINTR))
            {
                RM_TRACE(3, debug_level, "Interrupted read");
                intr = true;
            }
            else
                return -1;
        }
    } while (intr);
    return 0;
}

//----------------------------------------------------------------------------
/** Sends an abort message to all the clients
 */
int Rel_Mcast::kill_clients()
{
    RM_TRACE(3, debug_level, "entering kill_clients()");
    uchar *message;
    int size = max_seg_size + 4;
    int i;

    if (!is_server)
    {
        RM_TRACE(1, debug_level, "kill_clients() only available for the server");
        return -1;
    }
    message = new uchar[size];
    for (i = 0; i < 8; i++)
        message[i] = 0;
    if (sendto(sockfd, &message[0], size, 0, (struct sockaddr *)&host_addr,
               host_addrlen) != size)
    {
        RM_ERRNO(1, debug_level, "Error: sendto()");
        delete[] message;
        return -1;
    }
    RM_TRACE(1, debug_level, "Sending abort");
    RM_TRACE(3, debug_level, "leaving kill_clients()");
    delete[] message;
    return 0;
}

//----------------------------------------------------------------------------
/** Checks if there's data on the socket to read, for a time of length
 read_timeout. x
 @return -1 for error, 0 for timeout, > 0 for data to read
*/
int Rel_Mcast::readable_timeo()
{
    fd_set rset;
    struct timeval tv;

    FD_ZERO(&rset);
    FD_SET((unsigned int)sockfd, &rset);
    if (is_server)
    {
        tv.tv_sec = 0;
        tv.tv_usec = 1000 * read_timeout;
    }
    else
    {
        tv.tv_sec = read_timeout;
        tv.tv_usec = 0;
    }
    return (select(sockfd + 1, &rset, 0, 0, &tv));
}

//----------------------------------------------------------------------------
/** Checks if there's data on the socket to read
 */
int Rel_Mcast::data_waiting()
{
    int nbytes;

    if (ioctl(sockfd, FIONREAD, &nbytes))
    {
        RM_ERRNO(1, debug_level, "Error: ioctl()");
        return -1;
    }
    return nbytes;
}

//----------------------------------------------------------------------------
/** Sets the socket buffer size. NOT A REAL BUFFER IN UDP
 @param sbs  desired socket buffsize in bytes.
*/
void Rel_Mcast::set_sock_buffsize(int sbs)
{
    sock_buffsize = sbs;
}

//----------------------------------------------------------------------------
/** Sets the debug level. Range from 0 (lowest level) to 3 (highest level)
 @param level  debug level to set to.
*/
void Rel_Mcast::set_debuglevel(int level)
{
    debug_level = level;
}

//---------------------------------------------------------------------------- x
/** Sets the loopback support on the server side
 @param loop   1 for loopback support, 0 for no loopback
*/
void Rel_Mcast::set_loopback(uchar loop)
{
    if (!is_server)
    {
        RM_TRACE(1, debug_level, "set_loopback() only available for the server");
        return;
    }
    loopback = loop;
}

//----------------------------------------------------------------------------
/** Sets the MTU
 @param mtu   MTU in bytes
*/
void Rel_Mcast::set_mtu(int mtu)
{
    max_seg_size = mtu - 32; //20 bytes ip-header 8 bytes udp-header
    //and 4 bytes sequenz number
}

//----------------------------------------------------------------------------
/** Sets the timeout for read operations on the socket
    !!! For Server in msec and for client in sec
 @param timeout   sec/msec to wait before timeout for client/server
*/
void Rel_Mcast::set_readtimeout(int timeout)
{
    read_timeout = timeout;
}

//----------------------------------------------------------------------------
/** Sets the number of messages to hold on the server side for recovery
 @param buff_size   number of messages
*/
void Rel_Mcast::set_msg_buffer(int buff_size)
{
    if (!is_server)
    {
        RM_TRACE(1, debug_level, "set_msg_buffer() only available for the server");
        return;
    }
    msg_buff_size = buff_size;
}

//----------------------------------------------------------------------------
/** Sets the number of retries for synchronization
 @param retry   number of retries
*/
void Rel_Mcast::set_retry_counter(int retry)
{
    if (!is_server)
    {
        RM_TRACE(1, debug_level, "set_retry_counter() only available for the server");
        return;
    }
    num_retry = retry;
}

//---------------------------------------------------------------------------
/** Prints an error message.
 @param prefix  prefix for identifying the error place.
*/
void Rel_Mcast::printErrorMessage(const char *prefix)
{
    if (prefix == NULL)
        cerr << "Mcast error: ";
    else
        cerr << prefix << ": ";
    cerr << strerror(errno) << " (" << errno << ")" << endl;
}
