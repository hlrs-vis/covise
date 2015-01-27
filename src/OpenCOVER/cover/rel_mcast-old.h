/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//****************************************************************************
// Filename:            rel_mcast.h
// Project Affiliation: none
// Funcionality:        Header file for rel_mcast.cpp
// Author:              Michael Poehnl
// Institution:         University of Stuttgart, HPC Center
// Operating Systems:   Linux
// Creation Date:       03-09-23
//****************************************************************************

#ifndef REL_MCAST_H
#define REL_MCAST_H

#ifdef WIN32
#include <Winsock2.h>
#endif

#include <util/coTypes.h>

#include <iostream>
#include <string.h>
#ifdef WIN32
#include <MSWSock.h>
#include <windows.h>
#include <ws2tcpip.h>
//#include <windows.h>
/*
struct iovec
{
   void  *iov_base;	// BSD uses caddr_t (1003.1g requires void *)
   DWORD iov_len; // Must be size_t (1003.1g)
};

struct msghdr {
   void	*	msg_name;	// Socket name
   int		msg_namelen;	// Length of name
   struct iovec *	msg_iov;	// Data blocks
__kernel_size_t	msg_iovlen;	// Number of blocks
void 	*	msg_control;	// Per protocol magic (eg BSD file descriptor passing)
__kernel_size_t	msg_controllen;	// Length of cmsg list
unsigned	msg_flags;
};*/
#define msghdr _WSAMSG
#define msg_name name
#define msg_namelen namelen
#define msg_iov lpBuffers
#define msg_iovlen dwBufferCount
#define msg_control Control
#define msg_flags dwFlags
#define recvmsg WSARecvMsg
#define sendmsg WSASend
#define iovec _WSABUF
#define iov_base buf
#define iov_len len

#else
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <unistd.h>
#include <sys/uio.h>
#endif
namespace opencover
{
typedef unsigned char uchar; ///< abbreviation for unsigned char
typedef unsigned short ushort; ///< abbreviation for unsigned short
typedef unsigned int uint; ///< abbreviation for unsigned int
typedef unsigned long ulong; ///< abbreviation for unsigned long

#define RM_TRACE(DBG_LVL, ACT_LVL, STRING)    \
    {                                         \
        if (ACT_LVL >= DBG_LVL)               \
            std::cerr << STRING << std::endl; \
    }

#define RM_ERRNO(DBG_LVL, ACT_LVL, STRING) \
    {                                      \
        if (ACT_LVL >= DBG_LVL)            \
            printErrorMessage(STRING);     \
    }

//----------------------------------------------------------------------------
/** This class provides reliable multicast messages over UDP <BR>
       default values:
          - No loopback support, means no client is running on the server's host
          - last 100 messages to hold on the server's side for recovery
          - MTU set to 576 bytes
          - debuglevel=1 (only errors are reported)
          - 3 retries if synchronization fails
          - multicast address 224.223.222.221
          - multicast interface chosen by kernel
          - read timeout on server side 2 seconds
          - read timeout on client side 5 seconds

      <BR>
      Here is an example code fragment to generate a multicast server which sends some messsages to 3 clients and one of the clients which reads these messages.
      <BR>
      <PRE>

      For the server:

      // Create a multicast server (e.g. port number 23232 and 3 clients)
      Rel_Mcast* mc = new Rel_Mcast(23232, 3);
      char* buffer;

      buffer = new char[10000];
      // highest debug level
      mc->set_debuglevel(3);
      // No loopback support (default)
      mc->set_loopback(0);
      // Set the MTU (e.g. 1500 bytes for Ethernet)
      mc->set_mtu(1500);
      // last 50 messages to hold
      mc->set_msg_buffer(50);
      mc->set_readtimeout(2);
      //Initialization
      if (mc->init() != Rel_Mcast::RM_OK)
      {
      delete mc;
      return -1;
      }
      // write 10000 bytes without synchronization afterwards
      if(mc->write_mcast(buffer, 10000, 0) != Rel_Mcast::RM_OK)
      {
      mc->kill_clients();
      delete[] buffer;
      delete mc;
      return -1;
      }
      // write 10000 bytes with synchronization afterwards
      if(mc->write_mcast(buffer, 10000, 1) != Rel_Mcast::RM_OK)
      {
      mc->kill_clients();
      delete[] buffer;
      delete mc;
      return -1;
      }
      // just synchronize
      if(mc->write_mcast(0, 0, 1) != Rel_Mcast::RM_OK)
      {
      mc->kill_clients();
      delete[] buffer;
      delete mc;
      return -1;
      }
      delete[] buffer;
      delete mc;

      For a client:

      Rel_Mcast* mc = new Rel_Mcast(23232);
      char* buffer;

      buffer = new char[10000];
      mc->set_debuglevel(3);
      mc->set_mtu(1500);
      mc->set_readtimeout(10);
      if (mc->init() != Rel_Mcast::RM_OK)
      {
      delete[] buffer;
      delete mc;
      return -1;
      }
      if (mc->read_mcast(buffer, 10000, 0) != Rel_Mcast::RM_OK)
      {
      delete[] buffer;
      delete mc;
      return -1;
      }
      if (mc->read_mcast(buffer, 10000, 1) != Rel_Mcast::RM_OK)
      {
      delete[] buffer;
      delete mc;
      return -1;
      }
      if (mc->read_mcast(0, 0, 1) != Rel_Mcast::RM_OK)
      {
      delete[] buffer;
      delete mc;
      return -1;
      }
      delete mc;
      delete[] buffer;
      </PRE>
      @author Michael Poehnl
      */
class COVEREXPORT Rel_Mcast
{

public:
    enum RM_Error_Type /// Error Codes
    {
        RM_OK, ///< no error
        RM_TIMEOUT_ERROR,
        RM_SOCK_ERROR,
        RM_WRITE_ERROR,
        RM_READ_ERROR,
        RM_ALLOC_ERROR,
        RM_SYNC_ERROR
    };

    Rel_Mcast(int portnumber, int number_clients,
              const char *addr = "224.223.222.221", const char *interfaceName = 0);
    Rel_Mcast(int portnumber, const char *addr = "224.223.222.221",
              const char *interfaceName = 0);
    ~Rel_Mcast();
    RM_Error_Type init();
    RM_Error_Type write_mcast(const void *, int, bool);
    RM_Error_Type read_mcast(void *, int, bool);
    int kill_clients();
    void set_sock_buffsize(int sbs);
    void set_debuglevel(int);
    void set_loopback(uchar);
    void set_mtu(int);
    void set_readtimeout(int);
    void set_msg_buffer(int);
    void set_retry_counter(int);

private:
    RM_Error_Type init_server();
    RM_Error_Type init_client();
    int synchronize(uint &);
    void printErrorMessage(const char *prefix);
    int readable_timeo();
    int data_waiting();
    int write_msg(const struct msghdr *, int);
    int read_msg(struct msghdr *, int);

    struct sockaddr_in host_addr;
#ifdef __linux__
    socklen_t host_addrlen;
#else
    int host_addrlen;
#endif
    int sockfd;
    int port;
    int num_clients;
    const char *mcast_addr;
    const char *if_name;
    int read_timeout;
    int debug_level;
    char *buffer;
    int msg_buff_size;
    int sock_buffsize;
    uchar loopback;
    bool is_server;
    int max_seg_size;
    uint seq_num;
    uint sync_seq_num;
    int num_retry;
    uint tail;
};
}
#endif
