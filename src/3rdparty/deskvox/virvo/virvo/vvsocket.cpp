// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <cassert>
#include <sstream>

#include "vvsocket.h"
#include "vvdebugmsg.h"

#ifndef _WIN32
#include <signal.h>
#endif

#ifdef _WIN32
namespace
{
  class WinsockInit
  {
  public:
    int ErrorCode;

    WinsockInit() {
      WSADATA data;
      ErrorCode = WSAStartup(MAKEWORD(2,2), &data);
    }

    ~WinsockInit() {
      WSACleanup();
    }
  };

  void EnsureWinsockInitialized()
  {
    static const WinsockInit init;

    if (init.ErrorCode != 0)
      vvDebugMsg::msg(0, "Failed to initialize Winsock");
  }
}
#endif

//----------------------------------------------------------------------------
vvSocket::vvSocket()
  : _sockfd(VV_INVALID_SOCKET)
  , _sockBuffsize(-1)
{
  _bufflen = sizeof(_sendBuffsize);
#ifdef _WIN32
  EnsureWinsockInitialized();
#endif
}

//----------------------------------------------------------------------------
/// Destructor
vvSocket::~vvSocket()
{
  if (_sockfd == VV_INVALID_SOCKET)
    return;

#ifdef _WIN32
  if(closesocket(_sockfd))
    if (WSAGetLastError() ==  WSAEWOULDBLOCK)
      vvDebugMsg::msg(1, "Linger time expires");
#else
  if (close(_sockfd))
    if (errno ==  EWOULDBLOCK)
      vvDebugMsg::msg(1, "Linger time expires");
#endif
}

//----------------------------------------------------------------------------
/** ...........
*/
vvSocket::ErrorType vvSocket::setParameter(const SocketOption so, const float value)
{
  switch(so)
  {
  case VV_NONBLOCKING:
    {
#ifndef _WIN32
      int flags = fcntl(_sockfd, F_GETFL, 0);
      if(flags < 0)
      {
        vvDebugMsg::msg(1, "vvSocket::setSocketOption() error: Getting flags of socket failed");
        return VV_SOCKOPT_ERROR;
      }
#endif
      if (value == 0.0 ? false : true)
      {
#ifdef _WIN32
        unsigned long tru = 1;
        ioctlsocket(_sockfd, FIONBIO, &tru);
#else
        if(fcntl(_sockfd, F_SETFL, flags|O_NONBLOCK))
        {
          vvDebugMsg::msg(1, "vvSocket::setSocketOption() error: setting O_NONBLOCK on socket failed");
          return VV_OK;
        }
#endif
      }
      else
      {
#ifdef _WIN32
        unsigned long tru = 0;
        ioctlsocket(_sockfd, FIONBIO, &tru);
#else
        if(fcntl(_sockfd, F_SETFL, flags & (~O_NONBLOCK)))
        {
          vvDebugMsg::msg(1, "vvSocket::setSocketOption() error: removing O_NONBLOCK from socket failed.");
          return VV_SOCKOPT_ERROR;
        }
#endif
      }
      return VV_OK;
    }
    break;
  case VV_NO_NAGLE:
    {
      if (setsockopt(_sockfd, IPPROTO_TCP, TCP_NODELAY , (char*)&value, sizeof(value)))
      {
        vvDebugMsg::msg(1, "vvSocket::setSocketOption() error: setsockopt()");
        return VV_SOCKOPT_ERROR;
      }
      return VV_OK;
    }
    break;
  case VV_LINGER:
    {
      struct linger ling;
      ling.l_onoff = 1;
      ling.l_linger = (unsigned short)value;
      if (setsockopt(_sockfd, SOL_SOCKET, SO_LINGER, (char*)&ling, sizeof(ling)))
      {
        vvDebugMsg::msg(1, "vvSocket::setSocketOption() error: setsockopt()");
        return VV_SOCKOPT_ERROR;
      }
      return VV_OK;
    }
    break;
  case VV_BUFFSIZE:
    _sockBuffsize=(int)value;
    break;
    default:
    vvDebugMsg::msg(1, "vvSocket::setSocketOption() Unknown SocketOption-value-combination");
    break;
  }
  return VV_SOCKOPT_ERROR;
}

//---------------------------------------------------------------------------
/** Reads a string (i.e. one line) from the socket.
 @param s  pointer to where the string is written.
 @param maxLen maximum length of string to read.
 Reads at most maxLen-1 characters from the socket,
 the last character is used for '\0' termination.
 @return OK if maxLen characters were sufficient, otherwise RETRY.
*/
vvSocket::ErrorType vvSocket::readString(char* s, int maxLen)
{
  int len = 0;
  bool done = false;

  while (len<maxLen-1 && !done)
  {
    readData((uchar*)(&s[len]), 1);
    if (s[len]=='\n') done = true;
    ++len;
  }
  if(len < maxLen)
    s[len] = '\0';

  if(done)
    return VV_OK;
  else
    return VV_RETRY;
}

//---------------------------------------------------------------------------
/** Writes a string to the socket.
 @param s pointer to the string to write.
*/
vvSocket::ErrorType vvSocket::writeString(const char* s)
{
  ErrorType ret;
  size_t len = strlen(s);
  char* stemp = new char[len + 1];

  strcpy(stemp, s);
  stemp[len] = '\n';
  ret = writeData((uchar*)stemp, len+1);
  delete[] stemp;
  return ret;
}

//---------------------------------------------------------------------------
/** Reads a one byte value from the socket
 */
uchar vvSocket::read8()
{
  uchar value;
  readData(&value, 1);
  return value;
}

//---------------------------------------------------------------------------
/** Writes a one byte value to the socket
 @param value  the byte to write
*/
vvSocket::ErrorType vvSocket::write8(uchar value)
{
  return writeData(&value, 1);
}

//---------------------------------------------------------------------------
/** Reads a two byte value with given endianess from the socket
 @param end  endianess
*/
ushort vvSocket::read16(vvSocket::EndianType end)
{
  uchar buf[2];
  vvSocket::readData(buf, 2);
  if (end == VV_LITTLE_END)
    return ushort((int)buf[0] + (int)buf[1] * (int)256);
  else
    return ushort((int)buf[0] * (int)256 + (int)buf[1]);
}

//---------------------------------------------------------------------------
/** Writes a two byte value in the given endianess to the socket
 @param value  two byte value to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::write16(ushort value, vvSocket::EndianType end)
{
  uchar buf[2];
  if (end == VV_LITTLE_END)
  {
    buf[0] = (uchar)(value & 0xFF);
    buf[1] = (uchar)(value >> 8);
  }
  else
  {
    buf[0] = (uchar)(value >> 8);
    buf[1] = (uchar)(value & 0xFF);
  }
  return vvSocket::writeData(buf, 2);
}

//---------------------------------------------------------------------------
/** Reads a four byte value with given endianess from the socket
 @param end  endianess
*/
uint vvSocket::read32(vvSocket::EndianType end)
{
  uchar buf[4];
  vvSocket::readData(buf, 4);
  if (end == VV_LITTLE_END)
  {
    return uint((ulong)buf[3] * (ulong)16777216 + (ulong)buf[2] * (ulong)65536 +
      (ulong)buf[1] * (ulong)256 + (ulong)buf[0]);
  }
  else
  {
    return uint((ulong)buf[0] * (ulong)16777216 + (ulong)buf[1] * (ulong)65536 +
      (ulong)buf[2] * (ulong)256 + (ulong)buf[3]);
  }
}

//---------------------------------------------------------------------------
/** Writes a four byte value in the given endianess to the socket
 @param value  four byte value to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::write32(uint value, vvSocket::EndianType end)
{
  uchar buf[4];
  if (end == VV_LITTLE_END)
  {
    buf[0] = (uchar)(value & 0xFF);
    buf[1] = (uchar)((value >> 8)  & 0xFF);
    buf[2] = (uchar)((value >> 16) & 0xFF);
    buf[3] = (uchar)(value  >> 24);
  }
  else
  {
    buf[0] = (uchar)(value  >> 24);
    buf[1] = (uchar)((value >> 16) & 0xFF);
    buf[2] = (uchar)((value >> 8)  & 0xFF);
    buf[3] = (uchar)(value & 0xFF);
  }
  return vvSocket::writeData(buf, 4);
}

//---------------------------------------------------------------------------
/** Reads a float with given endianess from the socket
 @param end  endianess
*/
float vvSocket::readFloat(vvSocket::EndianType end)
{
  uchar buf[4];
  float  fval;
  uchar* ptr;
  uchar  tmp;

  assert(sizeof(float) == 4);
  vvSocket::readData(buf, 4);
  memcpy(&fval, buf, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    ptr = (uchar*)&fval;
    tmp = ptr[0]; ptr[0] = ptr[3]; ptr[3] = tmp;
    tmp = ptr[1]; ptr[1] = ptr[2]; ptr[2] = tmp;
  }
  return fval;
}

//---------------------------------------------------------------------------
/** Writes a float in the given endianess to the socket
 @param value  float to write
 @param end  endianess
*/
vvSocket::ErrorType vvSocket::writeFloat(float value, vvSocket::EndianType end)
{
  uchar buf[4];
  uchar tmp;

  memcpy(buf, &value, 4);
  if (getEndianness() != end)
  {
    // Reverse byte order:
    tmp = buf[0]; buf[0] = buf[3]; buf[3] = tmp;
    tmp = buf[1]; buf[1] = buf[2]; buf[2] = tmp;
  }
  return vvSocket::writeData(buf, 4);
}

//----------------------------------------------------------------------------
/** Function to read data from the socket.
 @param dataptr pointer to where the read data is written.
 @param size number of bytes to read
 @param ret pointer for return value of internal reading function (optional)
  */
vvSocket::ErrorType vvSocket::readData(uchar* dataptr, size_t size, ssize_t *ret)
{
  ssize_t s;

  if (size <= 0) return VV_OK;

  vvDebugMsg::msg(3, "waiting .....");
  s = readn((char*)dataptr, size);
  if (s == -1)
  {
    vvDebugMsg::msg(1, "Reading data failed, read_nontimeo()");
    return VV_READ_ERROR;
  }
  if (s == 0)
  {
    vvDebugMsg::msg(1, "Peer performed orderly shutdown");
    return VV_PEER_SHUTDOWN;
  }
  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::ostringstream errmsg;
    errmsg << "Getting " << s << " Bytes of Data";
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }

  if(ret) *ret = s;
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Function to write data to the socket.
 @param dataptr  pointer to the data to write.
 @param size  number of bytes to write.
 @param ret pointer for return value of internal send function (optional)
  */
vvSocket::ErrorType vvSocket::writeData(const uchar* dataptr, size_t size, ssize_t *ret)
{
  ssize_t s;

  s = writen((char*)dataptr, size);

  if (s == -1)
  {
    vvDebugMsg::msg(1, "vvSocket::writeData(): Writing data failed, writen()");
    return VV_WRITE_ERROR;
  }
  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::ostringstream errmsg;
    errmsg << "Sending " << s << " Bytes of Data";
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }

  if(ret) *ret = s;
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Returns the number of bytes currently in the socket receive buffer.
 */
int vvSocket::isDataWaiting() const
{
#ifdef _WIN32
  unsigned long nbytes;
#else
  size_t nbytes;
#endif

#ifdef _WIN32
  if(ioctlsocket(_sockfd, FIONREAD, &nbytes))
  {
    vvDebugMsg::msg(1, "Error: ioctlsocket()");
    return -1;
  }
#else
  if(ioctl(_sockfd, FIONREAD, &nbytes))
  {
    vvDebugMsg::msg(1, "Error: ioctl()");
    return -1;
  }
#endif
  return nbytes;
}

//----------------------------------------------------------------------------
/** Sets the socket file descriptor. Use with caution!
  */
void vvSocket::setSockfd(vvsock_t fd)
{
  _hostname = NULL;
  _sockfd = fd;
}

//----------------------------------------------------------------------------
/** Returns the socket file descriptor.
 */
vvsock_t vvSocket::getSockfd() const
{
  return _sockfd;
}

//----------------------------------------------------------------------------
/** Returns the actual socket receive buffer size.
 */
int vvSocket::getRecvBuffsize()
{
  if (getsockopt(_sockfd, SOL_SOCKET, SO_RCVBUF, (char *) &_recvBuffsize, &_bufflen))
  {
    vvDebugMsg::msg(1, "Error: getsockopt()");
    return -1;
  }
  return _recvBuffsize;
}

//----------------------------------------------------------------------------
/** Returns the actual socket send buffer size.
 */
int vvSocket::getSendBuffsize()
{
  if (getsockopt(_sockfd, SOL_SOCKET, SO_SNDBUF, (char *) &_sendBuffsize, &_bufflen))
  {
    vvDebugMsg::msg(1, "Error: getsockopt()");
    return -1;
  }
  return _sendBuffsize;
}

//----------------------------------------------------------------------------
/** Tries to determine the MTU. Connection must be established for getting the
real value.
*/
int vvSocket::getMTU()
{

#ifndef TCP_MAXSEG
  vvDebugMsg::msg(2, "TCP_MAXSEG is not defined, use 576 bytes for MTU");
  return 576;
#else
  int rc;
  int theMSS = 0;
  socklen_t len = sizeof( theMSS );
  rc = getsockopt( _sockfd, IPPROTO_TCP, TCP_MAXSEG, (char*) &theMSS, &len );
  if(rc == -1 || theMSS <= 0)
  {
    vvDebugMsg::msg(2, "OS doesn't support TCP_MAXSEG querry? use 576 bytes for MTU");
    return 576;
  }
  else if ( checkMssMtu( theMSS, 1500 ))
  {
    vvDebugMsg::msg(2, "ethernet, mtu=1500 bytes");
    return 1500;
  }
  else if ( checkMssMtu( theMSS, 4352 ))
  {
    vvDebugMsg::msg(2, "FDDI, mtu=4352 bytes");
    return 4352;
  }
  else if ( checkMssMtu( theMSS, 9180 ))
  {
    vvDebugMsg::msg(2, "ATM, mtu=9180 bytes");
    return 9180;
  }
  else if ( checkMssMtu( theMSS, 65280 ))
  {
    vvDebugMsg::msg(2, "HIPPI, mtu=65280 bytes");
    return 65280;
  }
  else
  {
    std::ostringstream errmsg;
    errmsg << "unknown interface, mtu set to " << theMSS+40 << " bytes";
    vvDebugMsg::msg(2, errmsg.str().c_str());
    return  theMSS + 40;
  }
#endif
}

//----------------------------------------------------------------------------
/** signal function for timeouts
 @param signo
 @param func
*/
vvSocket::Sigfunc *vvSocket::signal(int signo, vvSocket::Sigfunc *func)
{
#ifdef _WIN32
  (void)signo; // unused
  (void)func; // unused
#else
  struct sigaction  act, oact;

  act.sa_handler = func;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  if (signo == SIGALRM)
  {
#ifdef  SA_INTERRUPT
    act.sa_flags |= SA_INTERRUPT;                 // SunOS 4.x
#endif
  }
  else
  {
#ifdef  SA_RESTART
    act.sa_flags |= SA_RESTART;                   // SVR4, 44BSD
#endif
  }
  if (sigaction(signo, &act, &oact) >= 0)
    return(oact.sa_handler);
#endif
  return(SIG_ERR);
}

//----------------------------------------------------------------------------
/** Signal function for timeouts
 @param signo
 @param func
*/
                                                  /* for signal() function */
vvSocket::Sigfunc *vvSocket::Signal(int signo, vvSocket::Sigfunc *func)
{
  Sigfunc *sigfunc;

  if ( (sigfunc = signal(signo, func)) == SIG_ERR)
    vvDebugMsg::msg(1, "signal error");
  return(sigfunc);
}

//----------------------------------------------------------------------------
/** Error message if there's no nameserver available.
 */
void vvSocket::noNameServer(int)
{
  vvDebugMsg::msg(0, "Nameserver not found. Contact your system administrator! Waiting for timeout...");
  return;
}

//----------------------------------------------------------------------------
/** Error message if SIGPIPE is issued on write() / peer closed the connection.
 */
void vvSocket::peerUnreachable(int)
{
  vvDebugMsg::msg(0, "Caught signal SIGPIPE. Peer unreachable.");
  return;
}

//----------------------------------------------------------------------------
/** Interrupter
 */
void vvSocket::interrupter(int)
{
  return;                                         // just interrupt
}

//---------------------------------------------------------------------------
/** Prints an error message.
 @param prefix  prefix for identifying the error place.
*/
void vvSocket::printErrorMessage(const char* prefix) const
{
  std::ostringstream errmsg;
  if (prefix==NULL)
    errmsg << "Socket error: ";
  else errmsg << prefix << ": ";

#ifdef _WIN32
  int errno;
  errno = WSAGetLastError();
#endif
  errmsg << strerror(errno) << " (" << errno << ")";
  vvDebugMsg::msg(0, errmsg.str().c_str());
}

//----------------------------------------------------------------------------
/** Reads from a socket without a timeout. Calls readn_tcp() for TCP sockets
and readn_udp() for UDP Sockets.
 @param dataptr  pointer to where the read data is written.
 @param size  number of bytes to read
*/
vvSocket::ErrorType vvSocket::read(uchar* dataptr, size_t size)
{
  ssize_t s;

  if (size <= 0) return VV_OK;

  vvDebugMsg::msg(3, "waiting .....");
  s = readn((char*)dataptr, size);
  if (s == -1)
  {
    vvDebugMsg::msg(1, "Reading data failed, read_nontimeo()");
    return VV_READ_ERROR;
  }
  if (s == 0)
  {
    vvDebugMsg::msg(1, "Peer performed orderly shutdown");
    return VV_PEER_SHUTDOWN;
  }
  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::ostringstream errmsg;
    errmsg << "Getting " << s << " Bytes of Data";
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Writes to a socket without a timeout. Calls writen_tcp() for TCP Sockets
 and writen_udp() for UDP sockets.
 @param dataptr  pointer to the data to write.
 @param size  number of bytes to write.
*/
vvSocket::ErrorType vvSocket::write(const uchar* dataptr, size_t size)
{
  ssize_t  s;

  s = writen((char*)dataptr, size);
  if (s == -1)
  {
    vvDebugMsg::msg(1, "Writing data failed, write_nontimeo()");
    return VV_WRITE_ERROR;
  }
  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::ostringstream errmsg;
    errmsg << "Sending " << s << " Bytes of Data";
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }
  return VV_OK;
}

//----------------------------------------------------------------------------
/** Server for Measurement of the bandwidth-delay-product. Socket
 buffers are set to the measured BDP if BDP is larger than default buffer sizes.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
*/
int vvSocket::measureBdpServer()
{
#if !defined(_WIN32) && defined(VV_BDP)
  int pid, status;
  int pip[2];

  if (pipe(pip) < 0)
  {
    vvDebugMsg::msg(1, "Error pipe()");
    return -1;
  }
  if ((pid = fork()) < 0)
  {
    vvDebugMsg::msg(1, "Error fork()");
    return -1;
  }
  else if (pid == 0)
  {
    uchar* buffer;
    ErrorType retval;
    int  bdp, recvbdp, mtu, recvmtu;

    vvSocket* sock = new vvSocket(port, VV_TCP);
    sock->set_debuglevel(debuglevel);
    sock->set_sock_buffsize(1048575);
    sock->set_timer((float)connect_timer, transfer_timer);
    if ((retval = sock->init()) != vvSocket::VV_OK)
    {
      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout connection establishment");
      }
      else
      {
        vvDebugMsg::msg(1, "Socket could not be opened");
      }
      exit(-1);
    }
    if ((retval = sock->read_data((uchar *)&recvmtu, 4)) != vvSocket::VV_OK)
    {

      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      exit(-1);
    }
    mtu = ntohl(recvmtu);
    ostringstream errmsg;
    errmsg << "Received MTU: " << mtu << " bytes";
    vvDebugMsg::msg(1, errmsg.str().c_str());
    if (RTT_server(mtu - 28))
    {
      vvDebugMsg::msg(1, "error RTT_server()");
      exit(-1);
    }
    buffer = new uchar[1000000];
    for (int j=0; j< 3 ; j++)
    {
      if ((retval = sock->write_data(buffer, 1000000)) != vvSocket::VV_OK)
      {
        delete[] buffer;
        delete sock;
        if  (retval == vvSocket::VV_TIMEOUT_ERROR)
        {
          vvDebugMsg::msg(1, "Timeout write");
        }
        else
        {
          vvDebugMsg::msg(1, "Writing data failed");
        }
        exit(-1);
      }
    }
    delete[] buffer;
    if ((retval = sock->read_data((uchar *)&recvbdp, 4)) != vvSocket::VV_OK)
    {

      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      exit(-1);
    }
    bdp = ntohl(recvbdp);
    ostringstream errmsg;
    errmsg << "Received bandwidth-delay-product: " << bdp << " bytes";
    vvDebugMsg::msg(1, errmsg.str().c_str();
    if (bdp < get_recv_buffsize())
      sock_buffsize = recv_buffsize;
    else
      sock_buffsize = bdp;
    delete sock;
    close(pip[0]);
    write(pip[1],(uchar *)&sock_buffsize, 4);
    exit(0);

  }
  else
  {
    if (waitpid(pid, &status, 0) != pid)
    {
      vvDebugMsg::msg(1, "error waitpid()");
      return -1;
    }
    if (status)
      return -1;
    close(pip[1]);
    read(pip[0],(uchar *)&sock_buffsize, 4);
  }
#endif
  return 0;
}

//----------------------------------------------------------------------------
/**CLient for Measurement of the bandwidth-delay-product. Socket
 buffers are set to the measured BDP if BDP is larger than default buffer sizes.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
*/
int vvSocket::measureBdpClient()
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* buffer;
  float time, rtt;
  int speed;
  ErrorType retval;
  int sum=0;
  int sendbdp, bdp, mtu, sendmtu;

  vvSocket* sock = new vvSocket(port, hostname, VV_TCP, cl_min_port, cl_max_port);
  sock->set_debuglevel(debuglevel);
  sock->set_sock_buffsize(1048575);
  sock->set_timer((float)connect_timer, 2.0f);
  if ((retval = sock->init()) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout connect");
    }
    else
    {
      vvDebugMsg::msg(1, "Socket could not be opened");
    }
    return -1;
  }
  mtu = sock->get_MTU();
  sendmtu = htonl(mtu);
  if ((retval = sock->write_data((uchar *)&sendmtu, 4)) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout write");
    }
    else
    {
      vvDebugMsg::msg(1, "Writing data failed");
    }
    return -1;
  }
  sleep(1);
  if ((rtt = RTT_client(mtu-28)) < 0)
  {
    vvDebugMsg::msg(1, "error get_RTT()");
    return -1;
  }
  buffer = new uchar[1000000];
  for (int j=0; j< 3 ; j++)
  {
    startTime(0);
    if ((retval = sock->read_data(buffer, 1000000)) != vvSocket::VV_OK)
    {
      delete[] buffer;
      delete sock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Timeout read");
      }
      else
      {
        vvDebugMsg::msg(1, "Reading data failed");
      }
      return -1;
    }
    time =  getTime(0);
    if (j > 0)                                    // first measurement is a bad one
    {
      speed = (int)(8000000/time);
      sum += speed;
      ostringstream errmsg;
      errmsg << "speed: "<<speed<<" Kbit/s";
      vvDebugMsg::msg(3, errmsg.str().c_str();
    }
  }
  delete[] buffer;
  speed = sum/2;
  ostringstream errmsg;
  errmsg << "average speed: "<<speed<<" Kbit/s";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  if (speed > 100000)
    speed = 1000000;
  else if (speed > 10000)
    speed = 100000;
  else if (speed > 1000)
    speed = 10000;
  bdp = (int)(speed * rtt)/8;
  errmsg.str("");
  errmsg << "bandwith-delay-product: "<<bdp<<" bytes";
  vvDebugMsg::msg(2, errmsg.str().c_str());
  sendbdp = htonl(bdp);
  if ((retval = sock->write_data((uchar *)&sendbdp, 4)) != vvSocket::VV_OK)
  {
    delete sock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Timeout write");
    }
    else
    {
      vvDebugMsg::msg(1, "Writing data failed");
    }
    return -1;
  }
  if (bdp < get_recv_buffsize())
    sock_buffsize = recv_buffsize;
  else
    sock_buffsize = bdp;

  delete sock;
  sleep(1);                                       //give server time to listen
#endif
  return 0;
}

//----------------------------------------------------------------------------
/**Server for round-trip-time measurement. Needed for bandwidth-delay-product.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
 @param payload   payload size in bytes for UDP packets
*/
int vvSocket::RttServer(int payload)
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* frame;
  ErrorType retval;

  vvSocket* usock = new vvSocket(port, VV_UDP);
  usock->set_debuglevel(debuglevel);
  usock->set_sock_buffsize(65535);
  usock->set_timer((float)connect_timer, 0.5f);
  if ((retval = usock->init()) != vvSocket::VV_OK)
  {
    delete usock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Error: RTT_server(): Timeout UDP init_client()");
    }
    else
    {
      vvDebugMsg::msg(1, "Error: RTT_server(): Socket could not be opened");
    }
    return -1;
  }
  frame = new uchar[payload];
  startTime(0);
  for ( ; ;)
  {
    if ((retval = usock->read_data(frame, payload)) == vvSocket::VV_OK)
    {
      if ((retval = usock->write_data(frame, 1)) != vvSocket::VV_OK)
      {
        delete usock;
        if  (retval == vvSocket::VV_TIMEOUT_ERROR)
        {
          vvDebugMsg::msg(1, "Error: RTT_server(): Timeout write");
        }
        else
        {
          vvDebugMsg::msg(1, "Error: RTT_server(): Writing data failed");
        }
        delete[] frame;
        return -1;
      }
    }
    else if(retval != vvSocket::VV_TIMEOUT_ERROR)
    {
      delete usock;
      vvDebugMsg::msg(1, "Error: RTT_server(): Reading data failed");
      delete[] frame;
      return -1;
    }
    if (getTime(0) > 2000)
      break;
  }
  delete usock;
  delete[] frame;
#else
  (void)payload;
#endif
  return 0;
}

//----------------------------------------------------------------------------
/** CLient for round-trip-time measurement. Needed for bandwidth-delay-product.
 NOT SUPPORTED UNDER WINDOWS AND WHEN THE VV_BDP FLAG IS NOT SET
 @param payload   payload size in bytes for UDP packets
*/
float vvSocket::RttClient(int payload)
{
#if !defined(_WIN32) && defined(VV_BDP)
  uchar* frame;
  int valid_measures=0;
  float rtt;
  float sum=0;
  ErrorType retval;

  vvSocket* usock = new vvSocket(port, hostname, VV_UDP, cl_min_port, cl_max_port);
  usock->set_debuglevel(debuglevel);
  usock->set_sock_buffsize(65535);
  usock->set_timer((float)connect_timer, 0.5f);
  if ((retval = usock->init()) != vvSocket::VV_OK)
  {
    delete usock;
    if  (retval == vvSocket::VV_TIMEOUT_ERROR)
    {
      vvDebugMsg::msg(1, "Error: RTT_client(): Timeout UDP init_client()");
    }
    else
    {
      vvDebugMsg::msg(1, "Error: RTT_client(): Socket could not be opened");
    }
    return -1;
  }
  frame = new uchar[payload];
  startTime(0);
  for ( ; ; )
  {
    startTime(1);
    if ((retval = usock->write_data(frame, payload)) != vvSocket::VV_OK)
    {
      delete usock;
      if  (retval == vvSocket::VV_TIMEOUT_ERROR)
      {
        vvDebugMsg::msg(1, "Error: RTT_client(): Timeout write");
      }
      else
      {
        vvDebugMsg::msg(1, "Error: RTT_client(): Writing data failed");
      }
      delete[] frame;
      return -1;
    }
    if ((retval = usock->read_data(frame, 1)) == vvSocket::VV_OK)
    {
      valid_measures ++;
      rtt = getTime(1);
      sum += rtt;
    }
    else if (retval != vvSocket::VV_TIMEOUT_ERROR)
    {
      delete usock;
      vvDebugMsg::msg(1, "Error: RTT_client(): Reading data failed");
      delete[] frame;
      return -1;
    }
    if (getTime(0) > 2000)
      break;
  }
  delete usock;
  delete[] frame;
  if (valid_measures > 5)
  {
    rtt = sum/valid_measures;
    if (debuglevel>1)
    {
      ostringstream errmsg;
      errmsg << "average rtt: "<<rtt<<" ms";
      vvDebugMsg::msg(2, errmsg.str().c_str());
    }
    return rtt;
  }
  else
    return 0;
#else
  (void)payload;
  return 0;
#endif
}

//----------------------------------------------------------------------------
/**Checks if the MSS belongs to a well-known MTU
 @param mss   given MSS
 @param mtu   MTU to check
*/
int vvSocket::checkMssMtu(int mss, int mtu)
{
  return (mtu-40) >= mss  &&  mss >= (mtu-80);
}

//----------------------------------------------------------------------------
/** Returns the current system's endianness.
 */
vvSocket::EndianType vvSocket::getEndianness()
{
  float one = 1.0f;                               // memory representation of 1.0 on big endian machines: 3F 80 00 00
  uchar* ptr;

  ptr = (uchar*)&one;
  if (*ptr == 0x3f)
    return VV_BIG_END;
  else
  {
    assert(*ptr == 0);
    return VV_LITTLE_END;
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
