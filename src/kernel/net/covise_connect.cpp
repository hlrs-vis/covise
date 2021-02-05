/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#endif

#ifdef HAVE_OPENSSL
//#include <openssl/rsa.h>       /* SSLeay stuff */
#include <openssl/crypto.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/err.h>

#if OPENSSL_VERSION_NUMBER >= 0x10000000L
typedef const SSL_METHOD *CONST_SSL_METHOD_P;
#else
typedef SSL_METHOD *CONST_SSL_METHOD_P;
#endif
#endif

#ifdef __alpha
#include "externc_alpha.h"
#endif

#include <sys/types.h>
#include <util/unixcompat.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/socket.h>
#endif

#include "covise_host.h"
#include "covise_socket.h"
#include "udpMessage.h"
#include "udp_message_types.h"
#include "tokenbuffer.h"
#include <util/coErr.h>
#include <config/CoviseConfig.h>

#include <iostream>
#include <algorithm>

using namespace std;
using namespace covise;

static const int SIZEOF_IEEE_INT = 4;

/***********************************************************************\ 
 **                                                                     **
 **   Connection  classes Routines                 Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : These classes present the user-seeable part of the **
 **                  socket communications (if necessary).              **
 **                  Connection is the base class, ServerConecction     **
 **                  and ClientConnection are subclasses tuned for the  **
 **                  server and the client part of a socket.            **
 **                  ControllerConnection and DataManagerConnection     **
 **                  are mere functional subclasses without additional  **
 **                  data.                                              **
 **                  ConnectionList provides the data structures        **
 **                  necessary to use the select UNIX system call       **
 **                  that allows to listen to many connections at once  **
 **                                                                     **
 **   Classes      : Connection, ServerConnection, ClientConnection,    **
 **                  ControllerConnection, DataManagerConnection,       **
 **                  ConnectionList                                     **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  26.05.93  Ver 1.1 sender_type and sender_id added  **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#if defined __sgi || defined __hpux || defined _AIX
#include <strings.h>
#endif

#undef SHOWMSG
#undef DEBUG

#ifdef HAVE_OPENSSL
SSL_CTX *SSLConnection::mCTX = NULL;
#endif

int Connection::get_id()
{
    //cerr << "sock == " << sock << " id: " << sock->get_id() << endl;
    if (sock)
        return sock->get_id();
    return -1;
}; // give socket id

Connection::Connection()
{
    sock = NULL;
    convert_to = DF_NONE;
    message_to_do = 0;
    //		   LOGINFO( "message_to_do == 0");
    read_buf = new char[READ_BUFFER_SIZE];
    bytes_to_process = 0;
    remove_socket = 0L;
    hostid = -1;
    peer_id_ = 0;
    peer_type_ = Message::UNDEFINED; // prepare connection (for subclasses)
    header_int = new int[4 * SIZEOF_IEEE_INT];
};

Connection::Connection(int sfd)
{
    send_type = Message::STDINOUT;
    message_to_do = 0;
    remove_socket = 0L;
    read_buf = new char[READ_BUFFER_SIZE];
    sock = new Socket(sfd, sfd);
    //		   fprintf(stderr, "new Socket: %x\n", sock);
    hostid = -1;
#if defined _WIN32
    tru = 1;
    ioctlsocket(sfd, FIONBIO, &tru);
#else
    fcntl(sfd, F_SETFL, O_NDELAY); // this is non-blocking, since we do not know, what will arrive
#endif
    peer_id_ = 0;
    peer_type_ = Message::UNDEFINED; // initialiaze connection with existing socket
    header_int = new int[4 * SIZEOF_IEEE_INT];
};

Connection::~Connection() // close connection (for subclasses)
{
    delete[] read_buf;
    delete[] header_int;
    delete sock;
}

// give socket id
int Connection::get_id(void (*remove_func)(int))
{
    remove_socket = remove_func;
    //	fprintf(stderr, "Socket: %x\n", sock);
    return sock->get_id();
}

Host *Connection::get_host()
{
    return sock ? sock->get_host() : NULL;
}

const char *Connection::get_hostname()
{
    return sock ? sock->get_hostname() : NULL;
}

UDPConnection::UDPConnection(int id, int s_type, int p, const char* address)
	: Connection()
{
	sender_id = id;
	send_type = s_type;
	sock = (Socket*)(new UDPSocket(p, address));
	port = p;
}
#define UDP_HEADER_SIZE 2 * SIZEOF_IEEE_INT
bool covise::UDPConnection::recv_udp_msg(UdpMessage* msg)
{
	int l;
	char* read_buf_ptr;
	int* int_read_buf;


#ifdef SHOWMSG
	char tmp_str[255];
#endif
#ifdef CRAY
	int tmp_buf[4];
#endif

	msg->sender = -1;
	msg->data = DataHandle();
	msg->type = udp_msg_type::EMPTY;

	message_to_do = 0;

	if (!sock)
		return 0;

	l = sock->Read(read_buf, READ_BUFFER_SIZE, msg->m_ip);
	if (l <= 0)
	{
		return false;
	}
	//read header
	read_buf_ptr = read_buf;
	int_read_buf = (int*)read_buf_ptr;

#ifdef SHOWMSG
	LOGINFO("read_buf: %d %d", int_read_buf[0], int_read_buf[1]);
#endif


#ifdef BYTESWAP
	swap_bytes((unsigned int*)int_read_buf, 2);
#endif
	msg->type = (udp_msg_type)int_read_buf[0];
	msg->sender = int_read_buf[1];
	l -= UDP_HEADER_SIZE;
	char* data = new char[l];
	memcpy(data, read_buf + UDP_HEADER_SIZE, l);
	msg->data = DataHandle(data, l);
	return true;

}

bool covise::UDPConnection::sendMessage(const UdpMessage* msg) const{
    return send_udp_msg(msg);
}


bool covise::UDPConnection::send_udp_msg(const UdpMessage* msg, const char* ip) const
{
	/*cerr << "sending udp msg to ";
	if (ip)
	{
		cerr << ip << endl;
	}*/

	int retval = 0;
	char write_buf[WRITE_BUFFER_SIZE];
	int* write_buf_int;

	if (!sock)
		return false;

	//Compose udp header

	write_buf_int = (int*)write_buf;
	write_buf_int[0] = msg->type;
	write_buf_int[1] = msg->sender;

	swap_bytes((unsigned int*)write_buf_int, 2);


    // Decide wether the message including the COVISE header
    // fits into one packet
    if (msg->data.length() >= WRITE_BUFFER_SIZE - UDP_HEADER_SIZE)
    {
        cerr << "udp message of type " << msg->type << " is to long! length = " << msg->data.length() << endl;
        return false;
    }

	if (msg->data.length() == 0)
    {
		retval = ((UDPSocket*)sock)->writeTo(write_buf, UDP_HEADER_SIZE, ip);
    }
	else
	{
#ifdef SHOWMSG
		LOGINFO("msg->data.length(): %d", msg->data.length());
#endif
        // Write data to socket for data blocks smaller than the
        // socket write-buffer reduced by the size of the header
        memcpy(&write_buf[UDP_HEADER_SIZE], msg->data.data(), msg->data.length());
        retval = ((UDPSocket*)sock)->writeTo(write_buf, UDP_HEADER_SIZE + msg->data.length(), ip);
	}

    if (retval < 0)
    {
        cerr << "UDPSOcket writeTo failed" << endl;
        return false;
    }

    return true;
}
SimpleServerConnection::SimpleServerConnection(Socket *s)
    : ServerConnection(s)
{
    buffer[10000] = '\0';
    buflen = 0;
}

SimpleServerConnection::SimpleServerConnection(int p, int id, int s_type)
    : ServerConnection(p, id, s_type)
{
    buffer[10000] = '\0';
    buflen = 0;
}

SimpleServerConnection::SimpleServerConnection(int *p, int id, int s_type)
    : ServerConnection(p, id, s_type)
{
    buffer[10000] = '\0';
    buflen = 0;
}

ServerConnection::ServerConnection(int p, int id, int s_type)
{
    sender_id = id;
    send_type = s_type;
    port = p;
    sock = new Socket(p);
    if (sock->get_id() == -1)
    {
        delete sock;
        sock = NULL;
        port = 0;
    }
}

ServerConnection::ServerConnection(int *p, int id, int s_type)
{
    sender_id = id;
    send_type = s_type;
    sock = new Socket(p);
    if (sock->get_id() == -1)
    {
        delete sock;
        sock = NULL;
        *p = 0;
    }
    port = *p;
}

int ServerConnection::listen() // listen for connection (after bind)
{
    return sock->listen();
}

#ifdef MULTICAST
MulticastConnection::MulticastConnection(int id, int s_type, int p,
                                         char *MulticastGroup, int ttl)
    : Connection()
{
    sender_id = id;
    send_type = s_type;
    sock = (Socket *)(new MulticastSocket(MulticastGroup, p, ttl));
    port = p;
}
#endif

ClientConnection::ClientConnection(Host *h, int p, int id, int s_type,
                                   int retries, double timeout)
{
    char dataformat;
    lhost = NULL;
    if (h) // host is not local
        host = h;
	else // host is local (usually DataManagerConnection uses this)
	{
		host = lhost = new Host("localhost");
		if (!host->hasValidAddress())
		{
			host = lhost = new Host("127.0.0.1");
		}
	}
    port = p;
    sender_id = id;
    send_type = s_type;
    sock = new Socket(host, port, retries, timeout);

    if (get_id() == -1)
        return; // connection not established
    if (sock->Read(&dataformat, 1) != 1)
    {
        delete sock;
        sock = NULL;
        return; // connection failed
    }
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
    if (sock->write(&df_local_machine, 1) == COVISE_SOCKET_INVALID)
    {
        LOGERROR("invalid socket in new ClientConnection");
    }
#ifdef SHOWMSG
    LOGINFO("convert: %d", convert_to);
#endif
}

ClientConnection::~ClientConnection() // close connection
{
    delete lhost;
}

void SimpleClientConnection::get_dataformat()
{
    char dataformat;
    int retries = 60;

    while (retries > 0 && sock->read(&dataformat, 1) < 1)
    {
        retries--;
        if (retries < 50)
            sleep(1);
    }

    if (sock->write(&df_local_machine, 1) != 1)
    {
        LOGERROR("invalid socket in ClientConnection::get_dataformat");
        return;
    }
    /*if(dataformat == DF_IEEE)
   {
      std::cerr << "Remote-Format is IEEE" << std::endl;
   }
   else
   {
      std::cerr << "Remote-Format is not IEEE. Format: "<< dataformat << std::endl;
   }

   if(df_local_machine == DF_IEEE)
   {
      std::cerr << "Local-Format is IEEE" << std::endl;
   }
   else
   {
      std::cerr << "Local-Format is not IEEE. Format: "<< dataformat << std::endl;
   } */
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
}

SimpleClientConnection::SimpleClientConnection(Host *h, int p, int retries)
{

    static int id = 1234;
    buflen = 0;
    buffer[10000] = '\0';
    lhost = NULL;
    if (h) // host is not local
        host = h;
    else // host is local (usually DataManagerConnection uses this)
        host = lhost = new Host("localhost");
    port = p;
    sender_id = id;
    id++;
    send_type = (int)20;
    sock = new Socket(host, port, retries);

    if (get_id() == -1)
        return; // connection not established
}

SimpleClientConnection::~SimpleClientConnection() // close connection
{
    delete lhost;
};

/* reads until it has found a \n
 * and returns this without the \n.
 * if no complete line could be read, NULL is returned
 * make sure non blocking is true!!
 */
const char *SimpleServerConnection::readLine()
{
    if (check_for_input())
    {
        int numBytes = sock->read(buffer + buflen, int(10000 - buflen));
        if (numBytes > 0)
        {
            buflen += numBytes;
        }
        else
        {
            char *line = new char[20];
            strcpy(line, "ConnectionClosed");
            return line;
        }
    }
    if (buflen)
    {
        char *pos = strchr(buffer, '\n');
        if (pos)
        {
            *pos = '\0';
            size_t len = (pos - buffer) + 1;
            char *line = new char[len];
            strcpy(line, buffer);
            buflen -= len;
            memmove(buffer, buffer + len, buflen);
            return line;
        }
        else
        {
            char *c = buffer + buflen - 1;
            int i = 0;
            while (i < buflen)
            {
                if (*c == '\0')
                    break;
                i++;
            }
            if (i < buflen)
            {
                std::cerr << "removing garbaage\n" << std::endl;
                memmove(buffer, c + 1, i);
                buflen = i;
            }
        }
    }
    return NULL;
}

/* reads until it has found a \n
 * and returns this without the \n.
 * if no complete line could be read, NULL is returned
 * make sure non blocking is true!!
 */
const char *SimpleClientConnection::readLine()
{
    if (check_for_input())
    {
        int numBytes = sock->read(buffer + buflen, int(10000 - buflen));
        if (numBytes > 0)
        {
            buflen += numBytes;
        }
        if (numBytes < 0)
        {
            char *line = new char[20];
            strcpy(line, "ConnectionClosed");
            return line;
        }
    }
    if (buflen)
    {
        char *pos = strchr(buffer, '\n');
        if (pos)
        {
            *pos = '\0';
			size_t len = (pos - buffer) + 1;
            char *line = new char[len];
            strcpy(line, buffer);
            buflen -= len;
            memmove(buffer, buffer + len, buflen);
            return line;
        }
        else
        {
            char *c = buffer + buflen - 1;
            int i = 0;
            while (i < buflen)
            {
                if (*c == '\0')
                    break;
                i++;
            }
            if (i < buflen)
            {
                std::cerr << "removing garbaage\n" << std::endl;
                memmove(buffer, c + 1, i);
                buflen = i;
            }
        }
    }
    return NULL;
}

void Connection::close_inform()
{
    // Crashes the OpenSG renderer while embedded into the map editor.
    // Does somebody need this anyway?
    //Message msg(CLOSE_SOCKET, (int)0, NULL, MSG_COPY);
    //send_msg(&msg);                                // will be closed anyway
    if (sock)
    {
        LOGINFO("close_inform port %d", sock->get_port());
        delete sock;
        sock = NULL;
    }
    else
    {
        LOGINFO("trying to close invalid socket");
    }
}

void Connection::close()
{
    if (sock)
    {
        LOGINFO("close port %d", sock->get_port());
        if (remove_socket)
        {
            remove_socket(sock->get_id());
            remove_socket = NULL;
        }
        delete sock;
        sock = NULL;
    }
}

void Connection::set_hostid(int id)
{
    TokenBuffer tb;
    tb << id;
    Message msg(tb);
    msg.type = Message::HOSTID;
    sendMessage(&msg);
    hostid = id;
}

int ServerConnection::acceptOne()
{
    char dataformat;

    if (!sock)
    {
        LOGERROR("no socket in ServerConnection::accept");
        return -1;
    }
    if (sock->acceptOnly() < 0)
    {
        return -1;
    }
    if (sock->write(&df_local_machine, 1) != 1)
    {
        LOGERROR("invalid socket in ServerConnection::accept");
        return -1;
    }

    int retries = 60;
    while (retries > 0 && sock->read(&dataformat, 1) < 1)
    {
        retries--;
        if (retries < 50)
            sleep(1);
    }
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;

#ifdef SHOWMSG
    LOGINFO("convert: %d", convert_to);
#endif

    return 0;
}

int ServerConnection::acceptOne(int wait)
{
    if (sock->acceptOnly(wait) != 0)
        return -1;
    get_dataformat();
    return 0;
}

ServerConnection *Socket::copy_and_accept()
{
    ServerConnection *tmp_conn;
    Socket *tmp_sock;
    tmp_sock = new Socket(*this);

    socklen_t length = sizeof(s_addr_in);

    tmp_sock->sock_id = (int)::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
    if (tmp_sock->sock_id == -1)
    {
        coPerror("ServerConnection: accept failed in copy_and_accept");
        return NULL;
    }
    tmp_conn = new ServerConnection(tmp_sock);
    return tmp_conn;
}

SimpleServerConnection *Socket::copySimpleAndAccept()
{
    SimpleServerConnection *tmp_conn;
    Socket *tmp_sock;
    tmp_sock = new Socket(*this);

    socklen_t length = sizeof(s_addr_in);

    tmp_sock->sock_id = (int)::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
    if (tmp_sock->sock_id == -1)
    {
        coPerror("ServerConnection: accept failed in copySimpleAndAccept");
        return NULL;
    }
    tmp_conn = new SimpleServerConnection(tmp_sock);
    return tmp_conn;
}

ServerConnection *ServerConnection::spawn_connection()
{
    ServerConnection *new_conn;
    new_conn = sock->copy_and_accept();
    new_conn->port = port;
    new_conn->sender_id = sender_id;
    new_conn->send_type = send_type;
    new_conn->get_dataformat();
    return new_conn;
}

SimpleServerConnection *ServerConnection::spawnSimpleConnection()
{
    SimpleServerConnection *new_conn;
    new_conn = sock->copySimpleAndAccept();
    new_conn->port = port;
    new_conn->sender_id = sender_id;
    new_conn->send_type = send_type;
    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    setsockopt(new_conn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

    new_conn->getSocket()->setNonBlocking(true);
    return new_conn;
}

ServerUdpConnection::ServerUdpConnection(UDPSocket* s)
	:ServerConnection((Socket*)s)
{
}
bool ServerUdpConnection::sendMessageTo(Message* msg, const char* address)
{
	UDPSocket* s = (UDPSocket*)sock;
	int retval = 0, tmp_bytes_written;
	char write_buf[WRITE_BUFFER_SIZE];
	int* write_buf_int;

	if (!sock)
		return 0;
#ifdef SHOWMSG
	LOGINFO("send: s: %d st: %d mt: %s l: %d", sender_id, send_type, covise_msg_types_array[msg->type], msg->data.length());
#endif
#ifdef CRAY
	int tmp_buf[4];
	tmp_buf[0] = sender_id;
	tmp_buf[1] = send_type;
	tmp_buf[2] = msg->type;
	tmp_buf[3] = msg->data.length();
#ifdef _CRAYT3E
	converter.int_array_to_exch(tmp_buf, write_buf, 4);
#else
	conv_array_int_c8i4(tmp_buf, (int*)write_buf, 4, START_EVEN);
#endif
#else
	//Compose COVISE header
	write_buf_int = (int*)write_buf;
	write_buf_int[0] = sender_id;
	write_buf_int[1] = send_type;
	write_buf_int[2] = msg->type;
	write_buf_int[3] = msg->data.length();
	swap_bytes((unsigned int*)write_buf_int, 4);
#endif

	if (msg->data.length() == 0)
		retval = s->writeTo(write_buf, 4 * SIZEOF_IEEE_INT, address);
	else
	{
#ifdef SHOWMSG
		LOGINFO("msg->data.length(): %d", msg->data.length());
#endif
		// Decide wether the message including the COVISE header
		// fits into one packet
		if (msg->data.length() < WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT)
		{
			// Write data to socket for data blocks smaller than the
			// socket write-buffer reduced by the size of the COVISE
			// header
			memcpy(&write_buf[4 * SIZEOF_IEEE_INT], msg->data.data(), msg->data.length());
			retval = s->writeTo(write_buf, 4 * SIZEOF_IEEE_INT + msg->data.length(), address);
		}
		else
		{
			// Message and COVISE header summed-up size is bigger than one network
			// packet. Therefore it is transmitted in multiple pakets.

			// Copy data block of size WRITE_BUFFER_SIZE to send buffer
			memcpy(&write_buf[16], msg->data.data(), WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT);
#ifdef SHOWMSG
			LOGINFO("write_buf: %d %d %d %d", write_buf_int[0], write_buf_int[1], write_buf_int[2], write_buf_int[3]);
#endif
			//Send first packet of size WRITE_BUFFER_SIZE
			retval = s->writeTo(write_buf, WRITE_BUFFER_SIZE, address);
#ifdef CRAY
			tmp_bytes_written = (UDPSocket*)sock->writeTo(&msg->data[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
				msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
#else
			// Send next packet with remaining data. Thnis code assumes that an overall
			// message size does never exceed 2x WRITE_BUFFER_SIZE.
			tmp_bytes_written = s->writeTo(&msg->data.data()[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
				msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT), address);
#endif
			// Check if the socket was invalidated while transmitting multiple packets.
			// Otherwise returned the summed-up amount of bytes as return value
			if (tmp_bytes_written == COVISE_SOCKET_INVALID)
				return COVISE_SOCKET_INVALID;
			else
				retval += tmp_bytes_written;
		}
	}
	return retval;
}
void ServerConnection::get_dataformat()
{
    char dataformat;

    if (sock->write(&df_local_machine, 1) != 1)
    {
        LOGERROR("invalid socket in ServerConnection::accept");
        return;
    }
    int retries = 60;
	int ret = 0;
    while (retries > 0 && (ret = sock->read(&dataformat, 1)) < 1)
    {
		if (ret < 0)
		{
			// socket closed, no need to wait
			LOGERROR("socket closed in get_dataformat\n");
			return;
		}
        retries--;
        if (retries < 50)
            sleep(1);
    }
    /*if(dataformat == DF_IEEE)
   {
      std::cerr << "Remote-Format is IEEE" << std::endl;
   }
   else
   {
      std::cerr << "Remote-Format is not IEEE. Format: "<< dataformat << std::endl;
   }

   if(df_local_machine == DF_IEEE)
   {
      std::cerr << "Local-Format is IEEE" << std::endl;
   }
   else
   {
      std::cerr << "Local-Format is not IEEE. Format: "<< dataformat << std::endl;
   }*/
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
}

void Connection::set_peer(int id, int type)
{
    peer_id_ = id;
    peer_type_ = type;
}

int Connection::get_peer_id()
{
    return peer_id_;
}

int Connection::get_peer_type()
{
    return peer_type_;
}

int Connection::receive(void *buf, unsigned nbyte)
{
    int retval;
    if (!sock)
        return 0;
    retval = sock->Read(buf, nbyte);

    return retval;
}

int Connection::send(const void *buf, unsigned nbyte)
{
    if (!sock)
        return 0;
    return sock->write(buf, nbyte);
}

int Connection::send_msg_fast(const Message *msg)
{
    int bytes_written = 0;
    int offset = 0;

    //Compose COVISE header
    header_int[0] = sender_id;
    header_int[1] = send_type;
    header_int[2] = msg->type;
    header_int[3] = msg->data.length();

    int ret = sock->write(header_int, 4 * SIZEOF_IEEE_INT);
    if (ret < 0)
    {
        return ret;
    }

    bytes_written += ret;

    while (offset < msg->data.length())
    {
        if (msg->data.length() - offset < WRITE_BUFFER_SIZE)
        {
            ret = sock->write((void *)&(msg->data.data()[offset]), msg->data.length() - offset);
        }
        else
        {
            ret = sock->write((void *)&(msg->data.data()[offset]), WRITE_BUFFER_SIZE);
        }
        if (ret < 0)
        {
            return ret;
        }
        bytes_written += ret;
        offset += ret;
    }
    return bytes_written;
}

bool Connection::sendMessage(const Message *msg) const
{
    int retval = 0, tmp_bytes_written;
    char write_buf[WRITE_BUFFER_SIZE];
    int *write_buf_int;

    if (!sock)
        return 0;
#ifdef SHOWMSG
    LOGINFO("send: s: %d st: %d mt: %s l: %d", sender_id, send_type, covise_msg_types_array[msg->type], data.length());
#endif
#ifdef CRAY
    int tmp_buf[4];
    tmp_buf[0] = sender_id;
    tmp_buf[1] = send_type;
    tmp_buf[2] = msg->type;
    tmp_buf[3] = msg->data.length();
#ifdef _CRAYT3E
    converter.int_array_to_exch(tmp_buf, write_buf, 4);
#else
    conv_array_int_c8i4(tmp_buf, (int *)write_buf, 4, START_EVEN);
#endif
#else
    //Compose COVISE header
    write_buf_int = (int *)write_buf;
    write_buf_int[0] = sender_id;
    write_buf_int[1] = send_type;
    write_buf_int[2] = msg->type;
    write_buf_int[3] = msg->data.length();
    swap_bytes((unsigned int *)write_buf_int, 4);
#endif

    if (msg->data.length() == 0)
        retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT);
    else
    {
#ifdef SHOWMSG
        LOGINFO("data.length(): %d", msg->data.length());
#endif
        // Decide wether the message including the COVISE header
        // fits into one packet
        if (msg->data.length() < WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT)
        {
            // Write data to socket for data blocks smaller than the
            // socket write-buffer reduced by the size of the COVISE
            // header
            memcpy(&write_buf[4 * SIZEOF_IEEE_INT], msg->data.data(), msg->data.length());
            retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT + msg->data.length());
        }
        else
        {
            // Message and COVISE header summed-up size is bigger than one network
            // packet. Therefore it is transmitted in multiple pakets.

            // Copy data block of size WRITE_BUFFER_SIZE to send buffer
            memcpy(&write_buf[16], msg->data.data(), WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT);
#ifdef SHOWMSG
            LOGINFO("write_buf: %d %d %d %d", write_buf_int[0], write_buf_int[1], write_buf_int[2], write_buf_int[3]);
#endif
            //Send first packet of size WRITE_BUFFER_SIZE
            retval = sock->write(write_buf, WRITE_BUFFER_SIZE);
#ifdef CRAY
            tmp_bytes_written = sock->writea(&msg->data[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
                                             msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
#else
            // Send next packet with remaining data. Thnis code assumes that an overall
            // message size does never exceed 2x WRITE_BUFFER_SIZE.
            tmp_bytes_written = sock->write(&msg->data.data()[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
                                            msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
#endif
            // Check if the socket was invalidated while transmitting multiple packets.
            // Otherwise returned the summed-up amount of bytes as return value
            if (tmp_bytes_written == COVISE_SOCKET_INVALID)
                return COVISE_SOCKET_INVALID;
            else
                retval += tmp_bytes_written;
        }
    }
    return retval;
}

bool Connection::sendMessage(const UdpMessage *msg) const{
    return false;
}


int Connection::recv_msg_fast(Message *msg)
{
    //Init values
    //int send_type = 0;
    //int sender_id = 0;
    int read_bytes = 0;

    //Store size of existing buffer in Message
    int existing_buffer_len = msg->data.length();

    //Read header
    read_bytes = sock->read(header_int, 4 * SIZEOF_IEEE_INT);
    if (read_bytes < 0)
    {
        msg->type = Message::SOCKET_CLOSED;
        return read_bytes;
    }

    //sender_id = header_int[0];
    //send_type = header_int[1];

    msg->type = header_int[2];
    //Extend buffer if necessary, which is costly
    if (existing_buffer_len < header_int[3])
    {
        //Realloc

        msg->data = DataHandle(header_int[3]);
    }

    //Now read data in 64K blocks
    char *buffer = msg->data.accessData();
    int ret = 0;
    int read_msg_bytes = 0;

    while (read_msg_bytes < msg->data.length())
    {
        if (msg->data.length() - read_msg_bytes < READ_BUFFER_SIZE)
        {
            ret = sock->read(buffer, msg->data.length() - read_msg_bytes);
        }
        else
        {
            ret = sock->read(buffer, READ_BUFFER_SIZE);
        }

        if (ret < 0)
        {
            resolveError();
            return ret;
        }
        read_msg_bytes += ret;
        buffer += ret;
    }
    return read_bytes + read_msg_bytes;
}

int Connection::recv_msg(Message *msg, char* ip)
{
    int bytes_read, bytes_to_read, tmp_read;
    char *read_buf_ptr;
    int *int_read_buf;
    int data_length;
    char *read_data;
#ifdef SHOWMSG
    char tmp_str[255];
#endif
#ifdef CRAY
    int tmp_buf[4];
#endif

    msg->sender = 0;
    msg->data.setLength(0);
    msg->send_type = Message::UNDEFINED;
    msg->type = Message::EMPTY;
    msg->conn = this;
    message_to_do = 0;

    if (!sock)
        return 0;

    ///  aw: this looks like stdin/stdout sending
    if (send_type == Message::STDINOUT)
    {
        //	LOGINFO( "in send_type == STDINOUT");
        do
        {
#ifdef _WIN32
            unsigned long tru = 1;
            ioctlsocket(sock->get_id(), FIONBIO, &tru);
#else
            fcntl(sock->get_id(), F_SETFL, O_NDELAY); // this is non-blocking
#endif
            bytes_read = sock->read(read_buf, READ_BUFFER_SIZE);
            if (bytes_read < 0)
            {
                //		      LOGINFO( "bytes read == -1");
                //		      perror("error after STDINOUT read");
                //		      sprintf(retstr, "errno: %d", errno);
                //		      LOGINFO( retstr);
                msg->type = Message::SOCKET_CLOSED;
                return 0;
            }
            else
            {
                read_buf[bytes_read] = '\0';
                std::cout << read_buf;
                std::cout.flush();
                sprintf(&read_buf[bytes_read], "bytes read: %d", bytes_read);
                LOGINFO(read_buf);
            }
        } while (bytes_read > 0);
        msg->type = Message::STDINOUT_EMPTY;
        return 0;
    }

    ////// this is sending to anything else than stdin/out

    /*  reading all in one call avoids the expensive read system call */

    while (bytes_to_process < 16)
    {
        tmp_read = sock->Read(read_buf + bytes_to_process, READ_BUFFER_SIZE - bytes_to_process, ip);
        if (tmp_read == 0)
            return (0);
        if (tmp_read < 0)
        {
            msg->type = Message::SOCKET_CLOSED;
            msg->conn = this;
            return tmp_read;
        }
        bytes_to_process += tmp_read;
    }

    read_buf_ptr = read_buf;

    while (1)
    {
        int_read_buf = (int *)read_buf_ptr;

#ifdef SHOWMSG
        LOGINFO("read_buf: %d %d %d %d", int_read_buf[0], int_read_buf[1], int_read_buf[2], int_read_buf[3]);
#endif

#ifdef CRAY
#ifdef _CRAYT3E
        converter.exch_to_int_array(read_buf_ptr, tmp_buf, 4);
//	conv_array_int_i4t8(int_read_buf, tmp_buf, 4, START_EVEN);
#else
        conv_array_int_i4c8(int_read_buf, tmp_buf, 4, START_EVEN);
#endif
        msg->sender = tmp_buf[0];
        msg->send_type = int(tmp_buf[1]);
        msg->type = covise_msg_type(tmp_buf[2]);
        msg->data.setLength(tmp_buf[3]);
#else
#ifdef BYTESWAP
        swap_bytes((unsigned int *)int_read_buf, 4);
#endif
        msg->sender = int_read_buf[0];
        msg->send_type = int(int_read_buf[1]);
        msg->type = int_read_buf[2];
        msg->data.setLength(int_read_buf[3]);
//	sprintf(retstr, "msg header: sender %d, sender_type %d, covise_msg_type %d, length %d",
//	        msg->sender, msg->send_type, msg->type, msg->data.length());
//	        LOGINFO( retstr);
#endif

#ifdef SHOWMSG
        sprintf(tmp_str, "recv: s: %d st: %d mt: %s l: %d",
                msg->sender, msg->send_type,
                (msg->type < 0 || msg->type > COVISE_MESSAGE_LAST_DUMMY_MESSAGE) ? (msg->type == -1 ? "EMPTY" : "(invalid)") : covise_msg_types_array[msg->type],
                msg->data.length());
        LOGINFO(tmp_str);
#endif

        bytes_to_process -= 4 * SIZEOF_IEEE_INT;
#ifdef SHOWMSG
        LOGINFO("bytes_to_process %d bytes, msg->data.length() %d", bytes_to_process, msg->data.length());
#endif
        read_buf_ptr += 4 * SIZEOF_IEEE_INT;
        if (msg->data.length() > 0) // if msg->data.length() == 0, no data will be received
        {
            // bring message data space to 16 byte alignment
            data_length = msg->data.length() + ((msg->data.length() % 16 != 0) * (16 - msg->data.length() % 16));
            msg->data = DataHandle(new char[data_length], msg->data.length());
            if (msg->data.length() > bytes_to_process)
            {
                bytes_read = bytes_to_process;
#ifdef SHOWMSG
                LOGINFO("bytes_to_process %d bytes, msg->data.length() %d", bytes_to_process, msg->data.length());
#endif
                if (bytes_read != 0)
                    memcpy(msg->data.accessData(), read_buf_ptr, bytes_read);
                bytes_to_process = 0;
                bytes_to_read = msg->data.length() - bytes_read;
                read_data = &msg->data.accessData()[bytes_read];
                while (bytes_read < msg->data.length())
                {

#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes", bytes_to_process);
                    LOGINFO("bytes_to_read    %d bytes", bytes_to_read);
                    LOGINFO("bytes_read       %d bytes", bytes_read);
#endif
                    tmp_read = sock->Read(read_data, bytes_to_read);
                    if (tmp_read < 0)
                    {
                        msg->data = DataHandle();
                        return 0;
                    }
                    else
                    {
                        // covise_time->mark(__LINE__, "nach weiterem sock->read(read_buf)");
                        bytes_read += tmp_read;
                        bytes_to_read -= tmp_read;
                        read_data = &msg->data.accessData()[bytes_read];
                    }
                }
#ifdef SHOWMSG
                LOGINFO("message_to_do = 0");
#endif
                //	        covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->data.length();
            }
            else if (msg->data.length() < bytes_to_process)
            {
                memcpy(msg->data.accessData(), read_buf_ptr, msg->data.length());
                bytes_to_process -= msg->data.length();
                read_buf_ptr += msg->data.length();
#ifdef SHOWMSG
                LOGINFO("bytes_to_process %d bytes, msg->data.length() %d", bytes_to_process, msg->data.length());
#endif
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;
                while (bytes_to_process < 16)
                {
#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes, msg->data.length() %d", bytes_to_process, msg->data.length());
#endif
                    tmp_read = sock->Read(&read_buf_ptr[bytes_to_process],
                                          READ_BUFFER_SIZE - bytes_to_process);
                    if (tmp_read < 0)
                    {
                        msg->data = DataHandle();
                        return 0;
                    }
                    bytes_to_process += tmp_read;
                }
                message_to_do = 1;
#ifdef SHOWMSG
                LOGINFO("message_to_do = 1");
#endif
                //	        covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->data.length();
            }
            else
            {
                memcpy(msg->data.accessData(), read_buf_ptr, bytes_to_process);
                bytes_to_process = 0;
#ifdef SHOWMSG
                LOGINFO("message_to_do = 0");
#endif
                //	            covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->data.length();
            }
        }

        else //msg->data.length() == 0, no data will be received
        {
            if (msg->data.data())
            {
                msg->data = DataHandle(nullptr, msg->data.length());
            }
            if (msg->data.length() < bytes_to_process)
            {
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;
                while (bytes_to_process < 16)
                {
#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes, msg->data.length() %d", bytes_to_process, msg->data.length());
#endif
                    tmp_read = sock->Read(&read_buf_ptr[bytes_to_process],
                                          READ_BUFFER_SIZE - bytes_to_process);
                    if (tmp_read < 0)
                    {
                        msg->data = DataHandle(nullptr, msg->data.length());
                        return 0;
                    }
                    bytes_to_process += tmp_read;
                }
                message_to_do = 1;
#ifdef SHOWMSG
                LOGINFO("message_to_do = 1");
#endif
            }

            return 0;
        }
    }
}

int Connection::check_for_input(float time)
{
    if (has_message())
        return 1;

    int i = 0;
    do
    {
        struct timeval timeout;
        timeout.tv_sec = (int)time;
        timeout.tv_usec = (int)((time - timeout.tv_sec) * 1000000);

        // initialize the bit fields according to the existing sockets
        // so far only reads are of interest

        fd_set fdread;
        FD_ZERO(&fdread);
        FD_SET(sock->get_id(), &fdread);

        i = select(sock->get_id() + 1, &fdread, NULL, NULL, &timeout);
#ifdef WIN32
    } while (i == -1 && (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAEINPROGRESS));
#else
    } while (i == -1 && errno == EINTR);
#endif

    // find the connection that has the read attempt

    if (i > 0)
        return (1);
    else if (i < 0)
    {
        LOGINFO("select failed: %s\n", Socket::coStrerror(Socket::getErrno()));
        coPerror("select failed");
    }
    return 0;
}

ConnectionList::ConnectionList()
{
    open_sock = 0;
    maxfd = 0;
    FD_ZERO(&fdvar);
}

ConnectionList::ConnectionList(ServerConnection *o_s)
{
    FD_ZERO(&fdvar); // the field for the select call is initiallized
    open_sock = o_s;
    if (open_sock->listen() < 0)
    {
        fprintf(stderr, "ConnectionList: listen failure\n");
    }
    int id = open_sock->get_id();
    maxfd = id;
    FD_SET(id, &fdvar);
    return;
}

int ConnectionList::count()
{
    return connlist.size();
}
/*
Connection ConnectionList::at(int index)
{
	if (connlist != NULL)
	{
		Connection connItem = *(connlist->at(index));
		return connItem;
	}
	return NULL;
}*/

ConnectionList::~ConnectionList()
{
    for (auto ptr: connlist)
    {
        ptr->close_inform();
        delete ptr;
    }
    connlist.clear();
    return;
}

void ConnectionList::deleteConnection(Connection *c)
{
    delete c;
}

Connection *ConnectionList::get_last() // get connection made recently
{
    if (connlist.empty())
        return nullptr;
    return connlist.back();
}

void ConnectionList::add_open_conn(ServerConnection *c)
{ // add a connection and update the
    // field for the select call
    if (open_sock)
        delete open_sock;
    open_sock = c;
    if (open_sock->listen() < 0)
    {
        fprintf(stderr, "ConnectionList: listen failure\n");
    }
    if (c->get_id() > maxfd)
        maxfd = c->get_id();
    FD_SET(c->get_id(), &fdvar);
    return;
}

void ConnectionList::add(Connection *c) // add a connection and update the
{ //c->print();
    connlist.push_back(c); // field for the select call
    if (c->get_id() > maxfd)
        maxfd = c->get_id();
    FD_SET(c->get_id(), &fdvar);
    return;
}

void ConnectionList::remove(Connection *c) // remove a connection and update
{
    if (!c)
        return;
    auto it = std::find(connlist.begin(), connlist.end(), c);
    if (it != connlist.end())
        connlist.erase(it);
    //FIXME curidx
    if (curidx >= connlist.size())
        curidx = connlist.size();
    // the field for the select call
    FD_CLR(c->get_id(), &fdvar);
}

// aw 04/2000: Check whether PPID==1 or no sockets left: prevent hanging
static void checkPPIDandFD(fd_set &actSet, int maxfd)
{
#ifndef _WIN32
    if (getppid() == 1)
    {
        std::cerr << "Process " << getpid()
                  << " quit because parent died" << std::endl;
        exit(1);
    }
#endif

    int numFD = 0;
    for (int j = 0; j <= maxfd; j++)
        if (FD_ISSET(j, &actSet))
            numFD++;

    if (numFD == 0)
    {
        std::cerr << "Process " << getpid()
                  << " quit because all connections lost" << std::endl;
        exit(1);
    }
}

/// Wait for input infinitely - replaced by loop with timeouts
/// - check every 10 sec against hang aw 04/2000
Connection *ConnectionList::wait_for_input()
{
    Connection *found;
    do
        found = check_for_input(10.0);
    while (!found);
    return found;
}

Connection *ConnectionList::check_for_input(float time)
{
    int numconn = 0;
    // if we already have a pending message, we return it
    for (auto ptr: connlist)
    {
        ++numconn;
        if (ptr->has_message())
            return ptr;
    }

    fd_set fdread;
    int i;
    do
    {
        // initialize timeout field
        struct timeval timeout;
        timeout.tv_sec = (int)time;
        timeout.tv_usec = (int)((time - timeout.tv_sec) * 1000000);

        // initialize the bit fields according to the existing sockets
        // so far only reads are of interest
        FD_ZERO(&fdread);
        for (int j = 0; j <= maxfd; j++)
            if (FD_ISSET(j, &fdvar))
                FD_SET(j, &fdread);

        // wait for the next read attempt on one of the sockets

        i = select(maxfd + 1, &fdread, NULL, NULL, &timeout);
#ifdef WIN32
    } while (i == -1 && (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAEINPROGRESS));
#else
    } while (i == -1 && errno == EINTR);
#endif

    //	LOGINFO( "something happened");

    // nothing? this might be a hanger ... better check it!
    if (i <= 0 && numconn > 0)
        checkPPIDandFD(fdvar, maxfd);

    // find the connection that has the read attempt
    if (i > 0)
    {
        if (open_sock && FD_ISSET(open_sock->get_id(), &fdread))
        {
            Connection *ptr = open_sock->spawn_connection();
            this->add(ptr);
            FD_ZERO(&fdread);
            for (int j = 0; j <= maxfd; j++)
                if (FD_ISSET(j, &fdvar))
                    FD_SET(j, &fdread);
            i--;
        }
        else
        {
            for (auto ptr: connlist)
            {
                if (FD_ISSET(ptr->get_id(), &fdread))
                {
                    return ptr;
                }
            }
        }
    }
    else if (i < 0)
    {
        LOGINFO("select failed: %s\n", Socket::coStrerror(Socket::getErrno()));
        coPerror("select failed");
    }
    return NULL;
}

void ConnectionList::reset() //
{
    curidx = 0;
}

Connection *ConnectionList::next() //
{
    ++curidx;
    if (curidx < connlist.size())
        return connlist[curidx];
    return nullptr;
}

#ifdef HAVE_OPENSSL
SSLConnection::SSLConnection()
    : Connection()
{
    mBuflen = 0;
    mBuffer[10000] = '\0';

    mState = SSLConnection::SSL_READY;
}
SSLConnection::SSLConnection(int sfd)
    : Connection(sfd)

{
    mBuflen = 0;
    mBuffer[10000] = '\0';

    mState = SSLConnection::SSL_READY;
}

SSLConnection::~SSLConnection()
{
    mState = SSLConnection::SSL_CLOSED;
}

int SSLConnection::AttachSSLToSocket(Socket *pSock)
{
    return SSL_set_fd(mSSL, pSock->get_id());
}

bool SSLConnection::IsClosed() const
{
    if (mState == SSL_CLOSED)
    {
        return true;
    }
    return false;
}

int SSLConnection::receive(void *buf, unsigned nbyte)
{
    if (IsClosed())
    {
        return -1;
    }

    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
    if (!locSocket)
    {
        return -1;
    }
    int rbytes = locSocket->read(buf, nbyte);
    if (rbytes == 0)
    {
        validateConnState();
        return -1;
    }
    return rbytes;
}

int SSLConnection::send(const void *buf, unsigned nbyte)
{
    if (IsClosed())
    {
        return -1;
    }
    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
    if (!locSocket)
    {
        return -1;
    }

    int rbytes = locSocket->write(buf, nbyte);
    if (rbytes == 0)
    {
        validateConnState();
        return -1;
    }
    return rbytes;
}

void SSLConnection::validateConnState() const
{
    if (SSL_get_shutdown(mSSL) == SSL_RECEIVED_SHUTDOWN)
    {
        this->mState = SSL_CLOSED;
        SSL_shutdown(this->mSSL);
    }
    else
    {
        // ungraceful shutdown
        // first renegotiate and if it fails
        // shutdown
        if (SSL_renegotiate(mSSL) == 0)
        {
            //shutdown
            this->mState = SSL_CLOSED;
            SSL_shutdown(mSSL);
        }
    }
}

std::string SSLConnection::getPeerAddress()
{
    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
    if (!locSocket)
    {
        fprintf(stderr, "SSLConnection::getPeerAddress(): Not a valid SSLSocket!\n");
        return "0.0.0.0";
    }
    return locSocket->getPeerAddress();
}

int SSLConnection::recv_msg(Message *msg)
{
    if (IsClosed())
    {
        return -1;
    }

    int bytes_read = 0;

    //Init block
    msg->sender = 0;
    msg->data.setLength(0);
    msg->send_type = Message::UNDEFINED;
    msg->type = Message::EMPTY;
    msg->conn = this;
    message_to_do = 0;

    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);

    if (send_type == Message::STDINOUT)
    {
        do
        {
#ifdef _WIN32
            unsigned long tru = 1;
            ioctlsocket(sock->get_id(), FIONBIO, &tru);
#else
            fcntl(sock->get_id(), F_SETFL, O_NDELAY);
#endif

            bytes_read = locSocket->read(read_buf, READ_BUFFER_SIZE);
            if (bytes_read < 0)
            {
                msg->type = Message::SOCKET_CLOSED;
                return -1;
            }
            if (bytes_read == 0)
            {
                validateConnState();
                return 0;
            }
            else
            {
                read_buf[bytes_read] = '\0';
                sprintf(&read_buf[bytes_read], "bytes read: %d", bytes_read);
            }

        } while (bytes_read > 0);
    }
    else if (send_type == Message::UNDEFINED)
    {
        bytes_read = locSocket->read(read_buf, READ_BUFFER_SIZE);
        if (bytes_read < 0)
        {
            msg->type = Message::SOCKET_CLOSED;
            return -1;
        }
        if (bytes_read == 0)
        {
            validateConnState();
            return 0;
        }
    }

    return 0;
}

/**
 * @Desc: SSL compatible send_msg function. Doesn't support Cray.
 */
int SSLConnection::send_msg(const Message *msg) const
{
    if (IsClosed())
    {
        return -1;
    }

    int retval = 0;
    int tmp_bytes_written = 0;
    int *write_buf_int = NULL;
    char write_buf[WRITE_BUFFER_SIZE];

    write_buf_int = (int *)write_buf;
    write_buf_int[0] = sender_id;
    write_buf_int[1] = send_type;
    write_buf_int[2] = msg->type;
    write_buf_int[3] = msg->data.length();
    swap_bytes((unsigned int *)write_buf_int, 4);

    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);

    if (msg->data.length() == 0)
    {
        //Zero length message, just write message header data to socket
        retval = locSocket->write(write_buf, 4 * SIZEOF_IEEE_INT);
    }
    else
    {
        //Data message write header and data to socket
        if (msg->data.length() < WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT)
        {
            // Handle case where data buffer fits in the given data buffer
            // of the socket.
            memcpy(&write_buf[4 * SIZEOF_IEEE_INT], msg->data.data(), msg->data.length());
            retval = locSocket->write(write_buf, 4 * SIZEOF_IEEE_INT + msg->data.length());
        }
        else
        {
            // Handle case where data buffer doesn't fit
            // in the given data buffer of the socket.
            // The buffer has to be split up in several write attempts
            memcpy(&write_buf[4 * SIZEOF_IEEE_INT], msg->data.data(), msg->data.length());
            retval = locSocket->write(write_buf, WRITE_BUFFER_SIZE);
            tmp_bytes_written = locSocket->write(&msg->data.data()[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
                                                 msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
            if (tmp_bytes_written == COVISE_SOCKET_INVALID)
            {
                // Error or nothing left to write anymore
                return COVISE_SOCKET_INVALID;
            }
            else if (tmp_bytes_written == 0)
            {
                validateConnState();
                return 0;
            }
            else
            {
                // Successfully written data to socket
                retval += tmp_bytes_written;
            }
        }
    }
    return retval;
}

bool SSLConnection::sendMessage(const Message *msg) const{
    return send_msg(msg) > 0;
}

int SSLConnection::get_id(void (*remove_func)(int))
{
    remove_socket = remove_func;
    //	fprintf(stderr, "Socket: %x\n", sock);
    return sock->get_id();
}

int SSLConnection::get_id() const
{
    //cerr << "sock == " << sock << " id: " << sock->get_id() << endl;
    if (sock)
        return sock->get_id();
    return -1;
}

const char *SSLConnection::readLine()
{
    int numBytes = 0;

    if (check_for_input())
    {
        SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
        if (locSocket)
        {
            numBytes = locSocket->read(mBuffer + mBuflen, int(10000 - mBuflen));
        }
        else
        {
            return "Read failed!";
        }
        if (numBytes > 0)
        {
            mBuflen += numBytes;
        }
        if (numBytes < 0)
        {
            char *line = new char[20];
            strcpy(line, "ConnectionClosed");
            return line;
        }
        if (numBytes == 0)
        {
            validateConnState();
            return "SSLConnectionClosed";
        }
    }
    if (mBuflen)
    {
        char *pos = strchr(mBuffer, '\n');
        if (pos)
        {
            *pos = '\0';
            size_t len = (pos - mBuffer) + 1;
            char *line = new char[len];
            strcpy(line, mBuffer);
            mBuflen -= len;
            memmove(mBuffer, mBuffer + len, mBuflen);
            return line;
        }
        else
        {
            char *c = mBuffer + mBuflen - 1;
            int i = 0;
            while (i < mBuflen)
            {
                if (*c == '\0')
                    break;
                i++;
            }
            if (i < mBuflen)
            {
                std::cerr << "removing garbaage\n" << std::endl;
                memmove(mBuffer, c + 1, i);
                mBuflen = i;
            }
        }
    }
    return NULL;
}

void SSLConnection::setPasswdCallback(SSLConnection::PasswordCallback *cb, void *userData)
{
    mSSLCB = cb;
    mSSLUserData = userData;

    if ((cb != NULL) && (userData != NULL))
    {
        SSL_CTX_set_default_passwd_cb_userdata(mCTX, userData);
        SSL_CTX_set_default_passwd_cb(mCTX, cb);
    }
}

//SSLServerConnection
SSLServerConnection::SSLServerConnection(SSLConnection::PasswordCallback *cb, void *userData)
    : SSLConnection()
{
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    CONST_SSL_METHOD_P meth = SSLv23_server_method();
    //SSL_METHOD* meth = SSLv2_server_method();
    //SSL_METHOD* meth = TLSv1_server_method();
    if (!mCTX)
    {
        mCTX = SSL_CTX_new(meth);
    }
    if (!mCTX)
    {
        exit(2);
    }

    SSL_CTX_set_mode(mCTX, SSL_MODE_AUTO_RETRY);
    setPasswdCallback(cb, userData);

    std::string certfile = coCoviseConfig::getEntry("System.CoviseDaemon.Certificate");
    std::string keyfile = coCoviseConfig::getEntry("System.CoviseDaemon.Keyfile");
    std::string cafile = coCoviseConfig::getEntry("System.CoviseDaemon.CAFile");

    int result = 0;
    result = SSL_CTX_load_verify_locations(mCTX, cafile.c_str(), NULL);
    if (result <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): verify_locations - Exitcode (5)\n");
        exit(5);
    }

    if (coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false))
    {

        SSL_CTX_set_client_CA_list(mCTX, SSL_load_client_CA_file(cafile.c_str()));
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    }
    else
    {
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_NONE, NULL);
    }

    if (SSL_CTX_use_certificate_file(mCTX, certfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): Exitcode (3)\n");
        exit(3);
    }
    if (SSL_CTX_use_PrivateKey_file(mCTX, keyfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): Exitcode (4)\n");
        exit(4);
    }

    if (!SSL_CTX_check_private_key(mCTX))
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): Private key does not match the certificate public key\n");
        exit(5);
    }

    //Setup for SSL based on socket allocated in Connection(int sfd)
    mSSL = SSL_new(mCTX);
    if (!mSSL)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): Couldn't create SSL object!\n");
        return;
    }
    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
    locSocket->setSSL(mSSL);
}

SSLServerConnection::SSLServerConnection(int *p, int id, int s_type, PasswordCallback *cb, void *userData)
    : SSLConnection()
{
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    CONST_SSL_METHOD_P meth = SSLv23_server_method();
    //SSL_METHOD* meth = SSLv2_server_method();
    //SSL_METHOD* meth = TLSv1_server_method();

    if (!mCTX)
    {
        mCTX = SSL_CTX_new(meth);
    }
    if (!mCTX)
    {
        exit(2);
    }

    SSL_CTX_set_mode(mCTX, SSL_MODE_AUTO_RETRY);
    setPasswdCallback(cb, userData);

    std::string certfile = coCoviseConfig::getEntry("System.CoviseDaemon.Certificate");
    std::string keyfile = coCoviseConfig::getEntry("System.CoviseDaemon.Keyfile");
    std::string cafile = coCoviseConfig::getEntry("System.CoviseDaemon.CAFile");

    int result = 0;
    result = SSL_CTX_load_verify_locations(mCTX, cafile.c_str(), NULL);
    if (result <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): verify_locations - Exitcode (5)\n");
        exit(5);
    }

    if (coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false))
    {

        SSL_CTX_set_client_CA_list(mCTX, SSL_load_client_CA_file(cafile.c_str()));
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    }
    else
    {
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_NONE, NULL);
    }

    if (SSL_CTX_use_certificate_file(mCTX, certfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int *p, int id, sender_type s_type): Exitcode (3)\n");
        exit(3);
    }
    if (SSL_CTX_use_PrivateKey_file(mCTX, keyfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int *p, int id, sender_type s_type): Exitcode (4)\n");
        exit(4);
    }

    if (!SSL_CTX_check_private_key(mCTX))
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int *p, int id, sender_type s_type): Private key does not match the certificate public key\n");
        exit(5);
    }

    //Setup for SSL based on socket allocated in Connection(int sfd)
    mSSL = SSL_new(mCTX);
    if (!mSSL)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int *p, int id, sender_type s_type): Couldn't create SSL object!\n");
        return;
    }

    sender_id = id;
    send_type = s_type;
    sock = new SSLSocket(p, this->mSSL);
    if (sock->get_id() == -1)
    {
        delete sock;
        sock = NULL;
        *p = 0;
    }
    port = *p;
}

SSLServerConnection::SSLServerConnection(int p, int id, int st, PasswordCallback *cb, void *userData)
    : SSLConnection()
{
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    CONST_SSL_METHOD_P meth = SSLv23_server_method();
    //SSL_METHOD* meth = SSLv2_server_method();
    //SSL_METHOD* meth = TLSv1_server_method();
    if (!mCTX)
    {
        mCTX = SSL_CTX_new(meth);
    }
    if (!mCTX)
    {
        exit(2);
    }

    SSL_CTX_set_mode(mCTX, SSL_MODE_AUTO_RETRY);
    setPasswdCallback(cb, userData);

    std::string certfile = coCoviseConfig::getEntry("System.CoviseDaemon.Certificate");
    std::string keyfile = coCoviseConfig::getEntry("System.CoviseDaemon.Keyfile");
    std::string cafile = coCoviseConfig::getEntry("System.CoviseDaemon.CAFile");
    int result = 0;
    result = SSL_CTX_load_verify_locations(mCTX, cafile.c_str(), NULL);
    if (result <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): verify_locations - Exitcode (5)\n");
        exit(5);
    }

    if (coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false))
    {

        SSL_CTX_set_client_CA_list(mCTX, SSL_load_client_CA_file(cafile.c_str()));
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    }
    else
    {
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_NONE, NULL);
    }

    if (SSL_CTX_use_certificate_file(mCTX, certfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int p, int id, sender_type st): Exitcode (3)\n");
        exit(3);
    }
    if (SSL_CTX_use_PrivateKey_file(mCTX, keyfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int p, int id, sender_type st): Exitcode (4)\n");
        exit(4);
    }

    if (!SSL_CTX_check_private_key(mCTX))
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int p, int id, sender_type st): Private key does not match the certificate public key\n");
        exit(5);
    }

    //Setup for SSL based on socket allocated in Connection(int sfd)
    mSSL = SSL_new(mCTX);
    if (!mSSL)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(int p, int id, sender_type st): Couldn't create SSL object!\n");
        return;
    }

    sender_id = id;
    send_type = st;
    port = p;
    this->sock = new SSLSocket(p, this->mSSL); //Creates a new socket and binds
    if (sock->get_id() == -1)
    {
        delete sock;
        sock = NULL;
        port = 0;
    }
}

SSLServerConnection::SSLServerConnection(SSLSocket *socket, PasswordCallback *cb, void *userData)
{
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    CONST_SSL_METHOD_P meth = SSLv23_server_method();
    //meth = SSLv2_server_method();
    //meth = TLSv1_server_method();
    if (meth == NULL)
    {
        fprintf(stderr, "SSLServerConnection(Socket*): Method creation failed!\n");
        return;
    }
    if (!mCTX)
    {
        mCTX = SSL_CTX_new(meth);
    }
    if (!mCTX)
    {
        fprintf(stderr, "SSLServerConnection(Socket*): Context creation failed!\n");
        return;
    }

    SSL_CTX_set_mode(mCTX, SSL_MODE_AUTO_RETRY);
    setPasswdCallback(cb, userData);

    std::string certfile = coCoviseConfig::getEntry("System.CoviseDaemon.Certificate");
    std::string keyfile = coCoviseConfig::getEntry("System.CoviseDaemon.Keyfile");
    std::string cafile = coCoviseConfig::getEntry("System.CoviseDaemon.CAFile");

    int result = 0;
    result = SSL_CTX_load_verify_locations(mCTX, cafile.c_str(), NULL);
    if (result <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(): verify_locations - Exitcode (5)\n");
        exit(5);
    }

    if (coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false))
    {

        SSL_CTX_set_client_CA_list(mCTX, SSL_load_client_CA_file(cafile.c_str()));
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    }
    else
    {
        SSL_CTX_set_verify(mCTX, SSL_VERIFY_NONE, NULL);
    }

    if (SSL_CTX_use_certificate_file(mCTX, certfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Exitcode (3)\n");
        exit(3);
    }
    if (SSL_CTX_use_PrivateKey_file(mCTX, keyfile.c_str(), SSL_FILETYPE_PEM) <= 0)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Exitcode (4)\n");
        exit(4);
    }

    if (!SSL_CTX_check_private_key(mCTX))
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Private key does not match the certificate public key\n");
        exit(5);
    }

    //Setup for SSL based on socket allocated in Connection(int sfd)
    mSSL = SSL_new(mCTX);
    if (!mSSL)
    {
        fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Couldn't create SSL object!\n");
        return;
    }

    sock = socket;
    SSLSocket *locSocket = dynamic_cast<SSLSocket *>(sock);
    locSocket->setSSL(mSSL);
}

SSLServerConnection::~SSLServerConnection()
{
    cerr << "SSLServerConnection::destroy()" << endl;
    cerr << "SSLServerConnection::destroy(): Uninit SSL" << endl;
    SSL_shutdown(mSSL);
    SSL_free(mSSL);
    SSL_CTX_free(mCTX);
    mSSL = NULL;
    mCTX = NULL;
    cerr << "SSLServerConnection::destroy() finished" << endl;
}

int SSLServerConnection::accept()
{
    cerr << "SSLServerConnection::accept(): SSL-Object = " << mSSL << endl;
    int result = SSL_accept(this->mSSL);
    if (result == 0)
    {
        SSL_get_error(mSSL, result);
    }

    return result;
}

int SSLServerConnection::sock_accept()
{
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);

    int tmpSock = (int)::accept(this->sock->get_id(), (sockaddr *)&client_addr, &client_addr_len);

#ifdef _WIN32
    ::closesocket(this->sock->get_id());
#else
    ::close(this->sock->get_id());
#endif

    delete sock;
    sock = new SSLSocket((int)tmpSock, &client_addr, this->mSSL);

    return sock->get_id();
}

int SSLServerConnection::listen()
{
    return sock->listen();
}

SSLServerConnection *SSLServerConnection::spawnConnection()
{
    // perform winsock accept --> obbey dataformat sync
    // attach ssl to new socket
    // ssl accept to new socket
    // create new connection with new socket
    SSLSocket *locSock = dynamic_cast<SSLSocket *>(sock);
    if (!locSock)
    {
        return NULL;
    }

    return locSock->spawnConnection(mSSLCB, mSSLUserData);
}

std::string SSLServerConnection::getSSLSubjectUID()
{
    X509 *client_cert;
    if (SSL_get_verify_result(mSSL) == X509_V_OK)
    {
        client_cert = SSL_get_peer_certificate(this->mSSL);
        cerr << "SSLServerConnection::getSSLSubjectUID() dump" << endl;
        if (client_cert == NULL)
        {
            return std::string("No cert found!");
        }
    }
    else
    {
        return std::string("Not valid!");
    }
    ASN1_INTEGER *uid = X509_get_serialNumber(client_cert);
    X509_free(client_cert);
    char *cuid = reinterpret_cast<char *>(uid->data);
    return std::string(cuid);
}

std::string SSLServerConnection::getSSLSubjectName()
{
    X509 *client_cert = NULL;

    cerr << "SSLServerConnection::getSSLSubjectName(): Cipher: " << SSL_get_cipher(mSSL) << endl;
    cerr << "SSLServerConnection::getSSLSubjectName(): SSL-Object = " << mSSL << endl;

    if (SSL_get_verify_result(mSSL) == X509_V_OK)
    {

        client_cert = SSL_get_peer_certificate(mSSL);
        cerr << "SSLServerConnection::getSSLSubjectName(): Error retrieving cert!" << endl;
        if (client_cert == NULL)
        {
            return std::string("No cert found!");
        }
    }
    else
    {
        return std::string("Not valid!");
    }

    char *cname = X509_NAME_oneline(X509_get_subject_name(client_cert), 0, 0);

    X509_free(client_cert);
    return std::string(cname);
}

//SSLClientConnection
SSLClientConnection::SSLClientConnection(Host *h, int p, SSLClientConnection::PasswordCallback *cb, void *userData /*,int retries*/)
{
    //SSL init
    SSL_load_error_strings();
    SSLeay_add_ssl_algorithms();
    //mMeth = SSLv2_client_method();
    CONST_SSL_METHOD_P meth = SSLv23_server_method();
    //mMeth = TLSv1_method();

    if (!mCTX)
    {
        this->mCTX = SSL_CTX_new(meth); // Create context
    }

    SSL_CTX_set_mode(mCTX, SSL_MODE_AUTO_RETRY);
    setPasswdCallback(cb, userData);

    //Always verify server
    SSL_CTX_set_verify(mCTX, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);

    std::string certfile = coCoviseConfig::getEntry("System.CoviseDaemon.Certificate");
    std::string keyfile = coCoviseConfig::getEntry("System.CoviseDaemon.Keyfile");
    std::string CAfile = coCoviseConfig::getEntry("System.CoviseDaemon.CAFile");

    if (CAfile != "")
    {
        fprintf(stderr, "SSLClientConnection::SSLClientConnection(Host*,int,int): Loading CA certificate...\n");
        if (SSL_CTX_load_verify_locations(mCTX, CAfile.c_str(), NULL) <= 0)
        {
            fprintf(stderr, "SSLClientConnection::SSLClientConnection(Host*,int,int): CA certificate loading failed\n");
            exit(3);
        }
        fprintf(stderr, "SSLClientConnection::SSLClientConnection(Host*,int,int): ...CA certificate loaded\n");
    }

    if (coCoviseConfig::isOn("System.CoviseDaemon.EnableCertificateCheck", false))
    {

        if (SSL_CTX_use_certificate_chain_file(mCTX, certfile.c_str()) <= 0)
        {
            fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Exitcode (3)\n");
            exit(3);
        }
        if (SSL_CTX_use_PrivateKey_file(mCTX, keyfile.c_str(), SSL_FILETYPE_PEM) <= 0)
        {
            fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Exitcode (4)\n");
            exit(4);
        }

        if (!SSL_CTX_check_private_key(mCTX))
        {
            fprintf(stderr, "SSLServerConnection::SSLServerConnection(SSLSocket*): Private key does not match the certificate public key\n");
            exit(5);
        }
    }

    // Make sure we have basic initialisation for Winsock
    SSLSocket::initialize();

    //create local socket and bind to network
    port = p;
    this->sock = new SSLSocket(); //Creates a new socket and binds
    if (sock->get_id() == -1)
    {
        delete sock;
        sock = NULL;
        port = 0;
    }

    // Connect socket to given server
    SSLSocket *locSock = dynamic_cast<SSLSocket *>(sock);
    this->mSA_server.sin_family = AF_INET;
    this->mSA_server.sin_port = htons(port);
	//this->mSA_server.sin_addr.s_addr = inet_addr(h->getAddress());
	int err = inet_pton(AF_INET, h->getAddress(), &this->mSA_server.sin_addr.s_addr);
	if (err != 1)
	{
#ifdef WIN32
		resolveError();
#endif
	}

    int result = locSock->connect(this->mSA_server /*, retries, 30.0*/);
    if (result != 0)
    {
//Error on connect
#ifdef WIN32
        resolveError();
#endif
    }
    this->host = locSock->get_host();

    //Adapt accordingly

    //Setup SSL connection
    mSSL = SSL_new(mCTX);
    locSock->setSSL(mSSL);
}

void SSLClientConnection::evaluate()
{
    X509 *peer = SSL_get_peer_certificate(mSSL);
    if (!peer)
    {
        cerr << "SSLClientConnection::SSLClientConnection(): No peer certificate!" << endl;
    }

    if (SSL_get_verify_result(mSSL) == X509_V_OK)
    {
        cerr << "SSLClientConnection::SSLClientConnection(): Peer certificate verify!" << endl;
    }
}

SSLClientConnection::~SSLClientConnection()
{
    SSL_shutdown(mSSL);
    SSL_free(mSSL);
    SSL_CTX_free(mCTX);
    mSSL = NULL;
    mCTX = NULL;
}

int SSLClientConnection::connect()
{
    int err = SSL_connect(this->mSSL);
    if (err <= 0)
    {
        long result = SSL_get_verify_result(mSSL);
        getErrorMessage(result);
        checkSSLError(mSSL, err);
    }
    return err;
}

std::string SSLClientConnection::getSSLSubject()
{
    X509 *client_cert;

    cerr << "SSLServerConnection::getSSLSubjectName(): Cipher: " << SSL_get_cipher(mSSL) << endl;

    if (SSL_get_verify_result(mSSL) == X509_V_OK)
    {
        client_cert = SSL_get_peer_certificate(this->mSSL);
        cerr << "SSLServerConnection::getSSLSubjectUID() dump" << endl;
        if (client_cert == NULL)
        {
            return std::string("No cert found!");
        }
    }
    else
    {
        return std::string("Not valid!");
    }
    char *str = X509_NAME_oneline(X509_get_subject_name(client_cert), 0, 0);
    X509_free(client_cert);
    return std::string(str);
}

void SSLClientConnection::getErrorMessage(long result)
{
    std::string sError = "";

    cerr << "Error code is: " << result << "!" << endl;

    switch (result)
    {
    case X509_V_OK:
    {
        sError = "";
    }
    case X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT:
    {
        sError = "";
    }
    case X509_V_ERR_UNABLE_TO_GET_CRL:
    {
        sError = "";
    }
    case X509_V_ERR_UNABLE_TO_DECRYPT_CERT_SIGNATURE:
    {
        sError = "";
    }
    case X509_V_ERR_UNABLE_TO_DECODE_ISSUER_PUBLIC_KEY:
    {
        sError = "";
    }
    case X509_V_ERR_CERT_SIGNATURE_FAILURE:
    {
        sError = "";
    }
    case X509_V_ERR_CRL_SIGNATURE_FAILURE:
    {
        sError = "";
    }
    }
}
#endif
