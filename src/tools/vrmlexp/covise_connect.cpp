/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTypes.h"

#include "covise_connect.h"

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
#undef SHOWMSG
#undef DEBUG

SimpleServerConnection::SimpleServerConnection(Socket *s)
    : ServerConnection(s)
{
    buffer[10000] = '\0';
    buflen = 0;
}

SimpleServerConnection::SimpleServerConnection(int p, int id, sender_type s_type)
    : ServerConnection(p, id, s_type)
{
    buffer[10000] = '\0';
    buflen = 0;
}

SimpleServerConnection::SimpleServerConnection(int *p, int id, sender_type s_type)
    : ServerConnection(p, id, s_type)
{
    buffer[10000] = '\0';
    buflen = 0;
}

ServerConnection::ServerConnection(int p, int id, sender_type s_type)
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

ServerConnection::ServerConnection(int *p, int id, sender_type s_type)
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

UDPConnection::UDPConnection(int id, sender_type s_type, int p,
                             char *address)
    : Connection()
{
    sender_id = id;
    send_type = s_type;
    sock = (Socket *)(new UDPSocket(address, p));
    port = p;
}

#ifdef MULTICAST
MulticastConnection::MulticastConnection(int id, sender_type s_type, int p,
                                         char *MulticastGroup, int ttl)
    : Connection()
{
    sender_id = id;
    send_type = s_type;
    sock = (Socket *)(new MulticastSocket(MulticastGroup, p, ttl));
    port = p;
}
#endif

ClientConnection::ClientConnection(Host *h, int p, int id, sender_type
                                                               s_type,
                                   int retries, double timeout)
{
    char dataformat;
#ifdef DEBUG
    char tmpstr[255];
#endif

    lhost = NULL;
    if (h) // host is not local
        host = h;
    else // host is local (usually DataManagerConnection uses this)
        host = lhost = new Host("localhost");
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
        printf("invalid socket in new ClientConnection\n");

#ifdef DEBUG
    sprintf(tmpstr, "convert: %d", convert_to);
    print_comment(__LINE__, __FILE__, tmpstr);
#endif
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
    send_type = (sender_type)20;
    sock = new Socket(host, port, retries);

    if (get_id() == -1)
        return; // connection not established
}

/* reads until it has found a \n
 * and returns this without the \n.
 * if no complete line could be read, NULL is returned
 * make sure non blocking is true!!
 */
const char *SimpleServerConnection::readLine()
{
    if (check_for_input())
    {
        int numBytes = sock->read(buffer + buflen, 10000 - buflen);
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
            int len = (int)(pos - buffer) + 1;
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
                cerr << "removing garbaage\n" << endl;
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
        int numBytes = sock->read(buffer + buflen, 10000 - buflen);
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
            int len = (int)(pos - buffer) + 1;
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
                cerr << "removing garbaage\n" << endl;
                memmove(buffer, c + 1, i);
                buflen = i;
            }
        }
    }
    return NULL;
}

void Connection::close_inform()
{
    char tmpstr[255];

    // Crashes the OpenSG renderer while embedded into the map editor.
    // Does somebody need this anyway?
    //Message msg(CLOSE_SOCKET, (int)0, NULL, MSG_COPY);
    //send_msg(&msg);                                // will be closed anyway
    sprintf(tmpstr, "close_inform port %d", sock->get_port());
    //print_comment(__LINE__, __FILE__, tmpstr);
    if (sock)
    {
        delete sock;
        sock = NULL;
    }
}

void Connection::close()
{
    char tmpstr[255];

    if (sock)
    {
        sprintf(tmpstr, "close port %d", sock->get_port());
        //print_comment(__LINE__, __FILE__, tmpstr);
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
    msg.type = COVISE_MESSAGE_HOSTID;
    send_msg(&msg);
    hostid = id;
    //tb.delete_data();
}

int ServerConnection::accept()
{
    char dataformat;
#ifdef SHOWMSG
    char tmpstr[255];
#endif

    if (sock->accept() < 0)
    {
        return -1;
    }
    if (sock->write(&df_local_machine, 1) != 1)
    {
        //print_error(__LINE__, __FILE__,
        //   "invalid socket in ServerConnection::accept");
        return -1;
    }

    while (sock->read(&dataformat, 1) < 1)
        ;
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;

#ifdef SHOWMSG
    sprintf(tmpstr, "convert: %d", convert_to);
    print_comment(__LINE__, __FILE__, tmpstr);
#endif

    return 0;
}

int ServerConnection::accept(int wait)
{
    if (sock->accept(wait) != 0)
        return -1;
    get_dataformat();
    return 0;
}

ServerConnection *Socket::copy_and_accept()
{
    ServerConnection *tmp_conn;
    Socket *tmp_sock;
    tmp_sock = new Socket(*this);

    int length = sizeof(s_addr_in);

    tmp_sock->sock_id = (int)::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
    if (tmp_sock->sock_id == -1)
    {
        fprintf(stderr, "ServerConnection: accept failed in copy_and_accept: %s\n", strerror(errno));
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

    int length = sizeof(s_addr_in);

    tmp_sock->sock_id = (int)::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
    if (tmp_sock->sock_id == -1)
    {
        fprintf(stderr, "ServerConnection: accept failed in copySimpleAndAccept: %s\n", strerror(errno));
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

void ServerConnection::get_dataformat()
{
    char dataformat;

    if (sock->write(&df_local_machine, 1) != 1)
    {
        // print_error(__LINE__, __FILE__,
        //    "invalid socket in ServerConnection::accept");
        return;
    }
    while (sock->read(&dataformat, 1) < 1)
        ;
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
}

void Connection::set_peer(int id, sender_type type)
{
    peer_id_ = id;
    peer_type_ = type;
}

int Connection::get_peer_id()
{
    return peer_id_;
}

sender_type Connection::get_peer_type()
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

int Connection::send_msg_fast(const Message* msg)
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
			ret = sock->write((void*) & (msg->data.data()[offset]), msg->data.length() - offset);
		}
		else
		{
			ret = sock->write((void*) & (msg->data.data()[offset]), WRITE_BUFFER_SIZE);
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

int Connection::send_msg(const Message* msg)
{
	int retval = 0, tmp_bytes_written;
	char write_buf[WRITE_BUFFER_SIZE];
	int* write_buf_int;

	if (!sock)
		return 0;
#ifdef SHOWMSG
	LOGINFO("send: s: %d st: %d mt: %s l: %d", sender_id, send_type, covise_msg_types_array[msg->type], data.length());
#endif
	//Compose COVISE header
	write_buf_int = (int*)write_buf;
	write_buf_int[0] = sender_id;
	write_buf_int[1] = send_type;
	write_buf_int[2] = msg->type;
	write_buf_int[3] = msg->data.length();
	swap_bytes((unsigned int*)write_buf_int, 4);

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
			// Send next packet with remaining data. Thnis code assumes that an overall
			// message size does never exceed 2x WRITE_BUFFER_SIZE.
			tmp_bytes_written = sock->write(&msg->data.data()[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
				msg->data.length() - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
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

int Connection::recv_msg_fast(Message* msg)
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
	char* buffer = msg->data.accessData();
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
			{
				std::string sError = "";

#ifdef WIN32
				int error = WSAGetLastError();

				cerr << "resolveError(): Error-Code: " << error << endl;

				switch (error)
				{
				case WSANOTINITIALISED:
				{
					sError = "A successful WSAStartup call must occur before using this function.";
				}
				break;
				case WSAENETDOWN:
				{
					sError = "The network subsystem has failed.";
				}
				break;
				case WSAEADDRINUSE:
				{
					sError = "The socket's local address is already in use and the socket was not marked to allow address reuse with SO_REUSEADDR. This error usually occurs when executing bind, but could be delayed until this function if the bind was to a partially wildcard address (involving ADDR_ANY) and if a specific address needs to be committed at the time of this function.";
				}
				break;
				case WSAEINTR:
				{
					sError = "The blocking Windows Socket 1.1 call was canceled through WSACancelBlockingCall.";
				}
				break;
				case WSAEINPROGRESS:
				{
					sError = "A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function.";
				}
				break;
				case WSAEALREADY:
				{
					sError = "A nonblocking connect call is in progress on the specified socket. \nNote:\n  In order to preserve backward compatibility, this error is reported as WSAEINVAL to Windows Sockets 1.1 applications that link to either Winsock.dll or Wsock32.dll.";
				}
				break;
				case WSAEADDRNOTAVAIL:
				{
					sError = "The remote address is not a valid address (such as ADDR_ANY).";
				}
				break;
				case WSAEAFNOSUPPORT:
				{
					sError = "Addresses in the specified family cannot be used with this socket.";
				}
				break;
				case WSAECONNREFUSED:
				{
					sError = "The attempt to connect was forcefully rejected.";
				}
				break;
				case WSAEFAULT:
				{
					sError = "The name or the namelen parameter is not a valid part of the user address space, the namelen parameter is too small, or the name parameter contains incorrect address format for the associated address family.";
				}
				break;
				case WSAEINVAL:
				{
					sError = "The parameter s is a listening socket.";
				}
				break;
				case WSAEISCONN:
				{
					sError = "The socket is already connected (connection-oriented sockets only).";
				}
				break;
				case WSAENETUNREACH:
				{
					sError = "The network cannot be reached from this host at this time.";
				}
				break;
				case WSAEHOSTUNREACH:
				{
					sError = "A socket operation was attempted to an unreachable host.";
				}
				break;
				case WSAENOBUFS:
				{
					sError = "No buffer space is available. The socket cannot be connected.";
				}
				break;
				case WSAENOTSOCK:
				{
					sError = "The descriptor is not a socket.";
				}
				break;
				case WSAETIMEDOUT:
				{
					sError = "Attempt to connect timed out without establishing a connection.";
				}
				break;
				case WSAEWOULDBLOCK:
				{
					sError = "The socket is marked as nonblocking and the connection cannot be completed immediately.";
				}
				break;
				case WSAEACCES:
				{
					sError = "Attempt to connect datagram socket to broadcast address failed because setsockopt option SO_BROADCAST is not enabled.";
				}
				break;
				case WSAECONNRESET:
				{
					sError = "An existing connection was forcibly closed by the remote host. This normally results if the peer application on the remote host is suddenly stopped, the host is rebooted, the host or remote network interface is disabled, or the remote host uses a hard close (see setsockopt for more information on the SO_LINGER option on the remote socket). This error may also result if a connection was broken due to keep-alive activity detecting a failure while one or more operations are in progress. Operations that were in progress fail with WSAENETRESET. Subsequent operations fail with WSAECONNRESET.";
				}
				case WSAENOTCONN:
				{
					sError = "A request to send or receive data was disallowed because the socket is not connected and (when sending on a datagram socket using sendto) no address was supplied. Any other type of operation might also return this error?for example, setsockopt setting SO_KEEPALIVE if the connection has been reset.";
				}
				default:
				{
					sError = "Error unkown!";
				}
				break;
				}
#endif
				cerr << sError.c_str() << endl;
			}
			return ret;
		}
		read_msg_bytes += ret;
		buffer += ret;
	}
	return read_bytes + read_msg_bytes;
}

int Connection::recv_msg(Message* msg, char* ip)
{
	int bytes_read, bytes_to_read, tmp_read;
	char* read_buf_ptr;
	int* int_read_buf;
	int data_length;
	char* read_data;
#ifdef SHOWMSG
	char tmp_str[255];
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
				//LOGINFO(read_buf);
			}
		} while (bytes_read > 0);
		msg->type = Message::STDINOUT_EMPTY;
		return 0;
			}

	////// this is sending to anything else than stdin/out

	/*  reading all in one call avoids the expensive read system call */

	while (bytes_to_process < 16)
	{
		tmp_read = sock->Read(read_buf + bytes_to_process, READ_BUFFER_SIZE - bytes_to_process);
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
		int_read_buf = (int*)read_buf_ptr;

#ifdef SHOWMSG
		LOGINFO("read_buf: %d %d %d %d", int_read_buf[0], int_read_buf[1], int_read_buf[2], int_read_buf[3]);
#endif

#ifdef BYTESWAP
		swap_bytes((unsigned int*)int_read_buf, 4);
#endif
		msg->sender = int_read_buf[0];
		msg->send_type = int(int_read_buf[1]);
		msg->type = int_read_buf[2];
		msg->data.setLength(int_read_buf[3]);
		//	sprintf(retstr, "msg header: sender %d, sender_type %d, covise_msg_type %d, length %d",
		//	        msg->sender, msg->send_type, msg->type, msg->data.length());
		//	        LOGINFO( retstr);

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
    } while (i == -1 && errno == EINTR);

    // find the connection that has the read attempt

    if (i > 0)
        return (1);
    else if (i < 0)
    {
        char buf[1024];
        sprintf(buf, "select failed: %s\n", strerror(errno));
        // print_comment(__LINE__, __FILE__, buf);
    }
    return 0;
}

ConnectionList::ConnectionList()
{
    connlist = new List<Connection>;
    open_sock = NULL;
    maxfd = 0;
    FD_ZERO(&fdvar); // the field for the select call is initiallized
    return;
}

ConnectionList::ConnectionList(ServerConnection *o_s)
{
    connlist = new List<Connection>;
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

ConnectionList::~ConnectionList()
{
    Connection *ptr;
    connlist->reset();
    while ((ptr = connlist->next()))
    {
        ptr->close_inform();
        delete ptr;
    }
    delete connlist;
    return;
}

void ConnectionList::deleteConnection(Connection *c)
{
    delete c;
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
    connlist->add(c); // field for the select call
    if (c->get_id() > maxfd)
        maxfd = c->get_id();
    FD_SET(c->get_id(), &fdvar);
    return;
}

void ConnectionList::remove(Connection *c) // remove a connection and update
{
    connlist->remove(c); // the field for the select call
    FD_CLR(c->get_id(), &fdvar);
    return;
}

// aw 04/2000: Check whether PPID==1 or no sockets left: prevent hanging
static void checkPPIDandFD(fd_set &actSet, int maxfd)
{
#ifndef _WIN32
    if (getppid() == 1)
    {
        cerr << "Process " << getpid()
             << " quit because parent died" << endl;
        exit(0);
    }
#endif

    int j, numFD = 0;
    for (j = 0; j <= maxfd; j++)
        if (FD_ISSET(j, &actSet))
            numFD++;

    if (numFD == 0)
    {
        cerr << "Process " << _getpid()
             << " quit because all connections lost" << endl;
        //exit(0);
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
    fd_set fdread;
    int i, j;
    Connection *ptr;
    struct timeval timeout;

    // if we already have a pending message, we return it
    connlist->reset();
    while ((ptr = connlist->next()))
    {
        if (ptr->has_message())
            return ptr;
    }

    do
    {
        // initialize timeout field
        timeout.tv_sec = (int)time;
        timeout.tv_usec = (int)((time - timeout.tv_sec) * 1000000);

        // initialize the bit fields according to the existing sockets
        // so far only reads are of interest
        FD_ZERO(&fdread);
        for (j = 0; j <= maxfd; j++)
            if (FD_ISSET(j, &fdvar))
                FD_SET(j, &fdread);

        // wait for the next read attempt on one of the sockets

        i = select(maxfd + 1, &fdread, NULL, NULL, &timeout);
    } while (i == -1 && errno == EINTR);

    //	print_comment(__LINE__, __FILE__, "something happened");

    // nothing? this might be a hanger ... better check it!
    if (i <= 0)
        checkPPIDandFD(fdvar, maxfd);

    // find the connection that has the read attempt
    if (i > 0)
    {
        if (open_sock && FD_ISSET(open_sock->get_id(), &fdread))
        {
            ptr = open_sock->spawn_connection();
            this->add(ptr);
            FD_ZERO(&fdread);
            for (j = 0; j <= maxfd; j++)
                if (FD_ISSET(j, &fdvar))
                    FD_SET(j, &fdread);
            i--;
        }
        else
        {
            connlist->reset();
            while ((ptr = connlist->next()))
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
        char buf[1024];
        sprintf(buf, "select failed: %s\n", strerror(errno));
        //print_comment(__LINE__, __FILE__, buf);
    }
    return NULL;
}
