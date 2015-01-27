/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32_WCE
#include <winsock2.h>
#include <WS2tcpip.h>
#else
#include <sys/socket.h>
#include <stdlib.h>
#endif
//#include <util/unixcompat.h>

#include "wce_connect.h"
#include "wce_host.h"
#include "wce_socket.h"
#include <iostream>

using namespace std;

using namespace covise;
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

int Connection::get_id()
{
    //cerr << "sock == " << sock << " id: " << sock->get_id() << endl;
    if (sock)
        return sock->get_id();
    return -1;
}; // give socket id

Connection::Connection()
{
    convert_to = DF_NONE;
    message_to_do = 0;
    //		   LOGINFO( "message_to_do == 0");
    read_buf = new char[READ_BUFFER_SIZE];
    bytes_to_process = 0;
    remove_socket = 0L;
    hostid = -1;
    peer_id_ = 0;
    peer_type_ = UNDEFINED; // prepare connection (for subclasses)
};

Connection::Connection(int sfd)
{
    send_type = STDINOUT;
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
    fcntl(sfd, F_SETFL, O_NDELAY); // this is non-blocking, since
#endif
    // we do not know, what will arrive
    peer_id_ = 0;
    peer_type_ = UNDEFINED; // initializae connection with existing socket
};

Connection::~Connection() // close connection (for subclasses)
{
    delete[] read_buf;
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
    return sock->get_host();
}

const char *Connection::get_hostname()
{
    return sock->get_hostname();
}

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

int ServerConnection::listen() // listen for connection (after bind)
{
    return sock->listen();
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
            int len = (pos - buffer) + 1;
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
            int len = (pos - buffer) + 1;
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
    // Crashes the ViNCE renderer while embedded into the map editor.
    // Does somebody need this anyway?
    //Message msg(CLOSE_SOCKET, (int)0, NULL, MSG_COPY);
    //send_msg(&msg);                                // will be closed anyway
    if (sock)
    {
        //LOGINFO("close_inform port %d", sock->get_port());
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
        //LOGINFO("close port %d", sock->get_port());
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
    covise::TokenBuffer tb;
    tb << id;
    Message msg(tb);
    msg.type = COVISE_MESSAGE_HOSTID;
    send_msg(&msg);
    hostid = id;
    tb.delete_data();
}

int ServerConnection::accept()
{
    char dataformat;

    if (sock->accept() < 0)
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
        //if(retries < 50)
        //sleep(1);
    }
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;

#ifdef SHOWMSG
    LOGINFO("convert: %d", convert_to);
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

    socklen_t length = sizeof(s_addr_in);

    tmp_sock->sock_id = ::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
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

    tmp_sock->sock_id = ::accept(sock_id, (sockaddr *)((void *)&s_addr_in), &length);
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

void ServerConnection::get_dataformat()
{
    char dataformat;

    if (sock->write(&df_local_machine, 1) != 1)
    {
        LOGERROR("invalid socket in ServerConnection::accept");
        return;
    }
    int retries = 60;
    while (retries > 0 && sock->read(&dataformat, 1) < 1)
    {
        retries--;
        //if(retries < 50)
        //sleep(1);
    }
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

int Connection::send_msg(const Message *msg)
{
    int retval = 0, tmp_bytes_written;
    char write_buf[WRITE_BUFFER_SIZE];
    int *write_buf_int;

    if (!sock)
        return 0;
#ifdef SHOWMSG
    LOGINFO("send: s: %d st: %d mt: %s l: %d", sender_id, send_type, covise_msg_types_array[msg->type], msg->length);
#endif
#ifdef CRAY
    int tmp_buf[4];
    tmp_buf[0] = sender_id;
    tmp_buf[1] = send_type;
    tmp_buf[2] = msg->type;
    tmp_buf[3] = msg->length;
#ifdef _CRAYT3E
    converter.int_array_to_exch(tmp_buf, write_buf, 4);
#else
    conv_array_int_c8i4(tmp_buf, (int *)write_buf, 4, START_EVEN);
#endif
#else
    write_buf_int = (int *)write_buf;
    write_buf_int[0] = sender_id;
    write_buf_int[1] = send_type;
    write_buf_int[2] = msg->type;
    write_buf_int[3] = msg->length;
    swap_bytes((unsigned int *)write_buf_int, 4);
#endif

    if (msg->length == 0)
        retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT);
    else
    {
#ifdef SHOWMSG
        LOGINFO("msg->length: %d", msg->length);
#endif
        if (msg->length < WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT)
        {
            memcpy(&write_buf[4 * SIZEOF_IEEE_INT], msg->data, msg->length);
            retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT + msg->length);
        }
        else
        {
            memcpy(&write_buf[16], msg->data, WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT);
#ifdef SHOWMSG
            LOGINFO("write_buf: %d %d %d %d", write_buf_int[0], write_buf_int[1], write_buf_int[2], write_buf_int[3]);
#endif
            retval = sock->write(write_buf, WRITE_BUFFER_SIZE);
#ifdef CRAY
            tmp_bytes_written = sock->writea(&msg->data[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
                                             msg->length - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
#else
            tmp_bytes_written = sock->write(&msg->data[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT],
                                            msg->length - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
#endif
            if (tmp_bytes_written == COVISE_SOCKET_INVALID)
                return COVISE_SOCKET_INVALID;
            else
                retval += tmp_bytes_written;
        }
    }
    return retval;
}

int Connection::recv_msg(Message *msg)
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

    msg->sender = msg->length = 0;
    msg->send_type = UNDEFINED;
    msg->type = COVISE_MESSAGE_EMPTY;
    msg->conn = this;
    message_to_do = 0;

    if (!sock)
        return 0;

    ///  aw: this looks like stdin/stdout sending
    if (send_type == STDINOUT)
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
                msg->type = COVISE_MESSAGE_SOCKET_CLOSED;
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
        msg->type = COVISE_MESSAGE_STDINOUT_EMPTY;
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
            msg->type = COVISE_MESSAGE_SOCKET_CLOSED;
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
        msg->send_type = sender_type(tmp_buf[1]);
        msg->type = covise_msg_type(tmp_buf[2]);
        msg->length = tmp_buf[3];
#else
#ifdef BYTESWAP
        swap_bytes((unsigned int *)int_read_buf, 4);
#endif
        msg->sender = int_read_buf[0];
        msg->send_type = sender_type(int_read_buf[1]);
        msg->type = covise_msg_type(int_read_buf[2]);
        msg->length = int_read_buf[3];
//	sprintf(retstr, "msg header: sender %d, sender_type %d, covise_msg_type %d, length %d",
//	        msg->sender, msg->send_type, msg->type, msg->length);
//	        LOGINFO( retstr);
#endif

#ifdef SHOWMSG
        sprintf(tmp_str, "recv: s: %d st: %d mt: %s l: %d",
                msg->sender, msg->send_type,
                (msg->type < 0 || msg->type > COVISE_MESSAGE_LAST_DUMMY_MESSAGE) ? (msg->type == -1 ? "EMPTY" : "(invalid)") : covise_msg_types_array[msg->type],
                msg->length);
        LOGINFO(tmp_str);
#endif

        bytes_to_process -= 4 * SIZEOF_IEEE_INT;
#ifdef SHOWMSG
        LOGINFO("bytes_to_process %d bytes, msg->length %d", bytes_to_process, msg->length);
#endif
        read_buf_ptr += 4 * SIZEOF_IEEE_INT;
        if (msg->length > 0) // if msg->length == 0, no data will be received
        {
            if (msg->data)
            {
                delete[] msg -> data;
                msg->data = NULL;
            }
            // bring message data space to 16 byte alignment
            data_length = msg->length + ((msg->length % 16 != 0) * (16 - msg->length % 16));
            msg->data = new char[data_length];
            if (msg->length > bytes_to_process)
            {
                bytes_read = bytes_to_process;
#ifdef SHOWMSG
                LOGINFO("bytes_to_process %d bytes, msg->length %d", bytes_to_process, msg->length);
#endif
                if (bytes_read != 0)
                    memcpy(msg->data, read_buf_ptr, bytes_read);
                bytes_to_process = 0;
                bytes_to_read = msg->length - bytes_read;
                read_data = &msg->data[bytes_read];
                while (bytes_read < msg->length)
                {

#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes", bytes_to_process);
                    LOGINFO("bytes_to_read    %d bytes", bytes_to_read);
                    LOGINFO("bytes_read       %d bytes", bytes_read);
#endif
                    tmp_read = sock->read(read_data, bytes_to_read);
                    if (tmp_read < 0)
                    {
                        delete[] msg -> data;
                        msg->data = NULL;
                        return 0;
                    }
                    else
                    {
                        // covise_time->mark(__LINE__, "nach weiterem sock->read(read_buf)");
                        bytes_read += tmp_read;
                        bytes_to_read -= tmp_read;
                        read_data = &msg->data[bytes_read];
                    }
                }
#ifdef SHOWMSG
                LOGINFO("message_to_do = 0");
#endif
                //	        covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->length;
            }
            else if (msg->length < bytes_to_process)
            {
                memcpy(msg->data, read_buf_ptr, msg->length);
                bytes_to_process -= msg->length;
                read_buf_ptr += msg->length;
#ifdef SHOWMSG
                LOGINFO("bytes_to_process %d bytes, msg->length %d", bytes_to_process, msg->length);
#endif
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;
                while (bytes_to_process < 16)
                {
#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes, msg->length %d", bytes_to_process, msg->length);
#endif
                    bytes_to_process += sock->read(&read_buf_ptr[bytes_to_process],
                                                   READ_BUFFER_SIZE - bytes_to_process);
                }
                message_to_do = 1;
#ifdef SHOWMSG
                LOGINFO("message_to_do = 1");
#endif
                //	        covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->length;
            }
            else
            {
                memcpy(msg->data, read_buf_ptr, bytes_to_process);
                bytes_to_process = 0;
#ifdef SHOWMSG
                LOGINFO("message_to_do = 0");
#endif
                //	            covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg->length;
            }
        }

        else //msg->length == 0, no data will be received
        {
            if (msg->data)
            {
                delete[] msg -> data;
                msg->data = NULL;
            }
            if (msg->length < bytes_to_process)
            {
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;
                while (bytes_to_process < 16)
                {
#ifdef SHOWMSG
                    LOGINFO("bytes_to_process %d bytes, msg->length %d", bytes_to_process, msg->length);
#endif
                    bytes_to_process += sock->read(&read_buf_ptr[bytes_to_process],
                                                   READ_BUFFER_SIZE - bytes_to_process);
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
        //LOGINFO("select failed: %s\n", Socket::coStrerror(Socket::getErrno()));
        coPerror("select failed");
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

int ConnectionList::count()
{
    return connlist->count();
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
        std::cerr << "Process " << getpid()
                  << " quit because parent died" << std::endl;
        exit(0);
    }
#endif

    int j, numFD = 0;
    for (j = 0; j <= maxfd; j++)
        if (FD_ISSET(j, &actSet))
            numFD++;

    if (numFD == 0)
    {
        //std::cerr << "Process " << getpid()
        //   << " quit because all connections lost" << std::endl;
        exit(0);
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
    connlist->reset();
    while (Connection *ptr = connlist->next())
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
            connlist->reset();
            while (Connection *ptr = connlist->next())
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
        //LOGINFO("select failed: %s\n", Socket::coStrerror(Socket::getErrno()));
        coPerror("select failed");
    }
    return NULL;
}
