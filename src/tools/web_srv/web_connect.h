/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_CONNECTION_H
#define EC_CONNECTION_H

#define SERVER_SOFTWARE "web_cov/0.1b 19iun01"
#define SERVER_ADDRESS "http://www.visenso.de"

#include "web_socket.h"
#include "web_msg.h"
#include <covise/covise_list.h>
#include <covise/covise_global.h>
#include <covise/covise_conv.h>
#include <sysdep/net.h>
#include <iostream>
#ifdef _WIN32
#include <fcntl.h>
#else
#include <unistd.h>
#endif
#if defined(__sgi) || defined(__hpux) || defined(_SX)
#include <fcntl.h>
#endif
// inserted to work with linux! awi
#ifndef _POSIX_SOURCE
#define _POSIX_SOURCE
#endif
#include <fcntl.h>
#undef _POSIX_SOURCE
#include <sys/types.h>
#if defined(_AIX)
#include <sys/select.h>
#endif
#ifdef _SX
#include <sys/select.h>
#endif

const int EC_SERVER = 0;
const int EC_CLIENT = 1;

#define WRITE_BUFFER_SIZE 64000
#define READ_BUFFER_SIZE WRITE_BUFFER_SIZE

/***********************************************************************\ 
 **                                                                     **
 **   Connection  classes                          Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : These classes present the user-seeable part of the **
 **                  socket communications (if necessary).              **
 **                  Connection is the base class, ServerConecction     **
 **                  and ClientConnection are subclasses tuned for the  **
 **                  server and the client part of a socket.            **
 **                  ConnectionList provides the data structures        **
 **                  necessary to use the select UNIX system call       **
 **                  that allows to listen to many connections at once  **
 **                                                                     **
 **   Classes      : Connection, ServerConnection, ClientConnection,    **
 **                  ConnectionList                                     **
 **                                                                     **
 **   Copyright (C)                **
 **                                        **
 **                                        **
 **                                        **
 **                                                                     **
 **                                                                     **
 **   Author       :                                   **
 **                                                                     **
 **   History      :                                                    **
 **                                                   **
 **                         **
 **                                        **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

enum conn_type
{
    CON_GENERIC = 0, //  0
    CON_COVISE, //  1
    CON_HTTP //  2
};

class Connection
{
protected:
    friend class HServerConnection;
    friend class CServerConnection;
    friend class ConnectionList;

    conn_type m_conn_type;
    char *m_connid;
    int m_dynamic_usr;
    int m_dynamic_view;

    int m_status;

    class Socket *sock; // Socket for connection
    int port; // port for connection
    int sender_id; // id of the sending process

    int message_to_do; // if more than one message has been read
    int bytes_to_process;
    unsigned long tru;
    char *read_buf;
    Host *other_host;
    int hostid; //hostid of remote host
    void (*remove_socket)(int);

public:
    char convert_to; // to what format do we need to convert data?
    Connection()
    {
        convert_to = DF_NONE;
        message_to_do = 0;
        m_conn_type = CON_GENERIC;
        //		   print_comment(__LINE__, __FILE__, "message_to_do == 0");
        read_buf = new char[READ_BUFFER_SIZE];
        bytes_to_process = 0;
        remove_socket = 0L;
        m_status = -1;
        m_connid = NULL;
        m_dynamic_usr = 0;
        m_dynamic_view = 0;
        hostid = -1; // prepare connection (for subclasses)
    };

    virtual ~Connection() // close connection (for subclasses)
    {
        delete[] read_buf;
        delete sock;
    };
    conn_type get_type(void)
    {
        return m_conn_type;
    };
    int get_id()
    {
        //cerr << "sock == " << sock << " id: " << sock->get_id() << endl;
        if (sock == NULL)
            return -1;
        return sock->get_id();
    }; // give socket id
    void set_connid(char *sid);
    char *get_connid(void)
    {
        return m_connid;
    };

    void set_dynamic_usr(int i)
    {
        m_dynamic_usr = i;
    };
    int is_dynamic_usr(void)
    {
        return m_dynamic_usr;
    };

    void set_dynamic_view(int i)
    {
        m_dynamic_view = i;
    };
    int is_dynamic_view(void)
    {
        return m_dynamic_view;
    };

    int get_status()
    {
        return m_status;
    };
    void listen() // listen for connection (after
    {
        sock->listen();
    };

    virtual Connection *spawn_connection(void)
    {
        return NULL;
    }

    int is_connected() // return true if connection is established
    {
        return (get_id() != -1);
    };
    int receive(void *buf, unsigned nbyte); // receive from socket
    int send(const void *buf, unsigned nbyte); // send into socket
    virtual Message *recv_msg(void) // receive Message
    {
        return NULL;
    };
    virtual int send_msg(Message *msg) // send Message
    {
        (void)msg;
        return 0;
    };

    int check_for_input(float time = 0.0); // issue select call and return TRUE if there is an event or 0L otherwise
    int get_port() // give port number
    {
        return port;
    };
    void set_hostid(int id);
    int get_hostid()
    {
        return hostid;
    };
    int get_id(void (*remove_func)(int))
    {
        remove_socket = remove_func;
        //	fprintf(stderr, "Socket: %x\n", sock);
        return sock->get_id();
    }; // give socket id
    int get_sender_id()
    {
        return sender_id;
    };
    void close(); // send close msg for partner and delete socket
    void close_inform(); // close without msg for partner
    int has_message()
    {
        //	if(message_to_do)
        //	    print_comment(__LINE__, __FILE__, "message_to_do == 1");
        //	else
        //	    print_comment(__LINE__, __FILE__, "message_to_do == 0");
        return message_to_do; // message is already read
    };
    void print()
    {
        cerr << "web port: " << port << endl;
    };
    Host *get_host()
    {
        return sock->get_host();
    };
    char *get_hostname()
    {
        return sock->get_hostname();
    };
    virtual void sendError(msg_type mt, char *txt)
    {
        (void)mt;
        (void)txt;
    }
};

class HConnection : public Connection
{
    char *m_read_buff;
    long m_buffer_size;
    long m_bytes_read;

public:
    HConnection()
    {
        m_conn_type = CON_HTTP;
        m_status = 1;
        m_read_buff = NULL;
        m_buffer_size = 0;
        m_bytes_read = 0;
    };

    ~HConnection()
    {
        if (m_read_buff)
            delete[] m_read_buff;
    };

    HMessage *recv_msg(void); // receive HMessage
    int send_msg(Message *msg); // send HMessage //NEI const
    int send_file(FILE *fd);
    int send_file(char *name);
    int send_http_error(msg_type code, char *arg, int get_flag = 1);
    void sendError(msg_type mt, char *txt);
};

class CConnection : public Connection
{
    int m_view_ref;
    int m_covise_port;
    int m_http_port;

public:
    CConnection()
    {
        m_conn_type = CON_COVISE;
        m_status = 1;
        m_view_ref = 0;
        m_covise_port = 0;
        m_http_port = 0;
    };
    CMessage *recv_msg(void); // receive CMessage
    int send_msg(Message *msg); // send HMessage //NEI const

    void inc_ref(void);
    void dec_ref(void);
};

class HServerConnection : public HConnection
{
    // Connection that acts as server to HTTP requests
public:
    // bind socket to port p
    HServerConnection(int p, int id);
    // bind socket, port assigned automatically
    HServerConnection(int *p, int id);
    HServerConnection(Socket *s)
    {
        sock = s;
        m_status = 1;
    };
    ~HServerConnection() // close connection
        {};

    int accept(int); // accept connection (after bind) and wait int seconds
    HServerConnection *spawn_connection(); // accept after select for open socket
};

class CServerConnection : public CConnection
{
    // Connection that acts as server to Covise requests
    void get_dataformat();

public:
    // bind socket to port p
    CServerConnection(int p, int id);
    // bind socket, port assigned automatically
    CServerConnection(int *p, int id);
    CServerConnection(Socket *s)
    {
        sock = s;
        m_status = 1;
    };
    ~CServerConnection() // close connection
        {};
    int accept(int); // accept connection (after bind) and wait int seconds
    CServerConnection *spawn_connection(); // accept after select for open socket
};

// Connection that acts as client
class CClientConnection : public CConnection
{
    Host *host;
    Host *lhost;

public:
    // connect to server at port p on host h
    //   ClientConnection(Host *h, int p, int id, sender_type st);
    CClientConnection(Host *, int, int, int retries = 20);
    ~CClientConnection() // close connection
    {
        delete lhost;
    };
};

class ConnectionList // list connections in a way that select can be used
{
    Liste<Connection> *m_connlist; // list of connections
    Liste<Connection> *m_channels; // list of connections

    fd_set m_fdvar; // field for select call
    int m_maxfd; // maximum socket id
    //Connection *open_sock; // socket for listening
public:
    ConnectionList(); // constructor
    ConnectionList(Connection *); // constructor (listens always at port)
    ~ConnectionList(); // destructor

    Connection *get_conn(char *sid);

    void add(Connection *c); // add connection
    void remove(Connection *c); // remove connection
    Connection *get_last() // get connection made recently
    {
        return m_connlist->get_last();
    };
    void reset() //
    {
        m_connlist->reset();
    };
    Connection *next() //
    {
        return m_connlist->next();
    };

    void add_ch(Connection *c); // add channel
    void remove_ch(Connection *c); // remove channel
    Connection *get_last_ch() // get channel made recently
    {
        return m_channels->get_last();
    };
    void reset_ch() //
    {
        m_channels->reset();
    };
    Connection *next_ch() //
    {
        return m_connlist->next();
    };

    Connection *wait_for_input(); // issue select call and return the
    // connection that shows the first event
    // issue select call and return a connection if there is an event or 0L otherwise
    Connection *check_for_input(float time = 0.0);

    char *get_registered_users(char *);
    char *get_last_wrl(void);
    int add_dynamic_view(char *id);
    int remove_dynamic_view(char *id);
    int broadcast_usr(char *act, char *usr);
    int broadcast_view(char *client_id, Message *p_Msg);
};
#endif
