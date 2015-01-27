/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_CONNECTION_H
#define EC_CONNECTION_H

#include "coTypes.h"
#include "covise_socket.h"
#include "covise_msg.h"
#include "covise_list.h"
#include "covise_conv.h"

/*
 $Log:  $
 * Revision 1.9  1994/03/24  16:56:26  zrf30125
 * ConnectionList completed
 *
 * Revision 1.8  94/02/18  14:31:52  zrfg0125
 * ~Connection bug
 *
 * Revision 1.7  94/02/18  14:19:44  zrfg0125
 * ~Connection bug solved
 *
* Revision 1.6  94/02/18  14:08:18  zrfg0125
 * read_buf no longer static
 *
 * Revision 1.5  93/12/10  14:19:06  zrfg0125
 * modifications to speedup write and read calls
 *
 * Revision 1.4  93/10/20  15:14:18  zrhk0125
 * socket cleanup improved
 *
 * Revision 1.3  93/10/08  19:26:59  zrhk0125
 * shortened include filename
 * conversion included as connection dependent
 *
 * Revision 1.2  93/09/30  17:09:04  zrhk0125
 * basic modifications for CRAY
 *
 * Revision 1.1  93/09/25  20:39:31  zrhk0125
 * Initial revision
 *
 */

const int EC_SERVER = 0;
const int EC_CLIENT = 1;
class SimpleServerConnection;
#ifdef CRAY
#define WRITE_BUFFER_SIZE 393216
#else
#define WRITE_BUFFER_SIZE 64000
#endif
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
 **                  26.05.93  Ver 1.1  sender_type and sender_id       **
 **                                     introduced                      **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

class Connection
{
protected:
    friend class ServerConnection;
    friend class ConnectionList;
    class Socket *sock; // Socket for connection
    int port; // port for connection
    int sender_id; // id of the sending process
    sender_type send_type; // type of module for messages
    int peer_id_; // id of the peer process
    sender_type peer_type_; // type of peer
    int message_to_do; // if more than one message has been read
    int bytes_to_process;
    unsigned long tru;
    char *read_buf;
    Host *other_host;
    int hostid; //hostid of remote host
    void (*remove_socket)(int);
    int get_id()
    {
        //cerr << "sock == " << sock << " id: " << sock->get_id() << endl;
        if (sock)
            return sock->get_id();
        return -1;
    }; // give socket id
public:
    char convert_to; // to what format do we need to convert data?
    Connection()
    {
        convert_to = DF_NONE;
        message_to_do = 0;
        //		   print_comment(__LINE__, __FILE__, "message_to_do == 0");
        read_buf = new char[READ_BUFFER_SIZE];
        bytes_to_process = 0;
        remove_socket = 0L;
        hostid = -1;
        peer_id_ = 0;
        peer_type_ = UNDEFINED; // prepare connection (for subclasses)
    };
    Connection(int sfd)
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
    virtual ~Connection() // close connection (for subclasses)
    {
        delete[] read_buf;
        delete sock;
    };

    Socket *getSocket()
    {
        return sock;
    };
    void set_peer(int id, sender_type type);
    int get_peer_id();
    sender_type get_peer_type();

    int is_connected() // return true if connection is established
    {
        if (sock == NULL)
            return 0;
        return (get_id() != -1);
    };
    int receive(void *buf, unsigned nbyte); // receive from socket
    int send(const void *buf, unsigned nbyte); // send into socket
    int recv_msg(Message *msg); // receive Message
    int send_msg(const Message *msg); // send Message
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
    sender_type get_sendertype()
    {
        return (send_type);
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
        cerr << "port: " << port << endl;
    };
    Host *get_host()
    {
        return sock->get_host();
    };
    char *get_hostname()
    {
        return sock->get_hostname();
    };
};

// Connection that acts as server
class ServerConnection : public Connection
{
protected:
    void get_dataformat();

public:
    // bind socket to port p
    ServerConnection(int p, int id, sender_type st);
    // bind socket, port assigned automatically
    ServerConnection(int *p, int id, sender_type st);
    ServerConnection(Socket *s)
    {
        sock = s;
    };
    virtual ~ServerConnection() // close connection
        {};
    int accept(); // accept connection (after bind)
    int accept(int); // accept connection (after bind) and wait int seconds
    int listen() // listen for connection (after bind)
    {
        return sock->listen();
    };
    ServerConnection *spawn_connection(); // accept after select for open socket
    // accept after select for open socket
    SimpleServerConnection *spawnSimpleConnection();
};

// Connection that acts as server
class SimpleServerConnection : public ServerConnection
{
    void get_dataformat();
    char buffer[10001];
    int buflen;

public:
    // bind socket to port p
    SimpleServerConnection(int p, int id, sender_type st);
    // bind socket, port assigned automatically
    SimpleServerConnection(int *p, int id, sender_type st);
    SimpleServerConnection(Socket *s);
    const char *readLine();
};

// Connection that acts as client
class ClientConnection : public Connection
{
    Host *host;
    Host *lhost;

public:
    // connect to server at port p on host h
    //   ClientConnection(Host *h, int p, int id, sender_type st);
    ClientConnection(Host *, int, int, sender_type, int retries = 20, double timeout = 0.0);
    ~ClientConnection() // close connection
    {
        delete lhost;
    };
}; // Connection that acts as client, does not send a byteorder Byte
class SimpleClientConnection : public Connection
{
    Host *host;
    Host *lhost;
    char buffer[10001];
    int buflen;

public:
    // connect to server at port p on host h
    SimpleClientConnection(Host *, int p, int retries = 20);
    ~SimpleClientConnection() // close connection
    {
        delete lhost;
    };
    const char *readLine();
};

class DataManagerConnection : public ClientConnection
{
    // make connection to datamanager
public:
    //DataManagerConnection(Host *hp, int pp, int my_id, sender_type st): ClientConnection((Host*)hp, (int)pp, (int)my_id, (sender_type) st) {};
    DataManagerConnection(Host *hp, int pp, int my_id, sender_type st)
        : ClientConnection(hp, pp, my_id, st){};
    // contact datamanager on port p at host h

    // !!! irgendwie mag die HP das nicht
    //    DataManagerConnection(int pp, int my_id, sender_type st): ClientConnection(NULL,pp,my_id,st) {};
    // !!! in der Version vor dem Linux Mergen war das so:
    DataManagerConnection(int pp, int my_id, sender_type st)
        : ClientConnection((Host *)NULL, pp, my_id, st){};

    // contact local datamanager on port p
    ~DataManagerConnection() // close connection
        {};
};

class ControllerConnection : public ClientConnection
{
    // make connection to controller
public:
    ControllerConnection(Host *h, int p, int id, sender_type st)
        : ClientConnection(h, p, id, st){};
    // contact controller on port p at host h
    ~ControllerConnection() // close connection
        {};
};
class UDPConnection : public Connection
{
public:
    UDPConnection(int id, sender_type s_type, int p, char *address);
};
#ifdef MULTICAST
class MulticastConnection : public Connection
{
public:
    MulticastConnection(int id, sender_type s_type, int p, char *MulticastGroup = "224.10.10.10", int ttl = 200);
};
#endif

class ConnectionList // list connections in a way that select can be used
{
    List<Connection> *connlist; // list of connections
    fd_set fdvar; // field for select call
    int maxfd; // maximum socket id
    ServerConnection *open_sock; // socket for listening
public:
    ConnectionList(); // constructor
    ConnectionList(ServerConnection *); // constructor (listens always at port)
    ~ConnectionList(); // destructor
    void add_open_conn(ServerConnection *c);
    void add(Connection *c); // add connection
    void remove(Connection *c); // remove connection
    void deleteConnection(Connection *c);
    Connection *get_last() // get connection made recently
    {
        return connlist->get_last();
    };
    Connection *wait_for_input(); // issue select call and return the
    // connection that shows the first event
    // issue select call and return a
    Connection *check_for_input(float time = 0.0);
    // connection if there is an event or 0L otherwise
    void reset() //
    {
        connlist->reset();
    };
    Connection *next() //
    {
        return connlist->next();
    };
};
#endif
