/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_CONNECTION_H
#define EC_CONNECTION_H

#include <iostream>
#include <vector>

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <netinet/in.h>
#endif

#include <util/coExport.h>
#include "message.h"
#include "message_sender_interface.h"


typedef struct ssl_st SSL;
typedef struct ssl_ctx_st SSL_CTX;
typedef struct ssl_method_st SSL_METHOD;

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

namespace covise
{

class Host;
class SimpleServerConnection;
class SSLSocket;
class UDPSocket;
class UdpMessage;

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

class NETEXPORT Connection : public MessageSenderInterface
{
protected:
    friend class ServerConnection;
    friend class ConnectionList;
    class Socket *sock = nullptr; // Socket for connection
    int port; // port for connection
    int sender_id; // id of the sending process
    int send_type; // type of module for messages
    int peer_id_; // id of the peer process
    int peer_type_; // type of peer
    int message_to_do; // if more than one message has been read
    int bytes_to_process;
    unsigned long tru;
    char *read_buf = nullptr;
    Host *other_host = nullptr;
    int hostid; //hostid of remote host
    void (*remove_socket)(int);
    int get_id();
    int *header_int;

public:
    char convert_to; // to what format do we need to convert data?
    Connection();
    Connection(int sfd);
    virtual ~Connection(); // close connection (for subclasses)

    Socket *getSocket()
    {
        return sock;
    };
    void set_peer(int id, int type);
    int get_peer_id();
    int get_peer_type();

    int is_connected() // return true if connection is established
    {
        if (sock == NULL)
            return 0;
        return (get_id() != -1);
    };
    virtual int receive(void *buf, unsigned nbyte); // receive from socket
    virtual int send(const void *buf, unsigned nbyte); // send into socket
	virtual int recv_msg(Message *msg, char *ip = nullptr); // receive Message, can set ip to the ip adresss of the sender(for udp msgs)
    virtual int recv_msg_fast(Message *msg); // high-performace receive Message
    virtual bool sendMessage(const Message *msg) override; // send Message
    virtual bool sendMessage(const UdpMessage *msg) override; // send Message
    virtual int send_msg_fast(const Message *msg); // high-performance send Message
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
    int get_sendertype()
    {
        return (send_type);
    };
    int get_id(void (*remove_func)(int)); // give socket id
    int get_sender_id()
    {
        return sender_id;
    };
    void close(); // send close msg for partner and delete socket
    void close_inform(); // close without msg for partner
    int has_message()
    {
        //	if(message_to_do)
        //	    LOGINFO( "message_to_do == 1");
        //	else
        //	    LOGINFO( "message_to_do == 0");
        return message_to_do; // message is already read
    };
    void print()
    {
        std::cerr << "port: " << port << std::endl;
    };
    Host *get_host();
    const char *get_hostname();
};

class NETEXPORT UDPConnection : public Connection
{
public:
	UDPConnection(int id, int s_type, int p, const char* address);
	//receive a udp message from socket, return true on succsess (deletes old data and creates new data)
	bool recv_udp_msg(UdpMessage* msg);
	//send udp message to ip, if no ip given use member address. Retun true on succsess
	bool sendMessage(const UdpMessage* msg) override;
    bool send_udp_msg(const UdpMessage* msg, const char* ip = nullptr);
};
// Connection that acts as server
class NETEXPORT ServerConnection : public Connection
{
protected:
    void get_dataformat();

public:
    // bind socket to port p
    ServerConnection(int p, int id, int st);
    // bind socket, port assigned automatically
    ServerConnection(int *p, int id, int st);
    ServerConnection(Socket *s)
    {
        sock = s;
    };
    virtual ~ServerConnection() // close connection
        {};
    int acceptOne(); // accept connection (after bind)
    int acceptOne(int); // accept connection (after bind) and wait int seconds
    int listen(); // listen for connection (after bind)
    ServerConnection *spawn_connection(); // accept after select for open socket
    // accept after select for open socket
    SimpleServerConnection *spawnSimpleConnection();
};
class NETEXPORT ServerUdpConnection : public ServerConnection
{
public:
	ServerUdpConnection(UDPSocket* s);
	bool sendMessageTo(Message* msg, const char* address);
};

// Connection that acts as server
class NETEXPORT SimpleServerConnection : public ServerConnection
{
    void get_dataformat();
    char buffer[10001];
	size_t buflen;

public:
    // bind socket to port p
    SimpleServerConnection(int p, int id, int st);
    // bind socket, port assigned automatically
    SimpleServerConnection(int *p, int id, int st);
    SimpleServerConnection(Socket *s);
    const char *readLine();
};

// Connection that acts as client
class NETEXPORT ClientConnection : public Connection
{
    Host *host;
    Host *lhost;

public:
    // connect to server at port p on host h
    //   ClientConnection(Host *h, int p, int id, int st);
    ClientConnection(Host *host, int port, int id, int senderType, int retries = 20, double timeout = 0.0);
    ~ClientConnection(); // close connection
};

// Connection that acts as client, does not send a byteorder Byte
class NETEXPORT SimpleClientConnection : public Connection
{
    Host *host;
    Host *lhost;
    char buffer[10001];
	size_t buflen;

public:
    // connect to server at port p on host h
    SimpleClientConnection(Host *, int p, int retries = 20);
    ~SimpleClientConnection(); // close connection
    const char *readLine();
    void get_dataformat();
};

class NETEXPORT DataManagerConnection : public ClientConnection
{
    // make connection to datamanager
public:
    //DataManagerConnection(Host *hp, int pp, int my_id, int st): ClientConnection((Host*)hp, (int)pp, (int)my_id, (int) st) {};
    DataManagerConnection(Host *hp, int pp, int my_id, int st)
        : ClientConnection(hp, pp, my_id, st){};
    // contact datamanager on port p at host h

    // !!! irgendwie mag die HP das nicht
    //    DataManagerConnection(int pp, int my_id, int st): ClientConnection(NULL,pp,my_id,st) {};
    // !!! in der Version vor dem Linux Mergen war das so:
    DataManagerConnection(int pp, int my_id, int st)
        : ClientConnection((Host *)NULL, pp, my_id, st){};

    // contact local datamanager on port p
    ~DataManagerConnection() // close connection
        {};
};

class NETEXPORT ControllerConnection : public ClientConnection
{
    // make connection to controller
public:
    ControllerConnection(Host *h, int p, int id, int st)
        : ClientConnection(h, p, id, st){};
    // contact controller on port p at host h
    ~ControllerConnection() // close connection
        {};
};

#ifdef MULTICAST
class NETEXPORT MulticastConnection : public Connection
{
public:
    MulticastConnection(int id, int s_type, int p, char *MulticastGroup = "224.10.10.10", int ttl = 200);
};
#endif

class NETEXPORT ConnectionList // list connections in a way that select can be used
{
    long curidx = -1; // current index into vector
    std::vector<Connection *> connlist; // list of connections
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
    Connection *get_last();
    Connection *wait_for_input(); // issue select call and return the
    // connection that shows the first event
    // issue select call and return a
    Connection *check_for_input(float time = 0.0);
    // connection if there is an event or 0L otherwise
    void reset();
    Connection *next();
    //Connection at(int index);					  // get specific entry from listpos i
    int count(); // returns the number of current elements
};

#ifdef HAVE_OPENSSL
class NETEXPORT SSLConnection : public Connection
{
public:
    typedef int(PasswordCallback)(char *buf, int size, int rwflag, void *userdata);

    SSLConnection();
    SSLConnection(int sfd);
    ~SSLConnection();

    enum SSL_STATE
    {
        SSL_READY,
        SSL_CLOSED
    };

    int AttachSSLToSocket(Socket *sock);

    int receive(void *buf, unsigned nbyte); // receive from socket
    int send(const void *buf, unsigned nbyte); // send into socket
    int recv_msg(Message *msg); // receive Message
    int send_msg(const Message *msg); // send Message
    bool sendMessage(const Message *msg) override;
    const char *readLine(); // Read line
    int get_id(void (*remove_func)(int));
    int get_id() const;
    bool IsClosed() const;
    std::string getPeerAddress();
    SSL *mSSL;

protected:
    struct sockaddr_in mSA_client;
    struct sockaddr_in mSA_server;

    static SSL_CTX *mCTX;
    int mSFD;
    int mClieDsc;
    SSL_STATE mState;
	size_t mBuflen;
    char mBuffer[10001];

    void setPasswdCallback(PasswordCallback *, void *userData);
    void validateConnState();

    PasswordCallback *mSSLCB;
    void *mSSLUserData;

private:
};

class NETEXPORT SSLServerConnection : public SSLConnection
{
public:
    SSLServerConnection(PasswordCallback *cb, void *userData);
    SSLServerConnection(int *p, int id, int s_type, PasswordCallback *cb, void *userData);
    SSLServerConnection(int p, int id, int st, PasswordCallback *cb, void *userData);
    SSLServerConnection(SSLSocket *socket, PasswordCallback *cb, void *userData);
    ~SSLServerConnection();

    int accept();
    int sock_accept();
    int listen();

    std::string getSSLSubjectUID();
    std::string getSSLSubjectName();

    SSLServerConnection *spawnConnection();

protected:
private:
};

class NETEXPORT SSLClientConnection : public SSLConnection
{
public:
    SSLClientConnection(Host *, int p, PasswordCallback *cb, void *userData /*,int retries = 20*/);
    ~SSLClientConnection();

    int connect();
    void evaluate();
    std::string getSSLSubject();

    Host *host;
    Host *lhost;

protected:
    void getErrorMessage(long result);
};
#endif
}
#endif
