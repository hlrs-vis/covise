/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_PROCESS_H
#define EC_PROCESS_H

#include "covise.h"
#include <util/covise_list.h>
#include "covise_msg.h"
#include <shm/covise_shm.h>
#include <net/covise_connect.h>
#ifndef NO_SIGNALS
#include "covise_signal.h"
#endif

#ifndef _WIN32
#include <sys/param.h>
#endif

/***********************************************************************\ 
 **                                                                     **
 **   Process classes                              Version: 1.5         **
 **                                                                     **
 **                                                                     **
 **   Description  : All classes that work as the processes own data    **
 **                  and function base. Each process should make a new  **
 **                  object of its appropriate class type to have       **
 **                  access to the basic functionality.                 **
 **                                                                     **
 **   Classes      : Process, OrdinaryProcess, ApplicationProcess,      **
 **                  DataManagerProcess, UserInterfaceProcess,          **
 **                  Controller, ObjectEntry                            **
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
 **                  27.05.93  Ver 1.1 remote processstart added,       **
 **                                    DMEntry added,                   **
 **                                    partial redesign                 **
 **                                                                     **
\***********************************************************************/

/* each process has to create a process object according to its type.
   Process is the base class, that handles shared memory attachment.

         +-----------+
         |  Process  |
         +-----------+
        /		\ 
       /		 \ 
   +----------+		+------------+
   | ordinary |		| controller |
   |  process |		|  process   |
+----------+		+------------+
| \  \  \ 
|  \  \  \ 
|   \  \  ------------------------------------
|    \  ----------------------		\ 
|     --------		\		 \ 
|		\		 \		  \ 
+-------------+ +-------------+ +---------------+ +-------------+
| Application | | DataManager | | UserInterface | | Transfermgr |
|   process   | |   process   | |   process     | |   process   |
+-------------+ +-------------+ +---------------+ +-------------+

*/

#if !defined(_WIN32) && !defined(__linux__)
#define COVISE_Signals
#endif

namespace covise
{

class Connection;
class ConnectionList;
class ControllerConnection;
class Host;
class Message;

COVISEEXPORT int execProcessWMI(const char *commandLine, const char *wd, const char *host, const char *user, const char *password);

extern int COVISE_debug_level;

void COVISEEXPORT exitOnInappropriateCmdArgs(int argC, char* argV[]);

class COVISEEXPORT Process // base class for all processes
{
protected:
    const char *name; // name of the process
    int id; // number which identifies each process
    int hostid; // id of host, this process is running on
    Host *host; // host on which it is running, initialized by IP address
    char *covise_hostname; // official hostname sent from controller
    sender_type send_type; // type of this process
    ConnectionList *list_of_connections; // list of all connections
    List<Message> *msg_queue;
#ifndef _WIN32
#ifdef COVISE_Signals
#ifndef NO_SIGNALS
#ifdef __linux__
#define SignalHandler CO_SignalHandler
#endif
    SignalHandler sig_handler;
#endif
#endif
#endif
public:
    //void add_connection(Connection *c){list_of_connections->add(c);};
    //void remove_connection(Connection *c){list_of_connections->remove(c);};
    ConnectionList *getConnectionList()
    {
        return (list_of_connections);
    };
    static Process *this_process;
    Process(const char *n, int i, sender_type st); // initialize
    // initialize
    Process(const char *n, int i, sender_type st, int port);
    // initialize
    Process(const char *n, int arc, char *arv[], sender_type st);
    virtual ~Process(); // destructor
    void init_env();
    virtual void handle_shm_msg(Message *);
    void set_send_type(sender_type st)
    {
        send_type = st;
    };
    Message *wait_for_msg(); // wait for a message
    Message *check_queue();
    Message *check_for_msg(float time = 0.0); // wait for a message
    Message *wait_for_msg(int, Connection *); // wait for a specific message
    // wait for specific messages
    Message *wait_for_msg(int *, int, Connection *);
    int get_id()
    {
        return id;
    };
    char *get_list_of_interfaces();
    const char *getName()
    {
        return name;
    };
    const char *get_hostname(); // official COVISE controller hostname
    Host *get_covise_host(); // COVISE_HOST environment variable based host
    Host *get_host() // IP address based host
    {
        return host;
    };
    //    int set_default_hostname(char *);
    void covised_rexec(Host *, char *);
    void delete_msg(Message *m)
    {
        delete m;
    };
    int get_hostid()
    {
        return (hostid);
    };
};

class COVISEEXPORT OrdinaryProcess : public Process
{
protected:
    ControllerConnection *controller;

public:
    //
    // initialize process
    OrdinaryProcess(const char *name, int id, sender_type st)
        : Process(name, id, st)
        , controller(NULL){};

    // initialize process
    OrdinaryProcess(const char *n, int arc, char *arv[], sender_type st)
        : Process(n, arc, arv, st)
        , controller(NULL){};

    ~OrdinaryProcess(){
        //	delete controller;
    }; // destructor
    void send_ctl_msg(const Message *); // send a message to the controller
    // build connection to the controller:
    void send_ctl_msg(TokenBuffer); // send a Tokenbuffer to the controller
    int get_socket_id(void (*remove_func)(int));
    virtual void contact_controller(int, Host *)
    {
        abort();
    }
    int is_connected();
    ControllerConnection *getControllerConnection()
    {
        return controller;
    }
};

class ShmAccess;

class COVISEEXPORT AccessEntry
{
    friend class ObjectEntry;
    friend class DataManagerProcess;
    access_type acc; // general access type allowed
    access_type curr_acc; // connection to accessing process
    Connection *conn; // access currently actually used
public:
    AccessEntry(access_type a, access_type c_a, Connection *c)
    {
        acc = a;
        conn = c;
        curr_acc = c_a;
    };
    void print()
    {
        cerr << "Access: ";
        conn->print();
#ifndef _WIN32
        cerr << " = " << acc << endl;
#endif
    };
};

// process for the external access
class COVISEEXPORT SimpleProcess : public OrdinaryProcess
{
public:
    SimpleProcess(char *name, int arc, char *arv[]);
    ~SimpleProcess();

private:
    void contact_controller(int, Host *);
};
}
#endif
