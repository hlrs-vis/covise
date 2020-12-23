/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_DMGRPROC_H
#define COVISE_DMGRPROC_H

#include <covise/covise_process.h>
#include <shm/covise_shm.h>
#include "covise_avl_tree.h"
#include <covise/covise_global.h>
#include <covise/covise_signal.h>
#include <covise/covise.h>
#include <signal.h>

#include <net/dataHandle.h>
#ifndef _WIN32
#include <sys/time.h>
#endif
#ifdef _AIX
int kill(pid_t pid, int sig);
#endif

/***********************************************************************\ 
 **                                                                     **
 **   Datamanager Process classes		  Version: 1.5         **
 **                                                                     **
 **                                                                     **
 **   Description  : All classes that work as the processes own data    **
 **                  and function base. Each process should make a new  **
 **                  object of its appropriate class type to have       **
 **                  access to the basic functionality.                 **
 **                                                                     **
 **   Classes      : DataManagerProcess, ObjectEntry		       **
 **                                                                     **
 **   Copyright (C) 1996     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  19.11.96  Ver 1.0                                  **
 **                                                                     **
\***********************************************************************/

#if !defined(__linux__) && !defined(_WIN32)
#define COVISE_Signals
#endif

namespace covise
{
class DMEntry;

class DMGREXPORT ObjectEntry
{
    friend int ObjectEntry_compare(ObjectEntry *, ObjectEntry *);
    friend class DataManagerProcess;
    DataHandle name; // name of the object
    int version; // version of the object
    int shm_seq_no; // shm_seq_no of the object
    int offset; // offset of the object
    int type; // type of the object
    Connection *owner; // connection to the process that created this object
    List<AccessEntry> *access; // list of access rights to this object
    DMEntry *dmgr; // pointer to the DM which sent the object (can be like owner)
public:
    ObjectEntry()
    {
        name = 0L;
        shm_seq_no = offset = -1;
        dmgr = 0L;
        version = 1;
        access = new List<AccessEntry>;
    };
    ObjectEntry(const DataHandle& n);
    ObjectEntry(const DataHandle &n, int type, int no, int o, Connection *c, DMEntry *dm = 0L);
    ObjectEntry(const DataHandle &n, int no, int o, Connection *c, DMEntry *dm = 0L);

    int operator==(ObjectEntry &oe)
    {
        return (strcmp(name.data(), oe.name.data()) == 0 ? 1 : 0);
    };
    int operator>=(ObjectEntry &oe)
    {
        return (strcmp(name.data(), oe.name.data()) >= 0 ? 1 : 0);
    };
    int operator<=(ObjectEntry &oe)
    {
        return (strcmp(name.data(), oe.name.data()) <= 0 ? 1 : 0);
    };
    int operator>(ObjectEntry &oe)
    {
        return (strcmp(name.data(), oe.name.data()) > 0 ? 1 : 0);
    };
    int operator<(ObjectEntry &oe)
    {
        return (strcmp(name.data(), oe.name.data()) < 0 ? 1 : 0);
    };
    void add_access(Connection *c, access_type a, access_type c_a);
    void set_current_access(Connection *c, access_type c_a);
    access_type get_access_right(Connection *c);
    access_type get_access(Connection *c);
    void remove_access(Connection *c);
    void set_dmgr(DMEntry *dm)
    {
        dmgr = dm;
    };
    void pack_and_send_object(Message *msg, DataManagerProcess *dm);
    void pack_address(Message *msg);
    void print();
    ~ObjectEntry();
};

class DMGREXPORT DMEntry
{
    friend class DataManagerProcess;
    friend class ObjectEntry;

private:
    int id;
    Host *host;
    Connection *conn;
    Connection *data_conn;
    AVLTree<ObjectEntry> *objects;
    int transfermgr;
    DMEntry(int i, char *h, Connection *c);
    const char *get_hostname(void);
    Connection *get_conn(void)
    {
        return conn;
    };
    int make_data_conn(char *new_interface);
    int complete_data_conn(Message *msg);
    int recv_data_msg(Message *msg) // receive Data Message
    {
        return data_conn->recv_msg(msg);
    };
    int send_data_msg(const Message *msg) // send Data Message
    {
        return data_conn->sendMessage(msg);
    };

public:
    void print(){};
    int send_msg(Message *msg)
    {
        return conn->sendMessage(msg);
    };
};

class DMGREXPORT TMEntry
{
    friend class TransferManagerProcess;
    friend class ObjectEntry;

private:
    int id;
    Host *host;
    Connection *conn;
    Connection *data_conn;
    AVLTree<ObjectEntry> *objects;
    TMEntry(int i, char *h, Connection *c);
    int make_data_conn(char *new_interface);
    int complete_data_conn(Message *msg);
    int recv_data_msg(Message *msg) // receive Data Message
    {
        return data_conn->recv_msg(msg);
    };
    int send_data_msg(const Message *msg) // send Data Message
    {
        return data_conn->sendMessage(msg);
    };

public:
    void print(){};
};

class AddressOrderedTree;
class SizeOrderedTree;

class DMGREXPORT coShmAlloc : public ShmAccess
{
    class DataManagerProcess *dmgrproc;
    static class AddressOrderedTree *used_list;
    static class AddressOrderedTree *free_list;
    static class SizeOrderedTree *free_size_list;

public:
    coShmAlloc(int *key, DataManagerProcess *d);
    ~coShmAlloc()
    {
        delete shm;
    };
    coShmPtr *malloc(shmSizeType size);
    void get_shmlist(char *ptr)
    {
        shm->get_shmlist((int *)ptr);
    };
    void free(coShmPtr *adr)
    {
        free(adr->shm_seq_no, adr->offset);
    };
    void free(int shm_seq_no, shmSizeType offset);
    void print();
    void collect_garbage(){};
    void new_desk(void);
};

class DMGREXPORT DataManagerProcess : public OrdinaryProcess
{
#ifdef CRAY
    friend class ApplicationProcess;
#endif
    friend void ObjectEntry::pack_and_send_object(Message *, DataManagerProcess *);
    ServerConnection *transfermanager; // Connection to the transfermanager
    ServerConnection *tmpconn; // tmpconn for intermediate use
    coShmAlloc *shm; // pointer to the sharedmemory
    AVLTree<ObjectEntry> *objects;
    List<DMEntry> *data_mgrs;
    pid_t *pid_list;
    int no_of_pids;
    bool is_bad_; // connection failed?
    static int max_t;

public:
    DataManagerProcess(char *name, int id, int *key);
    //    DataManagerProcess(char *n, int arc, char *arv[]);
    ~DataManagerProcess()
    {
        //int sleep_count = 0,
        int i;
        save_object_id();
        //	delete shm;   // destructor (deletes shared memory)
        print_comment(__LINE__, __FILE__, "Anfang von ~DataManagerProcess");
        /*while(no_of_pids > 0 && sleep_count < 500)
         {
#if defined(__hpux)
            struct timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 50;

            select(0, 0, 0, 0, &timeout);
#else
            select(0, (fd_set *)0, (fd_set *)0, (fd_set *)0, &timeout);
#endif
            sleep_count++;
         }*/
        //cout << "sleep_count: " << sleep_count << endl;
        for (i = 0; i < no_of_pids; i++)
        {
//cout << "Killing process no. " << pid_list[i] << endl;
#ifndef _WIN32
            kill(pid_list[i], SIGTERM);
#else
            HANDLE hPid = OpenProcess(PROCESS_ALL_ACCESS, TRUE, pid_list[i]);
            TerminateProcess(hPid, SIGTERM);
            CloseHandle(hPid);
#endif
        }
        for (i = 0; i < SharedMemory::global_seq_no - 1; i++)
        {
            //cout << "Deleting SharedMemory no. " << i << endl;
            delete SharedMemory::shm_array[i];
        }
    };

    int DTM_new_desk(void);
    int rmv_rdmgr(char *hostname);
    void rmv_acc2objs(CO_AVL_Node<ObjectEntry> *root, Connection *conn);

    bool is_connected()
    {
        return is_bad_;
    };

    // build connection to the controller:
    void contact_datamanager(int port, Host *host);
    void contact_controller(int port, Host *host);
    void init_object_id();
    void save_object_id();
    void start_transfermanager(); // start the transfermgr.
    int get_trfmgr_port();
    void connect_trfmgr(Host *h, int port);
    void send_trf_msg(Message *); // send message to the transfermgr.
    void exch_trf_msg(Message *); // send message to the transfermgr.
    void prepare_for_contact(int *port); // perform bind and return port to use
    void wait_for_contact(); // establish prepared connection
    void wait_for_dm_contact(); // establish prepared connection
    int make_data_connection(char *name, char *new_interface);
    int complete_data_connection(Message *msg);
    Message *wait_for_msg(); // wait for a message
    Message *wait_for_msg(int, Connection *); // wait for a specific message
    // wait for specific messages
    Message *wait_for_msg(int *, int, Connection *);
    // take action according to msg
    int handle_msg(Message *msg);
    void ask_for_object(Message *msg); // answer requests immediately
    void has_object_changed(Message *msg); // answer requests immediately
    DataHandle get_all_hosts_for_object(const DataHandle &n); // looks for all hosts that have object
    // add new object in database
    int add_object(const DataHandle &n, int no, int o, Connection *c);
    // add new object in database
    int add_object(const DataHandle &n, int otype, int no, int o, Connection *c);
    int add_object(ObjectEntry *oe); // add new object in database
    ObjectEntry *get_object(const DataHandle &n); // get object from database
    // get object from database and take care that the
    // accesses are updated correctly:
    ObjectEntry *get_object(const DataHandle& n, Connection *c);
    ObjectEntry *get_local_object(const DataHandle &n); // get object only from local database
    int delete_object(const DataHandle& n); // delete object from database
    int destroy_object(const DataHandle &n, Connection *c); // remove obj from sharedmem.
    // create transferred object
    ObjectEntry *create_object_from_msg(Message *msg, DMEntry *dme);
    // update from transferred object
    int update_object_from_msg(Message *msg, DMEntry *dme);
    int forward_new_part(Message *); // sends new part of partitioned object
    int receive_new_part(Message *); // receives new part of partitioned object
    // inserts new part of
    int insert_new_part(ObjectEntry *, ObjectEntry *);
    // partitioned object in shm representation
    coShmPtr *shm_alloc(ShmMessage *msg); // allocate memory depending on msg
    // returns offset into shared memory
    // allocate memory of size
    coShmPtr *shm_alloc(int type, shmSizeType msize = 0);
    // returns offset into shared memory
    int shm_free(coShmPtr *); // free memory (recursive !!)
    void send_to_all_connections(Message *);
    int shm_free(int, int); // free memory (recursive !!)
    void print_shared_memory_statistics()
    {
        shm->print();
    };
    void *get_shared_memory_address()
    {
        return shm->get_pointer();
    };
    void signal_handler(int signo, void *);
    void remove_pid(int pid)
    {
        int i;
        for (i = 0; i < no_of_pids && pid_list[i] != pid; i++)
            ;
        if (i < no_of_pids)
        {
            pid_list[i] = pid_list[no_of_pids - 1];
            no_of_pids--;
        }
    }
    void clean_all(int);
    // nur temporaer eingefuegt, zum testen !! ^
};

// not yet implemented
class DMGREXPORT TransferManagerProcess : public OrdinaryProcess
{
    DataManagerConnection *datamanager;
    AVLTree<ObjectEntry> *objects;
    List<TMEntry> *transfer_mgrs;
    ServerConnection *tmpconn; // tmpconn for intermediate use
    ShmAccess *shm; // pointer to the sharedmemory
public:
    TransferManagerProcess(char *n, int arc, char *arv[]);
    //    ~TransferManagerProcess() {
    //		delete datamanager;
    //	};
    void contact_transfermanager(int port, Host *host);
    void contact_datamanager(int port); // build connection to datamanager
    void contact_controller(int, Host *){};
    void prepare_for_contact(int *port); // perform bind and return port to use
    void wait_for_tm_contact(); // establish prepared connection
    int make_data_connection(char *name, char *new_interface);
    int complete_data_connection(Message *msg);
    int handle_msg(Message *msg); // take action according to msg
    void handle_shm_msg(Message *){};
    void *get_shared_memory_address()
    {
        return shm->get_pointer();
    };
    // nur temporaer eingefuegt, zum testen !! ^
};

class DMGREXPORT DmgrMessage : public ShmMessage // message especially for memory allocation
{
public: // at the datamanager
    DmgrMessage(){};
    int process_new_object_list(DataManagerProcess *dmgr);
    int process_list(DataManagerProcess *dmgr);
};

#ifdef shm_ptr
#undef shm_ptr
#endif
}
#endif
