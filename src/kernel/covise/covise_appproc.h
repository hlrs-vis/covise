/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISE_APPPROC_H
#define COVISE_APPPROC_H

#include "covise_process.h"
#include "covise.h"

class ShmAccess;

namespace covise
{

class COVISEEXPORT ApplicationProcess : public OrdinaryProcess
{
//friend class DO_PartitionedObject;
#ifdef CRAY
    DataManagerProcess *datamgr;
#endif
    const DataManagerConnection *datamanager;
    ShmAccess *shm; // pointer to the sharedmemory
    //List<coDistributedObject> *part_obj_list;
protected:
    void process_msg_from_dmgr(Message *); // handle msg from datamgr
    int id;
    int instance;

public:
    static ApplicationProcess *approc;
    ApplicationProcess(const char *name, int id);
    ApplicationProcess(const char *n, int arc, char *arv[],
                       sender_type send_type = APPLICATIONMODULE);
    ~ApplicationProcess(); // destructor
    void handle_shm_msg(Message *);
    void contact_controller(int port, Host *host);
    void contact_datamanager(int port); // build connection to datamanager
    void send_data_msg(Message *); // send message to the datamanager
    void recv_data_msg(Message *); // recv a message from the datamanager
    void exch_data_msg(Message *, int...); // exch message with datamanager (recommended)
    //void add_new_part_obj(coDistributedObject *po) { part_obj_list->add(po); };
    // gets part obj out of list
    //coDistributedObject *get_part_obj(char *pname);
    Message *wait_for_ctl_msg(); // wait for a message from the controller
    Message *wait_for_data_msg(); // wait for a message from the datamanager
    Message *check_for_ctl_msg(float time = 0.0);
    int check_msg_queue(); // returns true if message is available in queue
    // wait for a message from the controller at most time seconds
    void *get_shared_memory_address();
    int get_instance()
    {
        return (instance);
    };
    int get_id()
    {
        return (id);
    };
    // nur temporaer eingefuegt, zum testen !! ^
};

// process for the userinterface
class COVISEEXPORT UserInterface : public ApplicationProcess
{
public:
    //    UserInterface(char *name, int id) :
    //	    ApplicationProcess(name, id, USERINTERFACE) {};
    UserInterface(const char *n, int arc, char *arv[])
        : ApplicationProcess(n, arc, arv, USERINTERFACE){};
    ~UserInterface() // destructor
        {};
    void contact_controller(int, Host *);
};
}
#endif
