/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "covise.h"
#include "covise_appproc.h"
#include <shm/covise_shm.h>
#include <net/covise_socket.h>
#include <net/covise_host.h>
#ifdef _WIN32
typedef int pid_t;
#endif

#ifdef _CRAYT3E
#include <dmgr/dmgr.h>
#endif

using namespace covise;

ApplicationProcess *ApplicationProcess::approc = NULL;

/***********************************************************************\ 
 **                                                                     **
 **   ApplicationProcess class Routines            Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The ApplicationProcess handles the infrastructure  **
 **                  for the environment                                **
 **                                                                     **
 **   Classes      : ApplicationProcess                                 **
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
 **                  26.05.93  Ver 1.1 now the shared memory key comes  **
 **                                    from the datamanager             **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

ApplicationProcess::~ApplicationProcess()
{
    //	delete datamanager;
    //delete part_obj_list;
    delete shm;
}; // destructor

void *ApplicationProcess::get_shared_memory_address()
{
    return shm->get_pointer();
}

void ApplicationProcess::send_data_msg(Message *msg)
{
#ifdef CRAY
    datamgr->handle_msg(msg);
#else
    if (datamanager->send_msg(msg) == COVISE_SOCKET_INVALID)
        list_of_connections->remove(datamanager);
#endif
}

void ApplicationProcess::recv_data_msg(Message *msg)
{
#ifndef CRAY
    datamanager->recv_msg(msg);
    msg->conn = datamanager;
#endif
}

void ApplicationProcess::exch_data_msg(Message *msg, int count...)
{
    va_list ap;

#ifdef DEBUG
    char tmp_str[255];
#endif

    va_start(ap, count);

    // Changed aw 05/00 : Always check for SOCKET_CLOSED
    int *type_list = new int[count + 1];
    for (int i = 0; i < count; i++)
        type_list[i] = va_arg(ap, int);
    va_end(ap);
    type_list[count] = COVISE_MESSAGE_SOCKET_CLOSED;

#ifdef CRAY
    datamgr->handle_msg(msg);
#else
    if (datamanager->send_msg(msg) == COVISE_SOCKET_INVALID)
    {
        list_of_connections->remove(datamanager);
        delete[] type_list;
        return;
    }
    msg->data = DataHandle{};

    int type_match = 0;
    while (!type_match)
    {
        datamanager->recv_msg(msg);
        for (int i = 0; i <= count; i++)
            if (msg->type == type_list[i])
                type_match = 1;
        if (type_match == 0)
        {
            if (msg->type == COVISE_MESSAGE_NEW_SDS)
            {
                handle_shm_msg(msg);
                msg->data = DataHandle{};
            }
            else
            {
                Message *list_msg = new Message;
                list_msg->copyAndReuseData(*msg);
                msg_queue->add(list_msg);
#ifdef DEBUG
                sprintf(tmp_str, "msg %s added to queue", covise_msg_types_array[msg->type]);
                print_comment(__LINE__, __FILE__, tmp_str);
#endif
            }
        }
    }

    msg->conn = datamanager;
#endif
    delete[] type_list; // Uwe Woessner
}

void ApplicationProcess::contact_datamanager(int p)
{
    Message *msg;
    int len = 0;
    datamanager = new DataManagerConnection(p, id, send_type);
    //    print_comment(__LINE__, __FILE__, "nach new Connection");
    list_of_connections->add(datamanager);
    msg = new Message{ COVISE_MESSAGE_GET_SHM_KEY, DataHandle{} };
    exch_data_msg(msg, 1, COVISE_MESSAGE_GET_SHM_KEY);
    if (msg->type != COVISE_MESSAGE_GET_SHM_KEY || msg->data.data() == nullptr)
    {
        cerr << "didn't get GET_SHM_KEY\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    shm = new ShmAccess(msg->data.accessData());
    delete msg;
}

int ApplicationProcess::check_msg_queue()
{
    msg_queue->reset();
    return (msg_queue->next() != NULL);
}

Message *ApplicationProcess::wait_for_ctl_msg()
{
    char tmp_str[255];
#ifdef CRAY
    Connection *conn;
    Message *msg = new Message;
    int is_data_msg = 0;
    do
    {
        conn = list_of_connections->wait_for_input();
        conn->recv_msg(msg);
        // this is a special case, if datamanager and applicationmodule
        // are merged into one executable on a machine w/o shared memory
        switch (msg->type)
        {
        case PREPARE_CONTACT_DM:
            is_data_msg = datamgr->handle_msg(msg);
            list_of_connections->add(datamgr->list_of_connections->get_last());
            break;
        case DM_CONTACT_DM:
            is_data_msg = datamgr->handle_msg(msg);
            list_of_connections->add(datamgr->list_of_connections->get_last());
            break;
        case NEW_SDS:
            break;
        case SOCKET_CLOSED:
            list_of_connections->remove(conn);
        //delete conn;
        default:
            is_data_msg = datamgr->handle_msg(msg);
            break;
        }
        if (is_data_msg == 2)
            if (msg->conn->send_msg(msg) == COVISE_SOCKET_INVALID)
                list_of_connections->remove(msg->conn);
    } while (is_data_msg);
    return (msg);
#else
    Message *msg;
    int end = 0;

    //    print_comment(__LINE__, __FILE__, "in wait_for_ctl_msg");
    while (!end)
    {
        msg = wait_for_msg();
        sprintf(tmp_str, "msg->type %d received", msg->type);
        print_comment(__LINE__, __FILE__, "msg->type %d received", msg->type);
        if (msg->conn == datamanager)
        {
            print_comment(__LINE__, __FILE__, "process_msg_from_dmgr");
            process_msg_from_dmgr(msg);
             msg->data = DataHandle();
            delete msg;
        }
        else
            end = 1;
    }
    print_comment(__LINE__, __FILE__, "return(msg)");
    return (msg);
#endif
}

Message *ApplicationProcess::check_for_ctl_msg(float time)
{
#ifdef CRAY
    Connection *conn;
    Message *msg = new Message;
    int is_data_msg = 0;
    conn = list_of_connections->check_for_input(time);
    if (conn)
    {
        conn->recv_msg(msg);
        // this is a special case, if datamanager and applicationmodule
        // are merged into one executable on a machine w/o shared memory
        switch (msg->type)
        {
        case PREPARE_CONTACT_DM:
            is_data_msg = datamgr->handle_msg(msg);
            list_of_connections->add(datamgr->list_of_connections->get_last());
            break;
        case DM_CONTACT_DM:
            is_data_msg = datamgr->handle_msg(msg);
            list_of_connections->add(datamgr->list_of_connections->get_last());
            break;
        case NEW_SDS:
            break;
        case SOCKET_CLOSED:
            list_of_connections->remove(conn);
        //delete conn;
        default:
            is_data_msg = datamgr->handle_msg(msg);
            break;
        }
        if (is_data_msg == 2)
            if (msg->conn->send_msg(msg) == COVISE_SOCKET_INVALID)
                list_of_connections->remove(msg->conn);
        return (msg);
    }
    else
        return 0L;
#else
    Message *msg;

    // print_comment(__LINE__, __FILE__, "in check_for_ctl_msg");
    msg = check_for_msg(time);
    while ((msg != NULL)
           && (msg->conn == datamanager))
    {
        print_comment(__LINE__, __FILE__, "msg->type %d received", msg->type);
        print_comment(__LINE__, __FILE__, "process_msg_from_dmgr");
        process_msg_from_dmgr(msg);
        msg->data = DataHandle();
        delete msg;
        print_comment(__LINE__, __FILE__, "recheck for msg");
        msg = check_for_msg(time);
    }
    if (msg != NULL)
    {
        print_comment(__LINE__, __FILE__, "msg->type %d received", msg->type);
        print_comment(__LINE__, __FILE__, "return(msg)");
    }
    else
    {
        print_comment(__LINE__, __FILE__, "return NULL");
    }
    return msg;
#endif
}

void ApplicationProcess::process_msg_from_dmgr(Message *msg)
{
#ifndef CRAY
    switch (msg->type)
    {
    case COVISE_MESSAGE_NEW_SDS:
        handle_shm_msg(msg);
        break;
    default:
        break;
    }
#endif
}

static ApplicationProcess *appl_process;

#ifndef _WIN32
#ifdef COVISE_Signals
//=====================================================================
//
//=====================================================================
void appproc_signal_handler(int, void *)
{
    delete appl_process;
    sleep(1);
    exit(EXIT_FAILURE);
}
#endif
#endif

// initialize process
ApplicationProcess::ApplicationProcess(const char *name, int id)
    : OrdinaryProcess(name, id, APPLICATIONMODULE)
{
    approc = appl_process = this;
    datamanager = NULL;
    //part_obj_list = new List<coDistributedObject>;
    shm = NULL;
#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)appproc_signal_handler, NULL);
    // SIGPIPE is handled by Process::Process
    //    sig_handler.addSignal(SIGPIPE,(void *)appproc_signal_handler,NULL);
    sig_handler.addSignal(SIGTERM, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)appproc_signal_handler, NULL);
//    init_emergency_message();
#endif
}

ApplicationProcess::ApplicationProcess(const char *n, int argc, char *argv[],
                                       sender_type send_type)
    : OrdinaryProcess(n, argc, argv, send_type)
{
    //fprintf(stderr,"---- in ApplicationProcess::ApplicationProcess\n");
    //part_obj_list = NULL;
    static Host *tmphost;
    int pid;
    unsigned int uport;
#ifdef DEBUG
    char tmp_str[255];
#endif
    shm = NULL;
    if ((argc < 7) || (argc > 8))
    {
        print_comment(__LINE__, __FILE__, "ApplicationModule with inappropriate arguments called: argc=%d", argc);
        for (int i = 0; i < argc; ++i)
        {
            print_comment(__LINE__, __FILE__, "%d: %s", i, argv[i]);
        }
        print_exit(__LINE__, __FILE__, 1);
    }
    id = atoi(argv[3]);
    tmphost = new Host(argv[2]);

    //fprintf(stderr,"---- contact_controller\n");

    datamanager = NULL;
    contact_controller(atoi(argv[1]), tmphost);
    // darf ich das? Uwe WOessner delete tmphost;
    if (!is_connected())
        return;
    approc = appl_process = this;
//part_obj_list = new List<coDistributedObject>;
#ifndef CRAY
    Message *msg = wait_for_ctl_msg();
    if (msg->type == COVISE_MESSAGE_APP_CONTACT_DM)
    {
        //	cerr << "contact DM at port " << *(int *)msg->data.data() << "\n";
        uport = *(int *)msg->data.data();
        swap_byte(uport);
        //	cerr << "swapped port: " << uport << endl;
        contact_datamanager((int)uport);
        msg->type = COVISE_MESSAGE_SEND_APPL_PROCID;
        pid = getpid();
        msg->data = DataHandle{ (char*)& pid, sizeof(pid_t), false };
#ifdef DEBUG
        sprintf(tmp_str, "SEND_APPL_PROCID: %d", *(pid_t *)msg->data);
        print_comment(__LINE__, __FILE__, tmp_str);
        sprintf(tmp_str, "pid: %d", pid);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        if (datamanager->send_msg(msg) == COVISE_SOCKET_INVALID)
        {
            list_of_connections->remove(datamanager);
            print_error(__LINE__, __FILE__,
                        "datamaner socket invalid in new ApplicationProcess");
        }
        //cerr << "contact succeeded\n";
    }
    else
    {
        cerr << "wrong message instead of APP_CONTACT_DM (" << msg->type << ")" << endl;
        cerr << "Did you specify Shared Memory type \"none\" in your covise config?" << endl;
        cerr << "Please, change it to shm." << endl;
        print_exit(__LINE__, __FILE__, 1);
    }
    delete msg;
#else
    int key = 1;
    print_comment(__LINE__, __FILE__, "vor new DataManagerProcess");
    datamgr = new DataManagerProcess("DataManager", id, &key);
#endif
#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGPIPE, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGTERM, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)appproc_signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)appproc_signal_handler, NULL);
//    init_emergency_message();
#endif
}

void ApplicationProcess::contact_controller(int p, Host *h)
{

    //fprintf(stderr,"---- controller = new ControllerConnection\n");
    controller = new ControllerConnection(h, p, id, send_type);
    //fprintf(stderr,"---- controller = %x\n", controller);

    list_of_connections->add(controller);
}

void ApplicationProcess::handle_shm_msg(Message *msg)
{
    int tmpkey, size;

    tmpkey = ((int *)msg->data.data())[0];
    size = ((int *)msg->data.data())[1];
    shm->add_new_segment(tmpkey, size);
    //cerr << "new SharedMemory" << endl;
    print_comment(__LINE__, __FILE__, "new SharedMemory");
}

#if 0
coDistributedObject *ApplicationProcess::get_part_obj(char *pname)
{
   class coDistributedObject *pobj;

   part_obj_list->reset();
   pobj = part_obj_list->next();
   while(pobj)
   {
      if(strcmp(pobj->getName(), pname) == 0)
         return pobj;
   }
   return 0;
}
#endif

void UserInterface::contact_controller(int p, Host *h)
{
    controller = new ControllerConnection(h, p, id, send_type);
    list_of_connections->add(controller);
}
