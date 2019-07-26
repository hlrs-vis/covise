/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <io.h>
#include <time.h>
#include <fcntl.h>
#include <process.h>
#include <direct.h>
#else
#include <sys/time.h>
#include <sys/wait.h>
#endif

#include <sys/types.h>
#include <signal.h>

#include <covise/covise.h>
#include <do/coDistributedObject.h>
#include <net/covise_host.h>

#include "dmgr_packer.h"

#undef DEBUG

using namespace covise;

static const int EMPTY_VALUE = 0xfadefade;

#define FILL_SHM

//extern CoviseTime *covise_time;

//extern "C" int gethostname (char *name, int namelen);

/***********************************************************************\
 **                                                                     **
 **   Controller Class Routines                    Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The infrastructure for the Data Manager            **
 **                  environmentis provided here.                       **                                 **
 **                  This includes the distributed object               **
 **                  administration.                                    **
 **                                                                     **
 **   Classes      : DataManagerProcess                                 **
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
 **                  26.05.93  Ver 1.1   inclusio of new types,         **
 **                                      ObjectEntry functionality for  **
 **                                      local Object-database          **
 **                                                                     **
\***********************************************************************/
namespace covise
{

static void itoa(char *dest, int src)
{
    int i = 9, j;

    while (src > 0 && i >= 0)
    {
        dest[i] = '0' + src % 10;
        src = src / 10;
        i--;
    }
    if (i > -1)
    {
        i++;
        for (j = 0; j <= 9 - i; j++)
            dest[j] = dest[j + i];
        dest[j] = '\0';
    }
}
}

static int rngbuf_type = coDistributedObject::calcType("RNGBUF");

void DataManagerProcess::ask_for_object(Message *msg)
{
    ObjectEntry *oe;
    DMEntry *dm_ptr;
    char data[20];
    const char *addr;
    char tmp_str[255];

    //    cerr << "local ASK_FOR_OBJECT: " << msg->data << "\n";
    oe = get_local_object(msg->data);
    sprintf(tmp_str, "sending Object %s ++++++", msg->data);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
    msg->data = DataHandle();
    if (oe)
    {
        data_mgrs->reset();
        while ((dm_ptr = data_mgrs->next()))
            if ((Connection *)dm_ptr->conn == msg->conn)
                break;
        if (oe->type == rngbuf_type)
        {
            if (transfermanager == NULL)
                start_transfermanager();
            if (dm_ptr->transfermgr == 0)
            {
                Message trfmsg{ COVISE_MESSAGE_GET_TRANSFER_PORT, DataHandle() };
                exch_trf_msg(&trfmsg);
                *(int *)data = *(int *)trfmsg.data.data();
                if (*(int *)data == -1)
                {
                    print_comment(__LINE__, __FILE__,
                                  "transfer manager port not available");
                    print_exit(__LINE__, __FILE__, 1);
                }
                addr = host->getAddress();
                strncpy(&data[4], addr, 16);
                trfmsg.data = DataHandle(data, 4 + 1 + (int)strlen(addr));
                trfmsg.type = COVISE_MESSAGE_CONNECT_TRANSFERMANAGER;
                dm_ptr->conn->send_msg(&trfmsg);
                dm_ptr->transfermgr = 1;
            }
        }
        msg->type = COVISE_MESSAGE_OBJECT_FOLLOWS;
        msg->data = DataHandle();
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "ASK: vor OBJECT_FOLLOWS", 4);
#endif
        //	covise_time->mark(__LINE__, tmp_str);
        msg->conn->send_msg(msg);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "ASK: nach OBJECT_FOLLOWS", 4);
#endif
        //      covise_time->mark(__LINE__, "object will be packed now");
        oe->pack_and_send_object(msg, this);
        oe->add_access(msg->conn, ACC_REMOTE_DATA_MANAGER, ACC_READ_ONLY);
#ifdef DEBUG
//	print_comment(__LINE__, __FILE__, "vor dm_ptr->send_data_msg");
#endif
//        covise_time->mark(__LINE__, "ASK: vor OBJECT send");
//	dm_ptr->send_data_msg(msg);
//        covise_time->mark(__LINE__, "ASK: nach OBJECT send");
#ifdef DEBUG
//        print_comment(__LINE__, __FILE__, "nach dm_ptr->send_data_msg");
#endif
        msg->data = DataHandle();
    }
    else
    {
        msg->data = DataHandle();
        msg->type = COVISE_MESSAGE_OBJECT_NOT_FOUND;
        msg->conn->send_msg(msg);
    }
    //   covise_time->print();
}

Message *DataManagerProcess::wait_for_msg()
{
    return Process::wait_for_msg();
}

void DataManagerProcess::send_trf_msg(Message *msg)
{
#ifndef CRAY
    transfermanager->send_msg(msg);
#endif
}

void DataManagerProcess::exch_trf_msg(Message *msg)
{
#ifndef CRAY
    transfermanager->send_msg(msg);
    transfermanager->recv_msg(msg);
#endif
}

Message *DataManagerProcess::wait_for_msg(int covise_msg_type, Connection *conn = 0)
{
    Message *msg;
    int covise_msg_type_arr[3];

    covise_msg_type_arr[0] = covise_msg_type;
    covise_msg_type_arr[1] = COVISE_MESSAGE_ASK_FOR_OBJECT;
    covise_msg_type_arr[2] = COVISE_MESSAGE_HAS_OBJECT_CHANGED;

    while (1)
    {
        msg = Process::wait_for_msg(covise_msg_type_arr, 3, conn);
        switch (msg->type)
        {
        case COVISE_MESSAGE_ASK_FOR_OBJECT:
            ask_for_object(msg);
            delete msg;
            break;
        case COVISE_MESSAGE_HAS_OBJECT_CHANGED:
            has_object_changed(msg);
            delete msg;
            break;
        case COVISE_MESSAGE_GET_TRANSFER_PORT:
            has_object_changed(msg);
            delete msg;
            break;
        default:
            return msg;
        }
    }
}

Message *DataManagerProcess::wait_for_msg(int *covise_msg_type, int no,
                                          Connection *conn = 0)
{

    Message *msg;
    int *covise_msg_type_arr = new int[no + 2];
    int i;
#ifdef DEBUG
    char tmp_str[255];
#endif

    for (i = 0; i < no; i++)
        covise_msg_type_arr[i] = covise_msg_type[i];
    covise_msg_type_arr[i++] = COVISE_MESSAGE_ASK_FOR_OBJECT;
    covise_msg_type_arr[i++] = COVISE_MESSAGE_HAS_OBJECT_CHANGED;

    while (1)
    {
        msg = Process::wait_for_msg(covise_msg_type_arr, no + 2, (Connection *)NULL);
        switch (msg->type)
        {
        case COVISE_MESSAGE_ASK_FOR_OBJECT:
            ask_for_object(msg);
            delete msg;
            break;
        case COVISE_MESSAGE_HAS_OBJECT_CHANGED:
            has_object_changed(msg);
            delete msg;
            break;
        default:
            if (conn)
            {
                if (msg->conn == conn)
                {
                    delete[] covise_msg_type_arr;
                    return msg;
                }
                else
                {
                    msg_queue->add(msg);
#ifdef DEBUG
                    sprintf(tmp_str, "msg %s added to queue", covise_msg_types_array[msg->type]);
                    print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
                }
            }
            else
            {
                delete[] covise_msg_type_arr;
                return msg;
            }
        }
    }
}

void DataManagerProcess::has_object_changed(Message *msg)
{
    //    ObjectEntry *oe;

    //    cerr << "local HAS_OBJECT_CHANGED:\n";
    //    cerr << "dummy so far\n";
    msg->type = COVISE_MESSAGE_OBJECT_OK;

    msg->data = DataHandle();
    msg->conn->send_msg(msg);
}

namespace covise
{
int ObjectEntry_compare(ObjectEntry *a, ObjectEntry *b)
{
    //    cerr << "a: " << a->name << "   b: " << b->name << " -> ";
    char *an = a->name.accessData(), *bn = b->name.accessData();

    while (*an && (*an == *bn))
    {
        an++;
        bn++;
    }
    return (*an - *bn);

    //    int cmp = strcmp(a->name, b->name);
    //    cerr << cmp << endl;
    //#ifdef DEBUG
    //    char tmp_str[255];
    //    sprintf(tmp_str, "%x: %s   %x: %s -> %d", &a->name, a->name,
    //					      &b->name, b->name, cmp);
    //    print_comment(__LINE__, __FILE__, tmp_str,4);
    //#endif
    //    return cmp;
}

void clean_all(int arg)
{
    ((DataManagerProcess *)Process::this_process)->clean_all(arg);
}
}

void DataManagerProcess::clean_all(int)
{
    delete Process::this_process;
    print_comment(__LINE__, __FILE__, "signal called, DataManager cleaning up and exiting");

    // we: Calling print_exit here is a double delete !!
    //print_exit(__LINE__, __FILE__, 1);
    exit(-1);
}

static DataManagerProcess *dmgr_process;

#ifndef _WIN32
#ifdef COVISE_Signals
//=====================================================================
//
//=====================================================================
void dmgr_signal_handler(int sg, void *)
{
    int cpid, status;

    switch (sg)
    {
    case SIGCHLD:
        cpid = waitpid(-1, &status, WNOHANG);
        //    	cout << "waited for process no. " << cpid << endl;
        dmgr_process->remove_pid(cpid);
        while (cpid > 0)
        {
            cpid = waitpid(-1, &status, WNOHANG);
            //    	    cout << "waited for process no. " << cpid << " in loop\n";
            dmgr_process->remove_pid(cpid);
        }
        break;
    default:
        delete dmgr_process;
        sleep(1);
        exit(EXIT_FAILURE);
        break;
    }
}
#endif
#endif

DataManagerProcess::DataManagerProcess(char *name, int id, int *key)
    : OrdinaryProcess(name, id, DATAMANAGER)
{
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "in DataManagerProcess", 4);
#endif
    list_of_connections = new ConnectionList;
    tmpconn = NULL;
    transfermanager = NULL;
    pid_list = NULL;
    no_of_pids = 0;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "vor new coShmAlloc", 4);
#endif
#ifndef _SX
    signal(SIGINT, ::clean_all);
#endif
    shm = new coShmAlloc(key, this);
    //    objects = new AVLTree<ObjectEntry>();
    objects = new AVLTree<ObjectEntry>(ObjectEntry_compare, "objects");
    data_mgrs = new List<DMEntry>();
    msg_queue = new List<Message>();
    init_object_id();
    this_process = dmgr_process = this;
#ifdef COVISE_Signals
    // Initialization of signal handlers
    sig_handler.addSignal(SIGBUS, (void *)dmgr_signal_handler, NULL);
    sig_handler.addSignal(SIGPIPE, (void *)dmgr_signal_handler, NULL);
    sig_handler.addSignal(SIGTERM, (void *)dmgr_signal_handler, NULL);
    sig_handler.addSignal(SIGSEGV, (void *)dmgr_signal_handler, NULL);
    sig_handler.addSignal(SIGFPE, (void *)dmgr_signal_handler, NULL);
    sig_handler.addSignal(SIGCHLD, (void *)dmgr_signal_handler, NULL);
//    init_emergency_message();
#endif
}

/*
DataManagerProcess::DataManagerProcess(char *n, int arc, char *arv[])
    : OrdinaryProcess(n, arc, arv, DATAMANAGER) {
    if(arc != 5) {
   print_comment(__LINE__, __FILE__, "DataManager with inappropriate arguments called");
   print_exit(__LINE__, __FILE__, 1);
    }

    int port = atoi(arv[1]);
    Host *host = new Host(arv[2]);
    int id = atoi(arv[3]);
//char *instance = arv[4];
int key = 2000 + (id << 24);

print_comment(__LINE__, __FILE__, "in DataManagerProcess",4);
list_of_connections = new ConnectionList;
tmpconn = NULL;
transfermanager = NULL;
#ifndef _SX
signal(SIGINT, ::clean_all);
#endif
shm = new coShmAlloc(&key, this);
objects = new AVLTree<ObjectEntry>(ObjectEntry_compare, "objects");
//    objects = new AVLTree<ObjectEntry>();
data_mgrs = new List<DMEntry>();
msg_queue = new List<Message>();

contact_controller(port, host);
init_object_id();
this_process = this;

}

*/

void DataManagerProcess::contact_controller(int p, Host *h)
{
#if defined(CRAY) && !defined(_CRAYT3E)
    print_comment(__LINE__, __FILE__,
                  "contact_controller not necessary on machines w/o shared memory\n", 4);
    is_bad_ = false;
#else
    controller = new ControllerConnection(h, p, id, DATAMANAGER);
    if (controller->is_connected())
    {
        list_of_connections->add(controller);
        is_bad_ = false;
    }
    is_bad_ = true;
#endif
}

void DataManagerProcess::contact_datamanager(int p, Host *host)
{
    Connection *dm;
    char *msg_data = new char[80];
    DMEntry *dme = NULL;
    unsigned int sid;

    dm = new ClientConnection(host, p, id, DATAMANAGER);
    list_of_connections->add(dm);
#if defined(CRAY) && !defined(_WIN32)
#ifdef _CRAYT3E
    converter.int_to_exch(id, msg_data);
#else
    conv_single_int_c8i4(id, (int *)msg_data);
#endif
#else
    *((int *)msg_data) = id;
    swap_byte(*((unsigned int *)msg_data));
#endif
    strcpy(&msg_data[SIZEOF_IEEE_INT], get_host()->getAddress());
    int len = SIZEOF_IEEE_INT + (int)strlen(&msg_data[SIZEOF_IEEE_INT]) + 1;
    Message msg{ COVISE_MESSAGE_SEND_ID, DataHandle(msg_data, len) };

    dm->send_msg(&msg);
    msg.data = NULL;
    msg = *wait_for_msg(COVISE_MESSAGE_SEND_ID, dm);

    if (msg.type == COVISE_MESSAGE_SEND_ID)
    {
#if defined(CRAY) && !defined(_WIN32)
#ifdef _CRAYT3E
        converter.exch_to_int(msg_data, (int *)&sid);
#else
        conv_single_int_i4c8(*(int *)msg_data, &sid);
#endif
#else
        sid = *((int *)msg_data);
        swap_byte(sid);
#endif
        dme = new DMEntry(sid, &msg_data[SIZEOF_IEEE_INT], dm);
    }
    else
    {
        print_comment(__LINE__, __FILE__, "wrong message type");
    }
    data_mgrs->add(dme);
}

void DataManagerProcess::prepare_for_contact(int *p)
{
    tmpconn = new ServerConnection(p, id, DATAMANAGER);
    tmpconn->listen();
}

void DataManagerProcess::wait_for_contact()
{
    if (tmpconn == NULL)
    {
        print_comment(__LINE__, __FILE__,
                      "call prepare_for_contact before wait_for_contact in datamanager");
        print_exit(__LINE__, __FILE__, 1);
    }
    if (tmpconn->acceptOne() < 0)
    {
        print_comment(__LINE__, __FILE__,
                      "wait_for_contact in datamanager failed");
        print_exit(__LINE__, __FILE__, 1);
    }
    list_of_connections->add(tmpconn);
}

void DataManagerProcess::wait_for_dm_contact()
{
    Message *msg;
    char *msg_data = new char[80];
    DMEntry *dme;

    if (tmpconn == NULL)
    {
        print_comment(__LINE__, __FILE__,
                      "call prepare_for_contact before wait_for_contact in datamanager");
        print_exit(__LINE__, __FILE__, 1);
    }

    if (tmpconn->acceptOne() < 0)
    {
        print_comment(__LINE__, __FILE__,
                      "wait_for_dm_contact in datamanager failed");
        print_exit(__LINE__, __FILE__, 1);
    }
    list_of_connections->add(tmpconn);
    msg = wait_for_msg(COVISE_MESSAGE_SEND_ID, tmpconn);

    if (msg->type == COVISE_MESSAGE_SEND_ID)
    {
        dme = new DMEntry(*((int *)msg->data.data()), &msg->data.accessData()[SIZEOF_IEEE_INT], tmpconn);
        data_mgrs->add(dme);
#if defined(CRAY) && !defined(_WIN32)
#ifdef _CRAYT3E
        converter.int_to_exch(id, msg_data);
#else
        conv_single_int_c8i4(id, (int *)msg_data);
#endif
#else
        *((int *)msg_data) = id;
        swap_byte(*((unsigned int *)msg_data));
#endif
        strcpy(&msg_data[SIZEOF_IEEE_INT], get_host()->getAddress());
        msg->data = DataHandle(msg_data, SIZEOF_IEEE_INT + (int)strlen(&msg_data[SIZEOF_IEEE_INT]) + 1);

        msg->conn->send_msg(msg);
    }
    else
    {
        print_comment(__LINE__, __FILE__, "wrong message type");
    }
    delete msg;
}

void DataManagerProcess::start_transfermanager()
{
    char chport[10], *mdata;
    char chkey[10], chid[10], *thost;
    int port, tport;
    DMEntry *dmg;
    Message *msg;

    print_comment(__LINE__, __FILE__, "in start_transfermanager");
    transfermanager = new ServerConnection(&port, id, DATAMANAGER);
    transfermanager->listen();
#ifdef _WIN32

    itoa(chport, port);
    itoa(chkey, shm->get_key());
    itoa(chid, id + 1);
    cerr << "TransferManager: " << chport << " " << chkey << " " << chid << "\n";
    spawnlp(P_NOWAIT, "TransferManager", "TransferManager", chport, chkey, chid, NULL);
    cerr << "nach spawnlp in start_transfermanager";
    itoa(chport, port);
    itoa(chkey, shm->get_key());
    itoa(chid, id + 1);
    cerr << "TransferManager: " << chport << " " << chkey << " " << chid << "\n";
    if (0)
    {
        //
    }
#else
    if (fork() == 0)
    {
        execlp("TransferManager", "TransferManager", chport, chkey, chid, NULL);
        cerr << "nach execlp in start_transfermanager";
    }
#endif
    else
    {
        if (transfermanager->acceptOne() < 0)
        {
            fprintf(stderr, "DataManagerProcess: accept failed in start_transfermanager\n");
            return;
        }
        list_of_connections->add(transfermanager);
        msg = wait_for_msg(COVISE_MESSAGE_GET_SHM_KEY, transfermanager);
        handle_msg(msg);
        transfermanager->send_msg(msg);
        delete msg;
        data_mgrs->reset();
        while ((dmg = data_mgrs->next()))
        {
            msg = new Message(COVISE_MESSAGE_GET_TRANSFER_PORT,DataHandle());
            dmg->conn->send_msg(msg);
            delete msg;
            msg = wait_for_msg(COVISE_MESSAGE_TRANSFER_PORT, dmg->conn);
            tport = *(int *)msg->data.data();
            if (tport != -1)
            {
                thost = &msg->data.accessData()[SIZEOF_IEEE_INT];
                mdata = new char[sizeof(int) + strlen(thost) + 1];

#ifdef CRAY
// conv
#else
                *(int *)mdata = tport;
                strcpy(&mdata[sizeof(int)], &msg->data.data()[SIZEOF_IEEE_INT]);
                msg->data = DataHandle(mdata, msg->data.length());
#endif
                msg->type = COVISE_MESSAGE_CONNECT_TRANSFERMANAGER;
                send_trf_msg(msg);
            }
        }
    }
}

void DataManagerProcess::send_to_all_connections(Message *msg)
{
    Connection *conn;

    print_comment(__LINE__, __FILE__, "in send_to_all_connections");
    list_of_connections->reset();
    while ((conn = list_of_connections->next()))
    {
        conn->send_msg(msg);
    }
}

int DataManagerProcess::forward_new_part(Message *partmsg)
{
    DataHandle object_name, part_name;
    ObjectEntry *object, *part;
    AccessEntry *acc;

    object_name = part_name = partmsg->data;
    part_name.movePtr(strlen(object_name.data()) + 1);
    object = get_object(object_name);
    part = get_object(part_name);
    if (object && part == 0)
    {
        print_error(__LINE__, __FILE__, "no object for forward_new_part");
        return 0;
    }
    object->access->reset();
    while ((acc = object->access->next()))
    {
        if (acc->acc != ACC_REMOTE_DATA_MANAGER)
        {
            // partmsg->data still holds the object name
            partmsg->data.setLength(strlen(object_name.data()) + 1);
            partmsg->type = COVISE_MESSAGE_NEW_PART_AVAILABLE;
            partmsg->conn->send_msg(partmsg);
        }
        else
        {
            // partmsg->data and partmsg->len should still be ok here
            partmsg->type = COVISE_MESSAGE_NEW_PART_AVAILABLE;
            acc->conn->send_msg(partmsg);
            part->pack_and_send_object(partmsg, this);
            part->add_access(partmsg->conn, ACC_REMOTE_DATA_MANAGER, ACC_READ_ONLY);
            acc->conn->send_msg(partmsg);
        }
    }
    return 1;
}

int DataManagerProcess::receive_new_part(Message *msg)
{
    DataHandle object_name; //, *part_name;
    ObjectEntry *object, *part;
    AccessEntry *acc;
    Message *part_msg = new Message;
    DMEntry *dme;

    object_name = msg->data;
    //part_name = &msg->data[strlen(object_name) + 1];
    object = get_object(object_name);
    if (object == NULL)
    {
        print_error(__LINE__, __FILE__, "no object for receive_new_part");
        return 0;
    }
    data_mgrs->reset();
    dme = data_mgrs->next();
    while (dme && dme->conn != msg->conn)
        dme = data_mgrs->next();
    if (dme)
    {
        dme->conn->recv_msg(part_msg);
        part = create_object_from_msg(part_msg, dme);
        delete part_msg;
        if (insert_new_part(object, part))
        {
            object->access->reset();
            while ((acc = object->access->next()))
            {
                if (acc->acc != ACC_REMOTE_DATA_MANAGER)
                {
                    msg->data =object_name;
                    msg->type = COVISE_MESSAGE_NEW_PART_AVAILABLE;
                    msg->conn->send_msg(msg);
                }
            }
        }
        else
            return 0;
    }
    else
        return 0;

    return 1;
}

int DataManagerProcess::insert_new_part(ObjectEntry *object, ObjectEntry *part)
{
    int *iptr, *srciptr, *destiptr, i;
    coShmArray *shmarr, *shmptrarr;
    coShmPtr *shmptr;
    coDoHeader *header;

    shmarr = new coShmArray(object->shm_seq_no, object->offset);
    iptr = (int *)shmarr->getPtr();
    header = (coDoHeader *)iptr;
    header->increase_version();

    //  iptr[15];  // Object type
    //  iptr[17];  // Max number of parts
    //  iptr[19];  // Current number of parts

    shmptrarr = new coShmArray(iptr[21], iptr[22]);
    srciptr = (int *)shmptrarr->getDataPtr();
    if (iptr[19] >= iptr[17])
    {
        iptr[17] += STD_GROW_SIZE;
        shmptr = shm_alloc(SHMPTRARRAY, iptr[17]);
        destiptr = (int *)shmptr->getDataPtr();
        for (i = 0; i < iptr[17]; i += 2)
        {
            destiptr[2 * i + 0] = srciptr[2 * i + 0];
            destiptr[2 * i + 1] = srciptr[2 * i + 1];
        }
        shm_free(iptr[21], iptr[22]);
        iptr[21] = shmptr->shm_seq_no;
        iptr[22] = shmptr->offset;
    }
    srciptr[iptr[19] * 2] = part->shm_seq_no;
    srciptr[iptr[19] * 2 + 1] = part->offset;
    iptr[19]++;
    return 1;
}

coShmPtr *DataManagerProcess::shm_alloc(int type, shmSizeType msize)
{
    coShmPtr *chptr = NULL;
#ifdef DEBUG
    char tmpstr[255];
#endif
    int i;

    /// Collect size in this variable:

    shmSizeType size = sizeof(int); // all shm vars start with type

    //
    switch (type)
    {
    case FLOATSHM:
        size += sizeof(float);
        break;
    case DOUBLESHM:
        size += sizeof(double);
        break;
    case CHARSHM:
        size += sizeof(char);
        break;
    case SHORTSHM:
        size += sizeof(short);
        break;
    case LONGSHM:
        size += sizeof(long);
        break;
    case INTSHM:
        size += sizeof(int);
        break;
    case FLOATSHMARRAY:
        size += msize * sizeof(float) + 2 * sizeof(int);
        break;
    case DOUBLESHMARRAY:
        size += msize * sizeof(double) + 2 * sizeof(int);
        break;

    // All these add 2 ints: one for length and one for safety value.
    // the additional third int is for alignment whwnwver this may be doubtful
    case STRINGSHMARRAY:
        size += msize * 2 * sizeof(int) + 2 * sizeof(int);
        break;
    case CHARSHMARRAY:
        size += msize * sizeof(char) + 3 * sizeof(int);
        break;
    case SHORTSHMARRAY:
        size += msize * sizeof(short) + 3 * sizeof(int);
        break;
    case LONGSHMARRAY:
        size += msize * sizeof(long) + 3 * sizeof(int);
        break;
    case INTSHMARRAY:
        size += msize * sizeof(int) + 2 * sizeof(int);
        break;
    case SHMPTRARRAY:
        size += msize * 2 * sizeof(int) + 2 * sizeof(int);
        break;
    default:
        print_comment(__LINE__, __FILE__, "unkown type %d for shm_alloc\ncannot provide memory\n", type);
        break;
    };

    // only ose aligned memory sizes
    int alignRest = size % SIZEOF_ALIGNMENT;
    if (alignRest)
        size += (SIZEOF_ALIGNMENT - alignRest);

    // size in 'ints'
    int intSize = size / sizeof(int);

    // allocate memory
    chptr = shm->malloc(size);

    // int pointer to our memory
    int *iptr = (int *)chptr->getPtr();

    // type info
    iptr[0] = type;

    // other info special to type
    switch (type)
    {
    // fill memory with NULL
    case SHMPTRARRAY:
        iptr[1] = (int)msize;
        unsigned int ui;
        for (ui = 2; ui < (2 * msize) + 2; ui++)
        {
            iptr[ui] = 0;
        }
        iptr[intSize - 1] = type;
        break;

    case FLOATSHMARRAY:
    case DOUBLESHMARRAY:
    case STRINGSHMARRAY:
    case CHARSHMARRAY:
    case SHORTSHMARRAY:
    case LONGSHMARRAY:
    case INTSHMARRAY:
        iptr[1] = (int)msize;
#ifdef FILL_SHM
        for (i = 2; i < intSize; i++)
            iptr[i] = EMPTY_VALUE;
#endif
        iptr[intSize - 1] = type;
        break;
    };
#ifdef DEBUG
    sprintf(tmpstr, "DataManagerProcess::shm_alloc size: %d of type %d", size, type);
    print_comment(__LINE__, __FILE__, tmpstr, 8);
    sprintf(tmpstr, "at address %d, %d", chptr->get_shm_seq_no(), chptr->get_offset());
    print_comment(__LINE__, __FILE__, tmpstr, 8);
#endif

    return chptr;
}

coShmPtr *DataManagerProcess::shm_alloc(ShmMessage *shmmsg)
{

    data_type dt = shmmsg->get_data_type();
    unsigned long size = shmmsg->get_count();
    return shm_alloc(dt, size);
}

int DataManagerProcess::add_object(const DataHandle& n, int otype, int no, int o, Connection *conn)
{
    coDoHeader *header;
    coShmArray *shmarr;
    int h, t, *iptr;
#ifndef _WIN32
    struct timeval tp;
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "list of objects for Datamanager ************************", 7);
    if (COVISE_debug_level >= 7)
    {
        int tmp = COVISE_debug_level;
        COVISE_debug_level = 9;
        objects->print("Datamanager Object List");
        COVISE_debug_level = tmp;
    }
    print_comment(__LINE__, __FILE__, "******************* end of list ************************", 7);
#endif
    h = host->get_ipv4();
#ifdef _WIN32
    time_t long_time;
    struct tm *newtime;

    time(&long_time); /* Get time as long integer. */
    newtime = localtime(&long_time); /* Convert to local time. */

    if (newtime->tm_sec < max_t)
        t = ++max_t;
    else
        t = max_t = newtime->tm_sec;

#else
    gettimeofday(&tp, NULL);

    if (tp.tv_sec < max_t)
        t = ++max_t;
    else
        t = max_t = tp.tv_sec;
#endif
    shmarr = new coShmArray(no, o);
    iptr = (int *)shmarr->getPtr(); // pointer to the structure data
    header = (coDoHeader *)iptr;
    header->set_objectid(h, t);
    ObjectEntry *oe = new ObjectEntry(n, otype, no, o, conn);

    // the if here is only for security, the controller or the userinterface
    // have to take care, that no name appears twice.
    // this has to be improved to guarantee consistency over the whole
    // environment

    //    if(get_object(n, conn) == NULL)
    return objects->insert_node(oe);
}

void DataManagerProcess::init_object_id()
{
    int hdl, no;
#ifndef WIN32
    struct timeval tp;
#endif
    hdl = open("/tmp/covise_max_object_id", O_RDWR | O_CREAT, 0666);
    if (hdl <= 0)
        hdl = open("/covise_max_object_id", O_RDWR | O_CREAT, 0666);
    if (hdl >= 0)
    {
        no = read(hdl, &max_t, sizeof(int));
        if (no != sizeof(int))
        {
#ifdef _WIN32
            time_t long_time;
            struct tm *newtime;

            time(&long_time); /* Get time as long integer. */
            newtime = localtime(&long_time); /* Convert to local time. */
            max_t = newtime->tm_sec;

#else

            gettimeofday(&tp, NULL);

            max_t = tp.tv_sec;
#endif
            lseek(hdl, 0, SEEK_SET);
            ssize_t retval;
            retval = write(hdl, &max_t, sizeof(int));
            if (retval == -1)
            {
                std::cerr << "DataManagerProcess::init_object_id: write failed" << std::endl;
                return;
            }
        }
        close(hdl);
    }
    else
    {
#ifdef _WIN32
        time_t long_time;
        struct tm *newtime;

        time(&long_time); /* Get time as long integer. */
        newtime = localtime(&long_time); /* Convert to local time. */
        max_t = newtime->tm_sec;

#else

        gettimeofday(&tp, NULL);

        max_t = tp.tv_sec;
#endif
    }
}

void DataManagerProcess::save_object_id()
{
    int hdl;

    hdl = open("/tmp/covise_max_object_id", O_RDWR, 0666);
    if (hdl <= 0)
        hdl = open("/covise_max_object_id", O_RDWR, 0666);
    if (hdl > 0)
    {
        ssize_t retval;
        retval = write(hdl, &max_t, sizeof(int));
        if (retval == -1)
        {
            std::cerr << "DataManagerProcess::save_object_id: write failed" << std::endl;
            return;
        }
        close(hdl);
    }
}

int DataManagerProcess::add_object(const DataHandle &n, int no, int o, Connection *conn)
{
    coDoHeader *header;
    coShmArray *shmarr;
    int h, t, *iptr;
#ifndef _WIN32
    struct timeval tp;
#endif

#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "list of objects for Datamanager ************************", 7);
    if (COVISE_debug_level >= 7)
    {
        int tmp = COVISE_debug_level;
        COVISE_debug_level = 9;
        objects->print("Datamanager Object List");
        COVISE_debug_level = tmp;
    }
    print_comment(__LINE__, __FILE__, "******************* end of list ************************", 7);
#endif

    h = host->get_ipv4();
#ifdef _WIN32
    time_t long_time;
    struct tm *newtime;

    time(&long_time); /* Get time as long integer. */
    newtime = localtime(&long_time); /* Convert to local time. */

    if (newtime->tm_sec < max_t)
        t = ++max_t;
    else
        t = max_t = newtime->tm_sec;

#else

    gettimeofday(&tp, NULL);

    if (tp.tv_sec < max_t)
        t = ++max_t;
    else
        t = max_t = tp.tv_sec;
#endif
    shmarr = new coShmArray(no, o);
    iptr = (int *)shmarr->getPtr(); // pointer to the structure data
    header = (coDoHeader *)iptr;
    header->set_objectid(h, t);
    ObjectEntry *oe = new ObjectEntry(n, no, o, conn);

    // the if here is only for security, the controller or the userinterface
    // have to take care, that no name appears twice.
    // this has to be improved to guarantee consistency over the whole
    // environment

    if (get_object(n, conn) == NULL)
        return objects->insert_node(oe);
    else
        return 0;
}

int DataManagerProcess::add_object(ObjectEntry *oe)
{
    return objects->insert_node(oe);
}

int DataManagerProcess::delete_object(const DataHandle& n)
{
#ifdef DEBUG
    char tmp_str[255];
#endif

    ObjectEntry *oe = new ObjectEntry(n);
#ifdef DEBUG
    sprintf(tmp_str, "Removing object %s from list", n);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
    int retval = (objects->remove_node(oe) != NULL);
    delete oe;
#ifdef DEBUG
    if (retval == 1)
    {
        print_comment(__LINE__, __FILE__, "object successfully removed", 4);
    }
    else
    {
        print_comment(__LINE__, __FILE__, "object removal failed", 4);
    }
#endif
    return retval;
}

int DataManagerProcess::DTM_new_desk(void)
{

    if (objects)
        objects->empty_tree();
    if (shm)
        shm->new_desk();

    return 1;
}

int DataManagerProcess::rmv_rdmgr(char *hostname)
{
    DMEntry *dme;
    Connection *p_conn;
    CO_AVL_Node<ObjectEntry> *p_root;
    data_mgrs->reset();
    while ((dme = data_mgrs->next()))
    {
        if (strcmp(hostname, dme->get_hostname()) == 0)
        {
            //cerr << endl << "---CRB: rmv_rdmgr(" << hostname << ")\n";
            data_mgrs->remove(dme);
            p_conn = dme->get_conn();
            if (p_conn)
            {
                p_root = objects->get_root();
                if (p_root)
                    rmv_acc2objs(p_root, p_conn);
                //else cerr << endl << "---CRB : root node of the objects is null !!!\n";
            }
            else
                cerr << endl << "---CRB : rdmgr connection null !!!\n";

            delete dme;
            return 1;
        }
    }

    return 0;
}

void DataManagerProcess::rmv_acc2objs(CO_AVL_Node<ObjectEntry> *root, Connection *conn)
{
    ObjectEntry *p_data;
    p_data = root->data;
    if (p_data)
        p_data->remove_access(conn);
    else
        cerr << endl << "---CRB : ObjectEntry with no data !!!\n";
    if (root->left)
        rmv_acc2objs(root->left, conn);
    if (root->right)
        rmv_acc2objs(root->right, conn);
}

int DataManagerProcess::destroy_object(const DataHandle& n, Connection* c)
{
    AccessEntry *ae;
    ObjectEntry *oe = NULL, *tmpoe = new ObjectEntry(n);
    int covise_msg_types[2];

#ifdef DEBUG
    int tmpval;
    print_comment(__LINE__, __FILE__, "list of objects for Datamanager ************************", 6);
    if (COVISE_debug_level >= 6)
    {
        int tmp = COVISE_debug_level;
        COVISE_debug_level = 9;
        objects->print("Datamanager Object List");
        COVISE_debug_level = tmp;
    }
    print_comment(__LINE__, __FILE__, "******************* end of list ************************", 6);
#endif
    oe = objects->search_node(tmpoe, COVISE_EQUAL);
    if (!oe)
    {
        delete tmpoe;
        return 0;
    }
    if (c)
    {
        //	cerr << "in destroy_object for conn <> NULL\n";
        if (oe->get_access_right(c) == ACC_READ_WRITE_DESTROY)
        {
            oe->access->reset();
            while ((ae = oe->access->next()))
            {
                if (ae->acc == ACC_REMOTE_DATA_MANAGER)
                {
                    //	    	    cerr << "destroy object " << oe->name << " on remote dmgr\n";
                    Message msg{ COVISE_MESSAGE_CTRL_DESTROY_OBJECT, n };
                    ae->conn->send_msg(&msg);

                    covise_msg_types[0] = COVISE_MESSAGE_MSG_OK;
                    covise_msg_types[1] = COVISE_MESSAGE_MSG_FAILED;
                    msg = *wait_for_msg(covise_msg_types, 2, ae->conn);
                    if (msg.type == COVISE_MESSAGE_MSG_FAILED)
                    {
                    }
                }
            }
#ifndef DEBUG
            objects->remove_node(oe);
#else
            // is done in shm_free recursively
            tmpval = (objects->remove_node(oe) != NULL);
            if (tmpval == 1)
            {
                print_comment(__LINE__, __FILE__, "object successfully removed", 4);
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object removal failed", 4);
            }
#endif
            shm_free(oe->shm_seq_no, oe->offset);
            delete oe;
            delete tmpoe;
            return 1;
        }
        else
        {
            delete tmpoe;
            return 0;
        }
    }
    else
    {
        //	cerr << "in destroy_object for conn == NULL\n";
        oe->access->reset();
        while ((ae = oe->access->next()))
        {
            if (ae->acc == ACC_REMOTE_DATA_MANAGER)
            {
#ifdef DEBUG
                sprintf(tmp_str, "destroy object %s for remote dmgr", oe->name);
                print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
                Message msg{ COVISE_MESSAGE_CTRL_DESTROY_OBJECT, n };
                ae->conn->send_msg(&msg);
                covise_msg_types[0] = COVISE_MESSAGE_MSG_OK;
                covise_msg_types[1] = COVISE_MESSAGE_MSG_FAILED;
                msg = *wait_for_msg(covise_msg_types, 2, ae->conn);
                if (msg.type == COVISE_MESSAGE_MSG_FAILED)
                {
                }
            }
        }
#ifdef DEBUG
        sprintf(tmp_str, "Removing object %s from list", oe->name);
        print_comment(__LINE__, __FILE__, tmp_str, 4);
        // is done in shm_free recursively
        tmpval = (objects->remove_node(oe) != NULL);
#else
        objects->remove_node(oe); // is done in shm_free recursively
#endif

#ifdef DEBUG
        if (tmpval == 1)
        {
            print_comment(__LINE__, __FILE__, "object successfully removed", 4);
        }
        else
        {
            print_comment(__LINE__, __FILE__, "object removal failed", 4);
        }
#endif
        shm_free(oe->shm_seq_no, oe->offset);
        delete oe;
        delete tmpoe;
        return 1;
    }
}

int DataManagerProcess::shm_free(coShmPtr *ptr)
{
    return shm_free(ptr->get_shm_seq_no(), ptr->get_offset());
}

int DataManagerProcess::shm_free(int shm_seq_no, int offset)
{
    int *objptr, incr;
    coCharShmArray *chshmarr;
    char *shmptr = NULL, *obj_name = NULL;
    char tmp_str[255];
    int type, count, i, no;

    if (shm_seq_no == 0)
        return 1;
    shmptr = (char *)shm->get_pointer(shm_seq_no);
    objptr = (int *)(shmptr + offset);
    type = objptr[0];
    switch (type)
    {
    // if the (shm_seq_no, offset) points directly to an allocated item:
    case CHARSHM:
    case SHORTSHM:
    case INTSHM:
    case LONGSHM:
    case FLOATSHM:
    case DOUBLESHM:
    case CHARSHMARRAY:
    case SHORTSHMARRAY:
    case INTSHMARRAY:
    case LONGSHMARRAY:
    case FLOATSHMARRAY:
    case DOUBLESHMARRAY:
    case COVISE_NULLPTR:
        shm->free(shm_seq_no, offset);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "direct free in shm_free");
#endif
        return 1;
    // if it points to a pointer, first the item the pointer points to gets freed:
    case SHMPTR:
//    case CHARSHMPTR:
//    case SHORTSHMPTR:
//    case INTSHMPTR:
//    case LONGSHMPTR:
//    case FLOATSHMPTR:
//    case DOUBLESHMPTR:
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "ptr free in shm_free");
#endif
        if (shm_free(objptr[1], objptr[2]))
        {
            shm->free(shm_seq_no, offset);
            return 1;
        }
        else
        {
            print_comment(__LINE__, __FILE__, "couldn't free correctly");
            return 0;
        }
    case STRINGSHMARRAY:
        no = objptr[1];
        for (i = 0; i < no; i++)
            if (objptr[2 + 2 * i] != 0)
                shm_free(objptr[2 + 2 * i], objptr[2 + 2 * i + 1]);
        shm->free(shm_seq_no, offset);
        return 1;
    case SHMPTRARRAY:
        no = objptr[1];
        for (i = 0; i < no; i++)
            shm_free(objptr[2 + 2 * i], objptr[2 + 2 * i + 1]);
        shm->free(shm_seq_no, offset);
        return 1;
    }
    // here we have a pointer to a non basic data type:
    objptr[10]--; // decrease reference counter
    if (objptr[10] != 0) // if object is referenced by other objects
        return 1; // then we are finished

    int length = objptr[1]; // length in char
    if (objptr[11] == SHMPTR) // get the name of this object
    {
        chshmarr = new coCharShmArray(objptr[12], objptr[13]);
        obj_name = (char *)chshmarr->getDataPtr();
        delete chshmarr;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "now freeing object %s", obj_name);
#endif
    }
    objptr += 2; // jump over type and length
// now we process all types that are stored here together
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "go through non simple type in shm_free", 4);
#endif
    for (count = 2 * sizeof(int); count < length;)
    {
#ifdef DEBUG
        sprintf(tmp_str, "count < length: %d < %d", count, length);
        print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
        switch (*objptr)
        {
        case CHARSHM: // we can mix all these, since they have
        case SHORTSHM: // sizeof <= sizeof(int), which is the
        case INTSHM: // minimum length due to alignment
        case FLOATSHM:
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "scalar part of shm_free", 4);
#endif
            objptr += 2;
            count += 2 * sizeof(int);
            break;
        case LONGSHM:
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "scalar part of shm_free", 4);
#endif
#ifdef CRAY
            objptr += 2;
            incr = 2;
#else
            incr = 1 + sizeof(long) / sizeof(int) + (sizeof(long) % sizeof(int) != 0);
            objptr += incr;
#endif
            count += incr * sizeof(int);
            break;
        case DOUBLESHM:
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "scalar part of shm_free");
#endif
#ifdef CRAY
            objptr += 2;
            incr = 2;
#else
            incr = 1 + sizeof(double) / sizeof(int) + (sizeof(double) % sizeof(int) != 0);
            objptr += incr;
#endif
            count += incr * sizeof(int);
            break;
        case COVISE_NULLPTR:
        case COVISE_OBJECTID:
            objptr += 3;
            count += 3 * sizeof(int);
            break;
        case SHMPTR:
//	case CHARSHMPTR:
//	case SHORTSHMPTR:
//	case INTSHMPTR:
//	case LONGSHMPTR:
//	case FLOATSHMPTR:
//	case DOUBLESHMPTR:
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "ptr part of shm_free");
#endif
            if (shm_free(objptr[1], objptr[2]))
            {
                objptr += 3;
                count += 3 * sizeof(int);
            }
            else
            {
                print_comment(__LINE__, __FILE__, "couldn't free correctly");
                return 0;
            }
            break;
        default:
            sprintf(tmp_str, "couldn't handle type %d in shm_free", *objptr);
            print_comment(__LINE__, __FILE__, tmp_str, 4);
            return 0;
        }
    }
    
    ObjectEntry* tmpoe = new ObjectEntry{ DataHandle{obj_name, strlen(obj_name) + 1, false} };
#ifdef DEBUG
    sprintf(tmp_str, "Removing object %s from list", obj_name);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
    if (objects->remove_node_compare(tmpoe))
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "object successfully removed");
    }
    else
    {
        print_comment(__LINE__, __FILE__, "object removal failed");
#endif
    }
    delete tmpoe;
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "object list ---------------------------");
    int tmp = COVISE_debug_level;
    COVISE_debug_level = 9;
    objects->print("Datamanager Object List");
    COVISE_debug_level = tmp;
    print_comment(__LINE__, __FILE__, "---------------------------------------");
#endif
    shm->free(shm_seq_no, offset);
#ifdef DEBUG
    print_comment(__LINE__, __FILE__, "end of shm_free");
#endif
    return 1;
}

DataHandle DataManagerProcess::get_all_hosts_for_object(const DataHandle& n)
{
    ObjectEntry *oe = get_object(n);
    DMEntry *tmpdmgr;
    AccessEntry *ae;
    Message *msg;
    char *data, *tmpdata;
    const char *hostname;
    size_t length;

    if (oe)
    {
        print_comment(__LINE__, __FILE__, "object %s found on local host", n.data());
        if (oe->dmgr)
        {
            msg = new Message(COVISE_MESSAGE_OBJECT_ON_HOSTS, n);
            oe->dmgr->send_data_msg(msg);
            oe->dmgr->recv_data_msg(msg);
            if (msg->type == COVISE_MESSAGE_OBJECT_ON_HOSTS)
            {
                print_comment(__LINE__, __FILE__, "object was created on other host and found there");
                DataHandle dh = msg->data;
                delete msg;
                return dh;
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object was created on other host but not found there");
                DataHandle dh{ strlen(get_host()->getAddress()) + 1 + 2 };
                dh.accessData()[0] = '?';
                dh.accessData()[1] = ' ';
                strcpy(&dh.accessData()[2], get_host()->getAddress());
                delete msg;
                return dh;
            }
        }
        else
        {
            print_comment(__LINE__, __FILE__, "object was created on local host");
            data = new char[strlen(get_host()->getAddress()) + 1];
            strcpy(data, get_host()->getAddress());
            oe->access->reset();
            ae = oe->access->next();
            while (ae)
            {
                if (ae->acc == ACC_REMOTE_DATA_MANAGER)
                {
                    hostname = ae->conn->get_hostname();
                    print_comment(__LINE__, __FILE__, "object has been copied to %s", hostname);
                    length = strlen(data);
                    tmpdata = new char[length + 1 + strlen(hostname) + 2];
                    strcpy(tmpdata, data);
                    tmpdata[length] = ',';
                    tmpdata[length + 1] = ' ';
                    strcpy(&tmpdata[length + 2], hostname);
                    delete[] data;
                    data = tmpdata;
                }
                ae = oe->access->next();
            }
            return DataHandle(data, strlen(data) + 1);
        }
    }
    else
    {
        print_comment(__LINE__, __FILE__, "object %s not found on local host", n.data());
        data_mgrs->reset();
        tmpdmgr = data_mgrs->next();
        while (tmpdmgr)
        {
            print_comment(__LINE__, __FILE__, "asking other host");
            msg = new Message(COVISE_MESSAGE_OBJECT_ON_HOSTS, n);
            tmpdmgr->send_data_msg(msg);
            tmpdmgr->recv_data_msg(msg);
            if (msg->type == COVISE_MESSAGE_OBJECT_FOUND)
            {
                print_comment(__LINE__, __FILE__, "object found on other host");
                DataHandle dh = msg->data;
                delete msg;
                return dh;
            }
            tmpdmgr = data_mgrs->next();
        }
        print_comment(__LINE__, __FILE__, "returning NULL");
        return NULL;
    }
}

ObjectEntry *DataManagerProcess::get_object(const DataHandle& n, Connection *conn)
{
    ObjectEntry *oe = get_object(n);

    if (oe)
    {
        switch (oe->get_access_right(conn))
        {
        case ACC_NONE:
            oe->add_access(conn, ACC_READ_ONLY, ACC_READ_ONLY);
            break;
        case ACC_DENIED:
            return NULL;
        default:
            oe->set_current_access(conn, ACC_READ_ONLY);
            break;
        }
    }
    return oe;
}

ObjectEntry *DataManagerProcess::get_object(const DataHandle &n)
{
    ObjectEntry *oe;
    ObjectEntry *tmpoe = new ObjectEntry(n);
    DMEntry *dme;
    size_t len;
    char *tmp_name;
    char tmp_str[255];
    char *tmp_str_ptr;
    int covise_msg_type_arr[2];
    Message *data_msg;

    tmp_name = new char[n.length() + sizeof(int)];
    strcpy(tmp_name, n.data());
#ifdef DEBUG
    sprintf(tmp_str, "in get_object: %s", tmp_name);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
    //    objects->print();
    oe = objects->search_node(tmpoe, COVISE_EQUAL);
    delete tmpoe;
    if (oe == NULL)
    {
#ifdef DEBUG
//	print_comment(__LINE__, __FILE__, "oe == NULL",4);
#endif
        data_mgrs->reset();
        int found = 0;
        while (!found && (dme = data_mgrs->next()))
        {
            len = strlen(tmp_name) + 1;
            Message *msg = new Message(COVISE_MESSAGE_ASK_FOR_OBJECT, DataHandle(tmp_name, len));
            tmp_str_ptr = new char[100];
            sprintf(tmp_str_ptr, "GET: asking for object %s ", tmp_name);
            //	    covise_time->mark(__LINE__, tmp_str_ptr);
            dme->conn->send_msg(msg);
            delete msg;
            covise_msg_type_arr[0] = COVISE_MESSAGE_OBJECT_FOLLOWS;
            covise_msg_type_arr[1] = COVISE_MESSAGE_OBJECT_NOT_FOUND;
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "vor wait_for_msg", 4);
#endif
            msg = wait_for_msg(covise_msg_type_arr, 2, dme->conn);
            switch (msg->type)
            {
            case COVISE_MESSAGE_OBJECT_FOLLOWS:
#ifdef DEBUG
                sprintf(tmp_str, "%s found at remote datamanager",
                        tmp_name);
                print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
                data_msg = new Message;
                dme->recv_data_msg(data_msg);
                //                covise_time->mark(__LINE__, "GET: object received");
                oe = create_object_from_msg(data_msg, dme);
                delete data_msg;
                add_object(oe);
                found = 1;
                break;
            case COVISE_MESSAGE_OBJECT_NOT_FOUND:
                //               covise_time->mark(__LINE__, "GET: object not found");
                sprintf(tmp_str, "%s not found at remote datamanager",
                        tmp_name);
                print_comment(__LINE__, __FILE__, tmp_str, 4);
                break;
            default:
                print_comment(__LINE__, __FILE__, "wrong message type");
                break;
            }
            delete msg;
        }
    }
    else
    {
//   	cerr << "Dmgr get_object: \"" << oe->name << "\" " <<
//            	oe->shm_seq_no << " " << oe->offset << endl;
#ifdef DEBUG
//	print_comment(__LINE__, __FILE__, "oe != NULL",4);
#endif
        if (oe->dmgr != NULL)
        {
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "oe->dmgr != NULL", 4);
#endif
            char *data = new char[strlen(oe->name.data()) + 1 + sizeof(int)];
#if defined(CRAY) && !defined(_WIN32)
#ifdef _CRAYT3E
            converter.int_to_exch(oe->version, data);
#else
            conv_single_int_c8i4(oe->version, (int *)data);
#endif
#else
            *(int *)data = oe->version;
            swap_byte(*(unsigned int *)data);
#endif
            strcpy(&data[SIZEOF_IEEE_INT], oe->name.data());
            len = n.length() + SIZEOF_IEEE_INT;

            // Objects never change, so forget about this for now
            /*
          Message *msg = new Message(HAS_OBJECT_CHANGED, len, data);
          oe->dmgr->conn->send_msg(msg);
          delete [] msg->data;
          msg->data = NULL;
          covise_msg_type_arr[0] = OBJECT_UPDATE;
          covise_msg_type_arr[1] = OBJECT_OK;
          msg = wait_for_msg(covise_msg_type_arr, 2, oe->dmgr->conn);
          switch(msg->type) {
          case OBJECT_UPDATE:
         //		cerr << oe->name << " updated\n";
         update_object_from_msg(msg, dme);
         break;
         case OBJECT_OK:
         //		cerr << oe->name << " not updated\n";
         break;
         }
         delete [] msg->data;
         msg->data = NULL;
         delete msg;
         */
        }
    }
    delete[] tmp_name;
#ifdef DEBUG
    char *tmp_ptr;
    if (oe)
        tmp_ptr = oe->name;
    else
        tmp_ptr = "";
    sprintf(tmp_str, "oe == %x returned: %s", oe, tmp_ptr);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
#endif
    return oe;
}

ObjectEntry *DataManagerProcess::get_local_object(const DataHandle &n)
{
    ObjectEntry *oe;
    ObjectEntry *tmpoe = new ObjectEntry(n);

    //    cerr << "in get_local_object: " << n << "\n";
    //    objects->print();
    oe = objects->search_node(tmpoe, COVISE_EQUAL);
    //    cerr << "search result: " << oe << endl;;
    delete tmpoe;
    return oe;
}

extern coShmPtr *covise_extract_list(List<PackElement> *, char);
extern int covise_decode_list(List<PackElement> *, char *,
                              DataManagerProcess *, char);

ObjectEntry *DataManagerProcess::create_object_from_msg(Message *msg, DMEntry *dme)
{
    //    cerr << "in create_object_from_msg\n";
    ObjectEntry *oe;
    //List<PackElement> *pack_list = new List<PackElement>;
    coShmPtr *shm_ptr;
    char *tmp_name;
    Packer *pack_object;

    //    print_comment(__LINE__, __FILE__, "vor: pack_object = new Packer(msg, this);");

    pack_object = new Packer(msg, this);

    shm_ptr = pack_object->unpack(&tmp_name);

    oe = new ObjectEntry(DataHandle(tmp_name, strlen(tmp_name) + 1), shm_ptr->shm_seq_no, shm_ptr->offset,
                         msg->conn, dme);

    delete pack_object;
    return oe;
}

int DataManagerProcess::update_object_from_msg(Message *, DMEntry *)
{
    //    cerr << "in update_object_from_msg\n";
    return 1;
}

int DataManagerProcess::make_data_connection(char *name, char *new_interface)
{
    DMEntry *dm_ptr;
    Host *tmphost;
    data_mgrs->reset();
    while ((dm_ptr = data_mgrs->next()))
    {
        tmphost = new Host(name);
        if (strcmp(dm_ptr->host->getAddress(), tmphost->getAddress()) == 0)
        {
            return dm_ptr->make_data_conn(new_interface);
        }
    }
    return 0;
}

int DataManagerProcess::complete_data_connection(Message *msg)
{
    DMEntry *dm_ptr;

    data_mgrs->reset();
    while ((dm_ptr = data_mgrs->next()))
    {
        if ((Connection *)dm_ptr->conn == msg->conn)
        {
            return dm_ptr->complete_data_conn(msg);
        }
    }
    return 0;
}

ObjectEntry::ObjectEntry(const DataHandle& n)
{
    //    printf("ObjectEntry new   : %x\n", this);
    name = n;
    shm_seq_no = offset = -1;
    version = 1;
    dmgr = NULL;
    type = 0;
    access = new List<AccessEntry>;
}

ObjectEntry::ObjectEntry(const DataHandle &n, int otype, int no, int o, Connection *conn, DMEntry *dm)
{
    coShmPtr *tmpptr;

    //    printf("ObjectEntry new   : %x\n", this);
    name = n;
    shm_seq_no = no;
    offset = o;
    tmpptr = new coShmPtr(no, o);
    type = otype;
    delete tmpptr;
    version = 1;
    dmgr = dm;
    owner = conn;
    access = new List<AccessEntry>;
    add_access(owner, ACC_READ_WRITE_DESTROY, ACC_READ_AND_WRITE);
}

ObjectEntry::ObjectEntry(const DataHandle& n, int no, int o, Connection *conn, DMEntry *dm)
{
    coShmPtr *tmpptr;

    //    printf("ObjectEntry new   : %x\n", this);
    name = n;
    shm_seq_no = no;
    offset = o;
    tmpptr = new coShmPtr(no, o);
    type = tmpptr->get_type();
    delete tmpptr;
    version = 1;
    dmgr = dm;
    owner = conn;
    access = new List<AccessEntry>;
    add_access(owner, ACC_READ_WRITE_DESTROY, ACC_READ_AND_WRITE);
}

ObjectEntry::~ObjectEntry()
{
    AccessEntry *ae;

    access->reset();
    while ((ae = access->next()))
    {
        delete ae;
    }
    //    printf("ObjectEntry delete: %x\n", this);
}

void ObjectEntry::print()
{
    coShmArray *tmparr;
    int *tmpiptr;
    char tmp_str[255];
    char *tmpname;

    if (shm_seq_no == -1 && offset == -1)
        return;
    tmparr = new coShmArray(shm_seq_no, offset);
    tmpiptr = (int *)tmparr->getPtr();
    tmpname = coDistributedObject::calcTypeString(tmpiptr[0]);

    sprintf(tmp_str, "(%d, %8d) %s of type %s", shm_seq_no, offset, name.data(), tmpname);
    print_comment(__LINE__, __FILE__, tmp_str, 4);
    sprintf(tmp_str, "Length: %d, Version: %d, Refcount: %d", tmpiptr[6], tmpiptr[8], tmpiptr[10]);
    print_comment(__LINE__, __FILE__, tmp_str, 4);

    //    cerr << " (" << shm_seq_no << "," << setw(4)
    //	 << offset << ") " << name;
    //    cerr << "  Type:     " << coDistributedObject::calcTypeString(tmpiptr[0]) << endl;
    //    cerr << "     Length:   " << tmpiptr[6];
    //    cerr << "     Version:  " << tmpiptr[8];
    //    cerr << "     Refcount: " << tmpiptr[10] << endl;
}

void ObjectEntry::remove_access(Connection *c)
{
    AccessEntry *ae;
    Message *msg;

    access->reset();
    while ((ae = access->next()))
        if (ae->conn == c)
        {
            access->remove(ae);
#ifdef DEBUG
            print_comment(__LINE__, __FILE__, "removing access", 4);
#endif
            delete ae;
            break;
        }
    if (dmgr)
    {
        access->reset();
        if (access->current() == NULL)
        {
            msg = new Message(COVISE_MESSAGE_OBJECT_NO_LONGER_USED, name);
            dmgr->conn->send_msg(msg);
            msg->data = NULL;
            delete msg;
        }
    }
}

void ObjectEntry::set_current_access(Connection *c, access_type c_a)
{
    AccessEntry *ae;
    Message *msg;

    access->reset();
    while ((ae = access->next()))
        if (ae->conn == c)
        {
            switch (c_a)
            {
            case ACC_NONE:
                ae->curr_acc = ACC_NONE;
                break;
            case ACC_READ_ONLY:
                if (ae->acc != ACC_DENIED)
                    ae->curr_acc = ACC_READ_ONLY;
                else
                    print_comment(__LINE__, __FILE__, "No access permitted");
                break;
            case ACC_WRITE_ONLY:
                if ((ae->acc != ACC_DENIED) && (ae->acc != ACC_READ_ONLY))
                {
                    ae->curr_acc = ACC_WRITE_ONLY;
                    msg = new Message(COVISE_MESSAGE_MSG_OK, DataHandle());
                    ae->conn->send_msg(msg);
                    delete msg;
                }
                else
                {
                    msg = new Message(COVISE_MESSAGE_MSG_FAILED, DataHandle());
                    ae->conn->send_msg(msg);
                    delete msg;
                }
                break;
            case ACC_READ_AND_WRITE:
                if ((ae->acc != ACC_DENIED) && (ae->acc != ACC_READ_ONLY) && (ae->acc != ACC_WRITE_ONLY))
                {
                    ae->curr_acc = ACC_READ_AND_WRITE;
                    msg = new Message(COVISE_MESSAGE_MSG_OK, DataHandle());
                    ae->conn->send_msg(msg);
                    delete msg;
                }
                else
                {
                    msg = new Message(COVISE_MESSAGE_MSG_FAILED, DataHandle());
                    ae->conn->send_msg(msg);
                    delete msg;
                }
                break;
            default:
                print_comment(__LINE__, __FILE__, "No such access possible");
                break;
            }
        }
}

access_type ObjectEntry::get_access_right(Connection *c)
{
    AccessEntry *ptr;

    access->reset();
    while ((ptr = access->next()))
    {
        if (ptr->conn == c)
            return ptr->acc;
    }
    return ACC_NONE;
}

void ObjectEntry::add_access(Connection *c, access_type a, access_type c_a)
{
    AccessEntry *ae = new AccessEntry(a, c_a, c);
    access->add(ae);
}

extern char *covise_encode_list(List<PackElement> *pack_list, int size, char convert);
extern void covise_create_list(List<PackElement> *pack_list, coShmAlloc *shm,
                               int shm_seq_no, int offset, int *size, char convert);

void ObjectEntry::pack_and_send_object(Message *msg, DataManagerProcess *)
{
    //    cerr << "in pack_object for " << name << endl;
    //    List<PackElement> *pack_list = new List<PackElement>;
    int size = 0;
    Packer *pack_object;

    //    covise_time->mark(__LINE__, "vor pack_object = new Packer");

    pack_object = new Packer(msg, shm_seq_no, offset);

    pack_object->pack();

    pack_object->flush();

    delete pack_object;

//    covise_time->mark(__LINE__, "packed object sent");

//    covise_time->mark(__LINE__, "vor covise_create_list");
//    covise_create_list(pack_list, dm->shm, shm_seq_no, offset, &size, msg->conn->convert_to);
//    covise_time->mark(__LINE__, "vor covise_create_list");
//    print_comment(__LINE__, __FILE__, "---------- the complete packlist -------------");
//    pack_list->print();
//    print_comment(__LINE__, __FILE__, "------- end of the complete packlist ---------");
//    msg->data = covise_encode_list(pack_list, size, msg->conn->convert_to);
//    covise_time->mark(__LINE__, "nach covise_encode_list");
#ifdef DEBUG
/*
       char tmp_str[255];
       int i;
       for(i = 0;i < size;i++) {
      sprintf(tmp_str, "data[%3d] = %d", i, (int)msg->data[i]);
      print_comment(__LINE__, __FILE__, tmp_str,4);
       }
   */
#endif
    msg->type = COVISE_MESSAGE_OBJECT_FOLLOWS;
    msg->data.setLength(size);
    //    delete pack_list;
}

void ObjectEntry::pack_address(Message *msg)
{
    //    cerr << "in pack_address for " << name->data() << endl;
    int *ia;

    // this is a local message, so no conversion is necessary
    ia = new int[2];

    ia[0] = shm_seq_no;
    ia[1] = offset;
    msg->data = DataHandle((char *)ia, 2 * sizeof(int));
    msg->type = COVISE_MESSAGE_OBJECT_FOUND;
}

DMEntry::DMEntry(int i, char *h, Connection *c)
{
    id = i;
    host = new Host(h);
    conn = c;
    data_conn = c;
    transfermgr = 0;
    //    objects = new AVLTree<ObjectEntry>();
    objects = new AVLTree<ObjectEntry>(ObjectEntry_compare, "objects");
}

const char *DMEntry::get_hostname()
{
    if (host)
        return host->getAddress();
    return NULL;
}

int DMEntry::complete_data_conn(Message *msg)
{
    int data_port;
    Host *data_host;

    data_host = new Host(&msg->data.data()[SIZEOF_IEEE_INT]);
    data_port = *(int *)msg->data.data();
    data_conn = new ClientConnection(data_host, data_port, id, DATAMANAGER);
    return 1;
}

int DMEntry::make_data_conn(char *new_interface)
{
    char *data;
    int length;
    int data_port;
    ServerConnection *tmp_conn;

    length = strlen(new_interface) + 1 + SIZEOF_IEEE_INT;
    data = new char[length];
    tmp_conn = new ServerConnection(&data_port, id, DATAMANAGER);
    tmp_conn->listen();
    *(int *)data = data_port;
    strcpy(&data[SIZEOF_IEEE_INT], new_interface);
    Message *msg = new Message(COVISE_MESSAGE_COMPLETE_DATA_CONNECTION, DataHandle(data, length));
    conn->send_msg(msg);
    delete msg;

    if (tmp_conn->acceptOne() < 0)
    {
        fprintf(stderr, "DMEntry: accept in make_data_conn failed\n");
        return -1;
    }

    data_conn = tmp_conn;

    return 1;
}
