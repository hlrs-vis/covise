/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr.h"
#include "dmgr_packer.h"
#include <do/coDistributedObject.h>
#include <net/covise_host.h>
#include <covise/covise.h>
#ifdef SGI
#include <sys/ipc.h>
#endif

#ifdef _WIN32
#undef CRAY
#endif

#undef DEBUG
//extern CoviseTime *covise_time;

using namespace covise;

static int print_shm_list = 0;
static int processing_object_on_hosts = 0;

// just once, not once per message
static int rngbuf = coDistributedObject::calcType("RNGBUF");

//extern "C" int gethostname (char *name, int namelen);

/*********************************************************\ 
  int DataManagerProcess::handle_msg(Message *msg)

  returns  0 for a Message which is not handled here
      1 for a Message without reply
      2 if the Message must be sent back
      3 for the QUIT Message (on SGI only)
\*********************************************************/


int DataManagerProcess::handle_msg(Message *msg)
{
    int ok;
    coShmPtr *shmptr = NULL;
    unsigned int port;
    int len, i, number;
    int retval = 2;
#ifdef DEBUG
    int bytes_sent;
#endif
    int *idata;
    pid_t *tmp_pid;
    char dmsg_data[80];
    char remote_name[256];
    char new_interface_name[256];
    char *tmp_ptr, *data;
    ObjectEntry *oe;
    static int first = 1;

//	covise_time->mark(__LINE__, "START handle_msg");
#ifdef DEBUG
//	sprintf(tmp_str, "Sender ID: %d", msg->conn->get_sender_id());
//	print_comment(__LINE__, __FILE__, tmp_str);
#endif
    //	tmp_ptr = new char[100];
    //	sprintf(tmp_ptr,"msg->type: %d", msg->type);
    //	covise_time->mark(__LINE__, tmp_ptr);
    switch (msg->type)
    {
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_ASK_FOR_OBJECT:
        //-------------------------------------------------------------------------
        // message from other datamanager, conversion may be necessary
#ifdef DEBUG
        sprintf(tmp_str, "ASK_FOR_OBJECT %s", msg->data);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        ask_for_object(msg);
        retval = 1;
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_HAS_OBJECT_CHANGED:
        //-------------------------------------------------------------------------
        // message from other datamanager, conversion may be necessary
#ifdef DEBUG
        sprintf(tmp_str, "HAS_OBJECT_CHANGED %s", &msg->data[SIZEOF_IEEE_INT]);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        has_object_changed(msg);
        retval = 1;
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_SHM_MALLOC_LIST:
    {
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "SHM_MALLOC_LIST");
#endif
        DmgrMessage* dmgrmsg = (DmgrMessage*)msg;
        dmgrmsg->process_list(this);

        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_SHM_MALLOC:
    {
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "SHM_MALLOC");
#endif
        DmgrMessage* dmgrmsg = (DmgrMessage*)msg;
        shmptr = shm_alloc(dmgrmsg);
        ShmMessage tmpmsg(shmptr);
        msg->data = tmpmsg.data;
        msg->type = tmpmsg.type;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_OBJECT:
        //-------------------------------------------------------------------------
    {
        // message from local application, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_OBJECT");
        sprintf(tmp_str, "in NEW_OBJECT: %s (%d, %d)",
            &msg->data[2 * sizeof(int)],
            *(int*)msg->data,
            *(int*)(&msg->data[sizeof(int)]));
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        DataHandle dh = msg->data;
        dh.movePtr(2 * sizeof(int));
        ok = add_object(dh, *(int*)msg->data.data(),
            *(int*)(&msg->data.data()[sizeof(int)]), msg->conn);
        if (ok == 1)
            msg->type = COVISE_MESSAGE_MSG_OK;
        else
            msg->type = COVISE_MESSAGE_MSG_FAILED;
        msg->data = DataHandle();
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_OBJECT finished");
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_OBJECT_VERSION:
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_OBJECT_VERSION");
#endif
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_GET_OBJECT:
        //-------------------------------------------------------------------------
    {
        // message from local application, no conversion necessary
// and message contains only character data
#ifdef DEBUG
        sprintf(tmp_str, "GET_OBJECT %s", msg->data);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        oe = get_object(msg->data, msg->conn);
        msg->data = DataHandle();
        if (oe)
        {
            oe->pack_address(msg);
        } else
        {
            msg->type = COVISE_MESSAGE_OBJECT_NOT_FOUND;
        }
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_OBJECT_SHM_MALLOC_LIST:
    {
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_OBJECT_SHM_MALLOC_LIST");
        //	    print_shared_memory_statistics();
        print_shm_list = 1;
#endif
        DmgrMessage* dmgrmsg = (DmgrMessage*)msg;
        ok = dmgrmsg->process_new_object_list(this);
        if (ok == 1)
            msg->type = COVISE_MESSAGE_NEW_OBJECT_OK;
        else
        {
            msg->type = COVISE_MESSAGE_NEW_OBJECT_FAILED;
            print_comment(__LINE__, __FILE__, "NEW_OBJECT_SHM_MALLOC_LIST failed");
        }
#ifdef DEBUG
        sprintf(tmp_str, "NEW_OBJECT_SHM_MALLOC_LIST: %s (%d, %d)",
            msg->data,
            *(int*)msg->data,
            *(int*)(&msg->data[sizeof(int)]));
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_PART_ADDED:
        //-------------------------------------------------------------------------
    {
        // message from local module, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_PART_ADDED");
#endif
        forward_new_part(msg);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "Data Object sent");
#endif
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_PART_AVAILABLE:
        //-------------------------------------------------------------------------
    {
        // message from local module, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_PART_AVAILABLE");
#endif
        receive_new_part(msg);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "Data Object received");
#endif
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_PREPARE_CONTACT:
    {
        //-------------------------------------------------------------------------
        // message from controller, conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "PREPARE_CONTACT");
#endif
        prepare_for_contact((int*)& port);
        //		cerr << host->get_name() << " port in PREPARE_CONTACT: " << port << endl;
        swap_byte(port);
        //		cerr << host->get_name() << " port nach swap:          " << port << endl;
        Message portmsg{ COVISE_MESSAGE_PORT, DataHandle{(char*)& port, sizeof(int), false} };
        msg->conn->sendMessage(&portmsg);
        wait_for_contact();
        retval = 1;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "PREPARE_CONTACT finished");
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_PREPARE_CONTACT_DM:
    {
        //-------------------------------------------------------------------------
        // message from controller, conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "PREPARE_CONTACT_DM");
#endif
        prepare_for_contact((int*)& port);
#ifdef DEBUG
        sprintf(tmp_str, "Port number is: %d", port);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
#if defined(CRAY)
#ifdef _CRAYT3E
        converter.int_to_exch(port, dmsg_data);
#else
        conv_single_int_c8i4(port, (int*)dmsg_data);
#endif
#else
        //		cerr << host->get_name() << " port in PREPARE_CONTACT_DM: " << port << endl;
        swap_byte(port);
        //		cerr << host->get_name() << " port nach swap:             " << port << endl;
        *((unsigned int*)dmsg_data) = port;
#endif
        Host* tmphost = get_host();
        strncpy(&dmsg_data[SIZEOF_IEEE_INT], tmphost->getAddress(), 76);
        //	    covise_gethostname(&dmsg_data[SIZEOF_IEEE_INT], 76);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, &dmsg_data[SIZEOF_IEEE_INT]);
#endif
        len = sizeof(int) + (int)strlen(&dmsg_data[SIZEOF_IEEE_INT]) + 1;
#ifdef DEBUG
        sprintf(tmp_str, "Message length is: %d", len);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        Message portmsg{ COVISE_MESSAGE_PORT, DataHandle{dmsg_data, len, false} };
#ifdef DEBUG
        bytes_sent = msg->conn->send_msg(&portmsg);
        sprintf(tmp_str, "%d bytes sent", bytes_sent);
        print_comment(__LINE__, __FILE__, tmp_str);
#else
        msg->conn->sendMessage(&portmsg);
#endif
        wait_for_dm_contact();
        retval = 1;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "PREPARE_CONTACT_DM finished");
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_DM_CONTACT_DM:
    {
        //-------------------------------------------------------------------------
        // message from controller, conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "DM_CONTACT_DM");
        //	    cerr << "contact DM on host " << &msg->data.data()[SIZEOF_IEEE_INT] <<
        //	    	    " at port " << *(int *)msg->data.data() << "\n";
#endif
        Host* tmphost = new Host(&msg->data.data()[SIZEOF_IEEE_INT]);
#if defined(CRAY)
#ifdef _CRAYT3E
        converter.exch_to_uint(msg->data.accessData(), &port);
#else
        conv_single_int_i4c8(*(int*)msg->data.data(), &port);
#endif
#else
        port = *(int*)msg->data.data();
        //		cerr << host->get_name() << " port in DM_CONTACT_DM: " << port << endl;
        swap_byte(port);
        //		cerr << host->get_name() << " port nach swap:        " << port << endl;
#endif
        contact_datamanager(port, tmphost);
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "contact succeeded");
#endif
        retval = 1;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "DM_CONTACT_DM finished");
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_GET_SHM_KEY:
        //-------------------------------------------------------------------------
        // this should not happen on a Cray, therefore no #ifdef CRAY
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "GET_SHM_KEY");
#endif
        {
            char* d = new char[sizeof(int) * (MAX_NO_SHM * 2 + 1)];
            shm->get_shmlist(d);
            msg->data = DataHandle(d, *(int*)d * sizeof(int) * 2 + sizeof(int));
            print_comment(__LINE__, __FILE__, "GET_SHM_KEY: %d: %x, %d msg->data.length(): %d", *(int*)msg->data.data(),
                ((int*)msg->data.data())[1], ((int*)msg->data.data())[2], msg->data.length());

        }
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_CONNECT_TRANSFERMANAGER:
    {
        //-------------------------------------------------------------------------
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "CONNECT_TRANSFERMANAGER");
#endif
        if (!transfermanager)
            start_transfermanager();
        Message portmsg(COVISE_MESSAGE_CONNECT_TRANSFERMANAGER, DataHandle());
        send_trf_msg(&portmsg);
        msg->conn->sendMessage(&portmsg);
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_GET_TRANSFER_PORT:
    {
        //-------------------------------------------------------------------------
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "GET_TRANSFER_PORT");
#endif
        if (!transfermanager)
            start_transfermanager();

        Message portmsg(COVISE_MESSAGE_GET_TRANSFER_PORT, DataHandle());
        exch_trf_msg(&portmsg);
        if (portmsg.type == COVISE_MESSAGE_TRANSFER_PORT)
        {
#ifndef CRAY
            msg->conn->sendMessage(&portmsg);
#endif
        }
        //	    } else {
        //		tport = -1;
        //		portmsg = new Message(GET_TRANSFER_PORT, sizeof(int), (char *)&tport);
        //		msg->conn->send_msg(portmsg);
        //		delete portmsg;
        //	    }
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_SHM_FREE:
        //-------------------------------------------------------------------------
    {
        // no data that needs to be converted
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "in SHM_FREE");
#endif
        number = (msg->data.length() / sizeof(int)) / 2;
        idata = (int*)msg->data.data();
        for (i = 0; i < number; i++)
            shm_free(idata[i * 2], idata[i * 2 + 1]);
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_DESTROY_OBJECT:
        //-------------------------------------------------------------------------
        // no data that needs to be converted
    {
#ifdef DEBUG
        sprintf(tmp_str, "DESTROY_OBJECT %s", msg->data);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        ok = destroy_object(msg->data, msg->conn);
        if (ok == 1)
            msg->type = COVISE_MESSAGE_MSG_OK;
        else
            msg->type = COVISE_MESSAGE_MSG_FAILED;
        msg->data = DataHandle();

        if (print_shm_list == 1)
        {
            //		print_shared_memory_statistics();
            print_shm_list = 0;
        }
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_CTRL_DESTROY_OBJECT:
        //-------------------------------------------------------------------------
        // no data that needs to be converted
    {
#ifdef DEBUG
        sprintf(tmp_str, "CTRL_DESTROY_OBJECT %s", msg->data.data());
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        ok = destroy_object(msg->data, nullptr);
        if (ok == 1)
            msg->type = COVISE_MESSAGE_MSG_OK;
        else
            msg->type = COVISE_MESSAGE_MSG_FAILED;
        msg->data = DataHandle();
        if (print_shm_list == 1)
        {
            //		print_shared_memory_statistics();
            print_shm_list = 0;
        }
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_OBJECT_NO_LONGER_USED:
        //-------------------------------------------------------------------------
        // message contains only character data, no conversion necessary
    {
#ifdef DEBUG
        sprintf(tmp_str, "OBJECT_NO_LONGER_USED %s", msg->data.data());
        print_comment(__LINE__, __FILE__, tmp_str);
        if (print_shm_list == 1)
        {
            //		print_shared_memory_statistics();
            print_shm_list = 0;
        }
#endif
        oe = get_object(msg->data);
        if (oe)
            oe->remove_access(msg->conn);
        else
            print_comment(__LINE__, __FILE__, "Object has not been accessed");
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_SET_ACCESS:
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
    {
#ifdef DEBUG
        sprintf(tmp_str, "SET_ACCESS %s", &msg->data.data()[sizeof(int)]);
        print_comment(__LINE__, __FILE__, tmp_str);
#endif
        auto dh = msg->data;
        dh.movePtr(sizeof(int));
        oe = get_object(dh);
        if (oe)
            oe->set_current_access(msg->conn, *(access_type*)msg->data.data());
        else
            print_comment(__LINE__, __FILE__, "No such Object to access");

        msg->type = COVISE_MESSAGE_EMPTY;
        msg->data = DataHandle();
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_REGISTER_TYPE:
        //-------------------------------------------------------------------------
        // message from local application, no conversion necessary
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "REGISTER_TYPE");
#endif
        if (first)
        {
            rngbuf = coDistributedObject::calcType("RNGBUF");
            first = 0;
        }
        if (*(int*)msg->data.data() == rngbuf)
        {
            if (transfermanager == 0L)
            {
                start_transfermanager();
            }
        }
        msg->type = COVISE_MESSAGE_MSG_OK;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_GET_LIST_OF_INTERFACES:
        //-------------------------------------------------------------------------
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "GET_LIST_OF_INTERFACES");
#endif
        {
            char* c = get_list_of_interfaces(); //allocates memory
            msg->data = DataHandle(c, (int)strlen(c));
        }
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_COMPLETE_DATA_CONNECTION:
        //-------------------------------------------------------------------------
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "COMPLETE_DATA_CONNECTION");
#endif
        complete_data_connection(msg);
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_MAKE_DATA_CONNECTION:
        //-------------------------------------------------------------------------
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "MAKE_DATA_CONNECTION");
#endif
        {
            tmp_ptr = msg->data.accessData();
            i = 0;
            while (*tmp_ptr != '\0')
                remote_name[i++] = *tmp_ptr++;
            tmp_ptr++;
            i = 0;
            while (*tmp_ptr != '\0')
                new_interface_name[i++] = *tmp_ptr++;
            if (make_data_connection(remote_name, new_interface_name) == 0)
                print_comment(__LINE__, __FILE__, "Making Dataconnection failed");
        }
        break;
        //-------------------------------------------------------------------------
    case COVISE_MESSAGE_OBJECT_ON_HOSTS:
        //-------------------------------------------------------------------------
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "OBJECT_ON_HOSTS");
#endif
        if (processing_object_on_hosts)
        {
            msg->data = DataHandle();
            msg->type = COVISE_MESSAGE_OBJECT_NOT_FOUND;
        } else
        {
            processing_object_on_hosts = 1;
            msg->data = get_all_hosts_for_object(msg->data);

            processing_object_on_hosts = 0;
        }
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_NEW_DESK:
        //-------------------------------------------------------------------------
    {
        // message from controller, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "NEW_DESK");
#endif
        ok = DTM_new_desk();

        retval = 1;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "Data Manager created a new desk");
#endif
        break;
    }

    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_CRB_QUIT:
        //-------------------------------------------------------------------------
        // message from controller, no conversion necessary
    {
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "CRB_QUIT");
#endif
        //cerr << endl << "---CRB_QUIT for : " << msg->data;
        rmv_rdmgr(msg->data.accessData());

        retval = 1;
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "Another Data Manager quits");
#endif
        break;
    }

    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_QUIT:
//-------------------------------------------------------------------------
    {
        // message from controller, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "QUIT");
#endif
#ifdef CRAY
        retval = 0;
#else
        //msg->conn->send_msg(msg);
        retval = 3;
#endif

#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "Data Manager correctly finished");
#endif
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_SEND_APPL_PROCID:
//-------------------------------------------------------------------------
    {
        // message from local process, no conversion necessary
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "SEND_APPL_PROCID");
#endif
        tmp_pid = pid_list;
        pid_list = new pid_t[++no_of_pids];
        for (i = 0; i < no_of_pids - 1; i++)
            pid_list[i] = tmp_pid[i];
        pid_list[i] = *(pid_t*)msg->data.data();
        delete[] tmp_pid;
        //#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "SEND_APPL_PROCID: %d", *(pid_t*)msg->data.data());
        //#endif
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    case COVISE_MESSAGE_CLOSE_SOCKET:
//-------------------------------------------------------------------------
    {
        // message from application module
#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "CLOSE_SOCKET");
#endif
        //#ifdef DEBUG
        print_comment(__LINE__, __FILE__, "CLOSE_SOCKET: %d", msg->conn->get_port());
        //#endif
        list_of_connections->remove(msg->conn);
        delete msg->conn;
        retval = 1;
        break;
    }
    //-------------------------------------------------------------------------
    default:
//-------------------------------------------------------------------------
#ifdef DEBUG
        sprintf(tmp_str, "DM: default == %d", msg->type);
        print_comment(__LINE__, __FILE__, tmp_str);
//	    cerr << "message " << msg->type << ": ";
//	    if(msg->data)
//	    	cerr << *(int *)msg->data.data() << " from " << *(int *)&msg->data[SIZEOF_IEEE_INT];
//	    cerr << endl;
//	    msg->conn->send_msg(msg);
#endif
        retval = 0;
        break;
    }
    //    covise_time->mark(__LINE__, "END   handle_msg");
    return retval;
}
