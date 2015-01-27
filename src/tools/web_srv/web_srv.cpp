/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:             ++
// ++                                                                     ++
// ++ Author:                  ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#define DEFINE_URI_TYPE
#include "web_srv.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

char *home_path;

void process_signal_handler(int sg, char *data)
{
    // int cpid, status;
    switch (sg)
    {
    case SIGPIPE:
        break;
    default:
        delete[] data;
        cerr << endl << " unexpected signal received -> exiting .... !!!!";
        sleep(1);
        exit(1);
        break;
    }
}

/////////////////////    InfoUri    ////////////////////////////

InfoURI::InfoURI()
{
    m_length = 0;
    m_type = M_LAST;
    m_name = NULL;
    m_URI = NULL;
    m_conn = NULL;
}

InfoURI::~InfoURI()
{
    m_length = 0;
    if (m_name != NULL)
        delete[] m_name;
    if (m_URI != NULL)
        delete[] m_URI;
}

/////////////////////    WHost    ////////////////////////////

WHost::WHost()
{
    m_address = NULL;
    m_wport = 0;
    m_cport = 0;
    m_refs = 0;
}

WHost::WHost(char *addr)
{
    if (addr == NULL)
        m_address = NULL;
    else
    {
        m_address = new char[strlen(addr) + 1];
        strcpy(m_address, addr);
    }

    m_refs = 1;
    m_wport = 0;
    m_cport = 0;
}

WHost::~WHost()
{
    if (m_address != NULL)
        delete[] m_address;
}

/////////////////////    WebSrv    ////////////////////////////

WebSrv::WebSrv()
{
    m_aws_port = 0;
    m_aws_conn = NULL;
    m_connList = new ConnectionList;
    m_hostlist = new Liste<WHost>(1);
    m_id = (int)getpid();
    sig_handler.addSignal(SIGPIPE, (void *)process_signal_handler, (void *)this);
    dConnection = NULL;
    last_loaded_map_ = NULL;
}

WebSrv::~WebSrv()
{
    if (m_connList)
        delete m_connList;
    if (m_hostlist)
    {
        char *addr;
        WHost *host;
        Host *tmp_host;
        int cport;

        m_hostlist->reset();
        while ((host = m_hostlist->next()) != NULL)
        {
            cport = host->get_cport();
            if (cport)
            {
                addr = host->getAddress();
                tmp_host = new Host(addr);
                remove_aws(tmp_host, cport);
                delete tmp_host;
            }
        }
        delete m_hostlist;
    }
    if (m_aws_conn != NULL)
        delete m_aws_conn;
    delete dConnection;
}

WHost *WebSrv::add_host(char *name)
{
    char *addr;
    WHost *host;

    // cerr << endl << "--- add_host(" << name << ")\n";

    if (name == NULL)
        return NULL;
    Host tmp_host(name);
    addr = (char *)tmp_host.getAddress();
    if (addr == NULL)
        return NULL;

    m_hostlist->reset();
    while ((host = m_hostlist->next()) != NULL)
    {
        if (strcmp(host->getAddress(), addr) == 0)
        {
            host->inc_refs();
            return host;
        }
    }
    host = new WHost(addr);
    m_hostlist->add(host);
    return host;
}

int WebSrv::remove_host(char *name)
{
    char *addr;
    WHost *host;
    Host *tmp_host;

    if (name == NULL)
        return -1;
    tmp_host = new Host(name);
    addr = (char *)tmp_host->getAddress();
    if (addr == NULL)
    {
        delete tmp_host;
        return -1;
    }

    m_hostlist->reset();
    while ((host = m_hostlist->next()) != NULL)
    {
        if (strcmp(host->getAddress(), addr) == 0)
        {
            int refs = host->dec_refs();
            if (refs == 0)
            {
                int cport = host->get_cport();
                if (cport)
                    remove_aws(tmp_host, cport);
                m_hostlist->remove(host);
            }
            delete tmp_host;
            return refs;
        }
    }
    delete tmp_host;
    return -2;
}

WHost *WebSrv::get_host(char *name)
{
    char *addr;
    WHost *host;

    //cerr << endl << "--- WebSrv::get_host(" << name << ")\n";

    if (name == NULL)
        return NULL;
    Host tmp_host(name);
    addr = (char *)tmp_host.getAddress();
    if (addr == NULL)
        return NULL;

    m_hostlist->reset();
    while ((host = m_hostlist->next()) != NULL)
    {
        if (strcmp(host->getAddress(), addr) == 0)
        {
            return host;
        }
    }
    return NULL;
}

int WebSrv::remove_aws(Host *host, int cport)
{
    CClientConnection *aws_conn;

    int id = (int)getpid();

    aws_conn = new CClientConnection(host, cport, id);
    if (aws_conn->get_id() <= 0)
    {
        delete aws_conn;
        return -1;
    }
    else
    {
        CMessage *p_msg = new CMessage;
        p_msg->m_type = C_QUIT;
        p_msg->m_data = "STOP AWS\n";

        p_msg->m_length = strlen(p_msg->m_data) + 1;

        aws_conn->send_msg(p_msg);

        delete p_msg;
        delete aws_conn;
    }
    return 1;
}

int WebSrv::open_channel(int port, conn_type type)
{
    Connection *open_channel;

    switch (type)
    {
    case CON_HTTP:
        open_channel = (HServerConnection *)new HServerConnection(port, m_id);
        break;
    case CON_COVISE:
        open_channel = (CServerConnection *)new CServerConnection(port, m_id);
        break;
    default:
        cerr << endl << " Channel type not implemented " << type << " !!";
        return -1;
    }
    if (open_channel->get_id() <= 0)
    {
        delete open_channel;
        return -1;
    }
    open_channel->listen();
    m_connList->add_ch(open_channel);
    return 1;
}

int WebSrv::open_channel(int *port, conn_type type)
{
    Connection *open_channel;

    switch (type)
    {
    case CON_HTTP:
        open_channel = (HServerConnection *)new HServerConnection(port, m_id);
        break;
    case CON_COVISE:
        open_channel = (CServerConnection *)new CServerConnection(port, m_id);
        break;
    default:
        cerr << endl << " Channel type not implemented " << type << " !!";
        return -1;
    }
    if (open_channel->get_id() <= 0)
    {
        delete open_channel;
        return -1;
    }
    open_channel->listen();
    m_connList->add_ch(open_channel);
    return 1;
}

InfoURI *WebSrv::get_infoURI(HMessage *p_Msg)
{
    InfoURI *p_info;
    struct stat file_stat;
    char *line;
    char *URI_name;
    int i;

    URI_name = p_Msg->m_URI;

    p_info = new InfoURI;

    p_info->m_type = M_APPLICATION_OCTET;

    // preprocess name

    p_info->m_name = new char[strlen(home_path) + strlen(URI_name) + 1];
    strcpy(p_info->m_name, home_path);
    strcat(p_info->m_name, URI_name); // absolute URI

    p_info->m_URI = new char[strlen(URI_name) + 1];
    strcpy(p_info->m_URI, URI_name); // received URI

#ifdef _AIRBUS
    if (strncmp(URI_name, "/launch", 7) == 0)
    {
        delete[] p_info -> m_URI;
        p_info->m_type = M_START_COVISE;
        char *file_name = strchr(URI_name + 2, '/');

        if (!file_name)
        {
            p_info->m_URI = NULL;
        }
        else
        {
            p_info->m_URI = new char[strlen(file_name)];
            strcpy(p_info->m_URI, file_name + 1);
            if (p_info->m_URI[0] = '+')
            {
                p_info->m_URI[0] = '/';
            }
        }
        return p_info;
    }
    else if (strcmp(URI_name, "/quit") == 0)
    {
        p_info->m_type = M_QUIT_COVISE;
        return p_info;
    }
    else if (strcmp(URI_name, "/forcequit") == 0)
    {
        p_info->m_type = M_QUIT_COVISE_SESSION;
        return p_info;
    }
    else if (strcmp(URI_name, "/clean") == 0)
    {
        p_info->m_type = M_CLEAN_COVISE;
        return p_info;
    }
    else if (strcmp(URI_name, "/daemon") == 0)
    {
        p_info->m_type = M_START_DAEMON;
        return p_info;
    }
    /****************************************
         else if(strncmp(URI_name,"/join", 5 )==0)
         {
            if( !dConnection || dConnection->isBad() )
       {
          startDaemon();
       }

            const char *answer = dConnection->askState();
       const char *page;

   if( !strcmp(answer, "MASTER") )
   {
   cout << "---------NetMeeting Server ---------" << endl;
   p_info->m_name = strcpy( new char[strlen("netmeeting_server.html")+1],"netmeeting_server.html" ) ;
   }
   else if( !strcmp(answer, "SLAVE") )
   {
   cout << "---------NetMeeting Client ---------" << endl;
   p_info->m_name = strcpy( new char[strlen("netmeeting_client.html")+1],"netmeeting_client.html" ) ;
   }
   else
   {

   }

   if(stat(p_info->m_name,&file_stat)==0)
   {
   p_info->m_length = file_stat.st_size;
   }
   return p_info;
   }
   *************************************/
    else if (0)
    {
        p_info->m_type = M_LAST;
        delete p_info;
        return NULL;
    }
#endif
    //=========== old cgi API

    if (strcmp(URI_name, "/registered_users") == 0)
    {
        p_info->m_type = M_REGISTERED_USERS;
        return p_info;
    }

    else if (strcmp(URI_name, "/fetch") == 0)
    {
        delete[] p_info -> m_URI;
        p_info->m_URI = new char[strlen(safe) + 1];
        strcpy(p_info->m_URI, safe);
        p_info->m_type = M_XWORLD_XWRML;
    }
    else if (strcmp(URI_name, "/get_last_wrl") == 0)
    {
        char *last_wrl = m_connList->get_last_wrl();
        delete[] p_info -> m_URI;

        p_info->m_URI = new char[strlen(last_wrl) + 1];
        strcpy(p_info->m_URI, last_wrl);
        delete[] last_wrl;
        if (strcmp(p_info->m_URI, "/default.wrl") == 0)
        {
            delete[] p_info -> m_name;
            p_info->m_name = new char[strlen(home_path) + strlen(p_info->m_URI) + 1];
            strcpy(p_info->m_name, home_path);
            strcat(p_info->m_name, p_info->m_URI);
        }
        p_info->m_type = M_XWORLD_XWRML;
    }
    //=========== end old cgi API
    // find media type
    else
    {
        line = strrchr(p_info->m_URI, '.');
        if ((line != NULL) && (strlen(line) > 1))
        {
            line++; // skip '.'
            i = 0;
            while ((URI_ext_array[i] != NULL) && (strcasecmp(URI_ext_array[i], line) != 0))
                i++;

            if (URI_ext_array[i] != NULL)
            {
                p_info->m_type = URI_ext_type[i];

                if ((p_info->m_type != M_XWRML_RENDERER) && (strncmp(line, "cgi-", 4) == 0))
                    return p_info;
            }
        }
    }

    if (p_info->m_type == M_XWRML_RENDERER)
    {
        p_info->m_conn = (CConnection *)m_connList->get_conn(p_info->m_URI);
        if (p_info->m_conn != NULL)
        {
            char reg_id[50];
            sprintf(reg_id, "app%s", p_info->m_URI);
            p_Msg->m_conn->set_connid(reg_id); //registering the user of rnd

            return p_info;
        }
        else
        {
            delete[] p_info -> m_name;
            p_info->m_name = new char[strlen(home_path) + strlen("/default.wrl") + 1];
            strcpy(p_info->m_name, home_path);
            strcat(p_info->m_name, "/default.wrl");
            p_info->m_type = M_XWORLD_XWRML;
            p_Msg->m_conn->set_connid(NULL); //unregistering the user of rnd
        }
    }

    // get length if URI exists
    if (stat(p_info->m_name, &file_stat) == 0)
    {
        p_info->m_length = file_stat.st_size;
    }
    else // URI doesn't exist
    {
        p_info->m_type = M_LAST;
        delete p_info;
        return NULL;
    }

    return p_info;
}

char *WebSrv::get_file(char *file_name, int get_flag)
{

    FILE *fd;
    char *buff;
    char *p_buff;
    int no, total = 0, ava = 0;

    fd = fopen(file_name, "r");

    if (!fd)
    {
        cerr << "$$$$ Error web_srv::get_file(...): could not open web file "
             << file_name << " !!!!! " << endl;
        perror("Error : ");
        return NULL;
    }

    buff = new char[READ_BUFFER_SIZE + 1];

    if (get_flag) // if we need the content
    {
        p_buff = buff;
        ava = READ_BUFFER_SIZE;
        while (!feof(fd))
        {
            if (ava <= total)
            {
                ava += READ_BUFFER_SIZE;
                p_buff = new char[ava + 1];
                memcpy(p_buff, buff, total);
                delete[] buff;
                buff = p_buff;
            }
            p_buff = &buff[total];
            no = fread(p_buff, 1, READ_BUFFER_SIZE, fd);
            if (no < 0)
            {
                cerr << "$$$$ Error web_srv send_file(...): could not read web file "
                     << file_name << " !!!!! " << endl;
                perror("Error : ");
                fclose(fd);
                return NULL;
            }
            else
            {
                total += no;
            }
        }
    }
    fclose(fd);

    buff[total] = '\0';

    //cerr << " The content of the file is :" << buff << "===";

    return buff;
}

void WebSrv::send_registered_users(HConnection *p_conn)
{
    char *users;
    HMessage *p_sHMsg;
    p_sHMsg = new HMessage(ST_200); // OK
    p_sHMsg->add_header(CONNECTION, "keep-alive");

    users = m_connList->get_registered_users(safe);
    if (users)
    {
        p_sHMsg->add_body(users, strlen(users), "en", M_REGISTERED_USERS);
        p_sHMsg->m_send_content = 1;
        p_conn->send_msg(p_sHMsg);
    }
    else
        cerr << endl << " registered users list == NULL !!\n";
    delete p_sHMsg;
}

void WebSrv::head_get(HMessage *p_Msg, HConnection *p_conn, int get_flag)
{
    InfoURI *p_info;
    CMessage *p_rCMsg;
    CMessage *p_sCMsg;
    //HMessage *p_rHMsg;
    HMessage *p_sHMsg;
    char buff[200];
    char *msg_body;
    char *tmp;
    int ret;

    p_info = get_infoURI(p_Msg);

    if (p_info == NULL)
    {
        p_conn->send_http_error(ST_404, p_Msg->m_URI, get_flag);
        return;
    }

    switch (p_info->m_type)
    {
#ifdef _AIRBUS
    case M_START_COVISE:

        system("xset s 1 ");
        system("xset s activate");

        if (dConnection && !dConnection->isBad())
        {
            dConnection->askState(buff);
            if (!strncmp(buff, "SLAVE", 5))
            {
                char *master_host = strchr(buff, ':');
                char wget_string[2056];
                if (p_info->m_URI[0] = '/')
                {
                    p_info->m_URI[0] = '+';
                }
                sprintf(wget_string, "wget -a .quit-log -t 1 -T 1 -b \"http://%s:8000/launch/%s\" ", master_host + 1, p_info->m_URI);
                cout << "Calling " << wget_string << endl;

                system(wget_string);
                p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
            }
            else
            {
                dConnection->sendLaunchMsg();

                if (last_loaded_map_ && !strcmp(last_loaded_map_, p_info->m_URI))
                {
                    p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
                }
                else
                {
                    if (dConnection->sendLoadMsg(p_info->m_URI))
                    {
                        last_loaded_map_ = new char[strlen(p_info->m_URI)];
                        strcpy(last_loaded_map_, p_info->m_URI);

                        p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
                    }
                    else
                    {
                        p_conn->send_http_error(ST_404, p_Msg->m_URI, get_flag);
                    }
                }
            }
        }
        else
        {
            system("killall -9 covised");

            cout << "WebSrv::Starting Daemon " << endl;
            startDaemon(true);

            dConnection->sendLaunchMsg();

            if (dConnection->sendLoadMsg(p_info->m_URI))
            {
                p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
                last_loaded_map_ = new char[strlen(p_info->m_URI) + 1];
                strcpy(last_loaded_map_, p_info->m_URI);
            }
            else
            {
                p_conn->send_http_error(ST_404, p_Msg->m_URI, get_flag);
            }
        }

        break;

    case M_QUIT_COVISE:
        system("xset s 1 ");
        system("xset s activate");

        delete[] last_loaded_map_;
        last_loaded_map_ = NULL;

        if (dConnection)
        {
            if (!dConnection->isBad())
            {
                dConnection->sendQuitMsg();

                int num_tries = 0, maxtries = 10;
                while (!dConnection->isBad() && num_tries++ < maxtries)
                {
                    sleep(2);
                    cout << "Waiting for closed daemon connection ..." << endl;
                }
                if (num_tries >= maxtries)
                {
                    cout << "Killing covised after " << num_tries << " tries " << endl;
                    system("killall -9 covised");
                    cleanCovise();
                    sleep(4);
                }
                delete dConnection;
                dConnection = NULL;
            }
        }
        else
        {
            system("killall -9 covised");
        }
        p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
        break;

    case M_QUIT_COVISE_SESSION:
        system("xset s 1 ");
        system("xset s activate");

        delete[] last_loaded_map_;
        last_loaded_map_ = NULL;

        if (dConnection)
        {
            dConnection->askState(buff);

            if (!dConnection->isBad() && !strcmp(buff, "MASTER"))
            {
                dConnection->sendQuitMsg();

                int num_tries = 0, maxtries = 30;
                while (!dConnection->isBad() && num_tries++ < maxtries)
                {
                    sleep(2);
                }
                if (num_tries >= maxtries)
                {
                    cout << "Killing covised after " << num_tries << " tries " << endl;
                    system("killall -9 covised");
                    cleanCovise();
                    sleep(4);
                }
                delete dConnection;
                dConnection = NULL;
            }
        }
        p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
        break;

    case M_CLEAN_COVISE:
        cleanCovise();

        p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);
        break;

    case M_START_DAEMON:
    {
        startDaemon();
        p_conn->send_http_error(ST_200, p_Msg->m_URI, get_flag);

        break;
    }
#endif

    case M_REGISTERED_USERS:
        send_registered_users(p_conn);
        break;
    case M_SET_DYNAMIC_USR:
        p_conn->set_dynamic_usr(1);
        //cerr << endl << "$$$$ M_SET_DYNAMIC_USR " << endl;
        break;
    case M_RST_DYNAMIC_USR:
        p_conn->set_dynamic_usr(0);
        //cerr << endl << "$$$$ M_RST_DYNAMIC_USR " << endl;
        break;
    case M_SET_DYNAMIC_VIEW:
        p_conn->set_dynamic_view(1);
        tmp = p_conn->get_connid();
        ret = m_connList->add_dynamic_view(tmp);
        //cerr << endl << "$$$$ M_SET_DYNAMIC_VIEW FOR " << tmp << " = " << ret << endl;
        break;
    case M_RST_DYNAMIC_VIEW:
        p_conn->set_dynamic_view(0);
        tmp = p_conn->get_connid();
        ret = m_connList->remove_dynamic_view(tmp);
        //cerr << endl << "$$$$ M_RST_DYNAMIC_VIEW FOR " << tmp << " = " << ret << endl;
        break;
    case M_TEXT_HTML:
    case M_TEXT_PLAIN:
        msg_body = get_file(p_info->m_name, get_flag);
        if (msg_body == NULL)
        {
            //Forbidden
            p_conn->send_http_error(ST_403, p_info->m_URI, get_flag);
        }
        else
        {
            p_sHMsg = new HMessage(ST_200); // OK
            p_sHMsg->add_header(CONNECTION, "keep-alive");
            p_sHMsg->add_header(CACHE_CONTROL, "no-cache");
            p_sHMsg->add_header(PRAGMA, "no-cache");
            p_sHMsg->add_body(msg_body, p_info->m_length, "en", p_info->m_type);
            p_sHMsg->m_send_content = get_flag;
            p_conn->send_msg(p_sHMsg);
            delete p_sHMsg;
        }
        break;
    case M_IMAGE_GIF:
    case M_APPLICATION_OCTET: //file transfer
    case M_XWORLD_XWRML:
    case M_CACERT:
        FILE *fd;

        fd = fopen(p_info->m_name, "r");
        if (fd == NULL) // without acces permission
        {
            //Forbidden
            p_conn->send_http_error(ST_403, p_info->m_URI, get_flag);
        }
        else
        {
            msg_body = new char[strlen(p_info->m_URI) + 1];
            strcpy(msg_body, p_info->m_URI);

            p_sHMsg = new HMessage(ST_200); // OK
            p_sHMsg->add_header(CONNECTION, "keep-alive");
            p_sHMsg->add_header(CACHE_CONTROL, "no-cache");
            p_sHMsg->add_header(PRAGMA, "no-cache");
            p_sHMsg->m_indirect = 1;
            p_sHMsg->m_fd = fd;
            p_sHMsg->add_body(msg_body, p_info->m_length, "en", p_info->m_type);
            p_sHMsg->m_send_content = get_flag;
            if (p_conn->send_msg(p_sHMsg) < 0)
            {
                cerr << endl << "Error sending response !!!";
            }
            fclose(fd);
            delete p_sHMsg;
        }
        break;
    case M_XWRML_RENDERER:
        //cerr << endl << " M_XWRML_RENDERER media type obtained ! ";
        CConnection *conn;
        conn = p_info->m_conn;

        p_sCMsg = new CMessage;
        p_sCMsg->m_type = C_ASK_FOR_OBJECT;
        p_sCMsg->m_data = NULL;
        p_sCMsg->m_length = 0;

        ret = conn->send_msg(p_sCMsg);
        if (ret < 0)
        {
            cerr << endl << "Communication error !!!\n";
            p_rCMsg = NULL;
        }
        else
        {
            delete p_sCMsg;
            p_rCMsg = conn->recv_msg();
            //p_rCMsg->print();
            while ((p_rCMsg != NULL) && (p_rCMsg->m_type == C_OBJECT_TRANSFER))
            {
                //cerr << "\n--- C_OBJECT_TRANSFER:" << &p_rCMsg->m_data[p_rCMsg->m_length-30] << "---\n";
                p_sHMsg = new HMessage(ST_200); // OK
                p_sHMsg->add_header(CONNECTION, "keep-alive");
                p_sHMsg->add_body(p_rCMsg->m_data, p_rCMsg->m_length, "en", p_info->m_type);
                p_sHMsg->m_send_content = 1;
                ret = p_conn->send_msg(p_sHMsg);
                if (ret < 0)
                {
                    cerr << endl << "Communication error !!!\n";
                    p_rCMsg = NULL;
                }
                else
                {
                    p_sHMsg->m_data = NULL; //double reference
                    delete p_sHMsg;
                    delete p_rCMsg;
                    p_rCMsg = conn->recv_msg();
                    //p_rCMsg->print();
                }
            }
        }
        if ((p_rCMsg == NULL) || (p_rCMsg->m_type != C_PARINFO))
        {
            // << p_rCMsg->m_type;
            cerr << "\n Wrong message received !!! ";
        }
        else
        {
            strcpy(buff, " 5 ");
            strcat(buff, p_rCMsg->m_data);

            p_sHMsg = new HMessage(ST_200); // OK
            p_sHMsg->add_header(CONNECTION, "keep-alive");
            p_sHMsg->add_body(buff, strlen(buff), "en", M_TEXT_HTML);
            p_sHMsg->m_send_content = 1;
            p_conn->send_msg(p_sHMsg);
            p_sHMsg->m_data = NULL; //reference to a "static" string
            delete p_sHMsg;
        }
        if (p_rCMsg)
            delete p_rCMsg;
        break;
    case M_LAST: // URI not found
        p_conn->send_http_error(ST_404, p_info->m_URI, get_flag);
        break;

    case M_APPLICATION_PROC: // TODO
        // media type not implemented yet
        p_conn->send_http_error(ST_415, URI_type_array[p_info->m_type], get_flag);
        break;

    default:
        // media type not implemented
        p_conn->send_http_error(ST_415, URI_type_array[p_info->m_type], get_flag);
        break;

    } // end switch

    delete p_info;
}

FILE *
WebSrv::openFile(const char *filename, char **file_buf, int *size)
{
    struct stat st;
    FILE *f = NULL;
    *size = 0;

    if (!stat(filename, &st) && (f = fopen(filename, "r")) != NULL)
    {
        *file_buf = new char[st.st_size + 1];
        *size = st.st_size;
    }
    return f;
}

void
WebSrv::getPartnerHosts(char **partner_hosts, int *num_partners)
{
    char host_file[1024], *filebuf;
    char line[1024];
    int size;

    sprintf(host_file, "%s/.partners", getenv("COVISEDIR"));
    FILE *f = openFile(host_file, &filebuf, &size);

    *num_partners = 0;

    if (f)
    {
        while (fgets(line, size, f))
        {
            int len = strlen(line);
            partner_hosts[(*num_partners)] = new char[len + 1];
            strcpy(partner_hosts[(*num_partners)], line);
            partner_hosts[(*num_partners)++][len - 1] = '\0';
        }

        delete[] filebuf;
    }
}

void
WebSrv::contactWeb_partner(const char *uri)
{
    char **partner_hosts = new char *[1024];
    int num_partners;

    getPartnerHosts(partner_hosts, &num_partners);

    char wget_string[1024];
    int i;

    for (i = 0; i < num_partners; i++)
    {
        sprintf(wget_string, "wget -a .quit-log -t 1 -T 1 -b \"http://%s:8000/%s\" ", partner_hosts[i], uri);
        system(wget_string);
        delete[] partner_hosts[i];
    }
    delete[] partner_hosts;
    sleep(4);
}

void
WebSrv::cleanCovise()
{
    system("xset s 1 ");
    system("xset s activate");
    system("clean_covise");
    system("killall -9 crb covise covised Mapeditor wget");
}

void
WebSrv::startDaemon(bool master)
{
    if (dConnection == NULL || dConnection->isBad())
    {
        system("killall -9 wget");

        delete dConnection;
        dConnection = NULL;

        int timeout = 10;

        system("start_daemon");
        sleep(timeout);

        dConnection = new DaemonConn;
        int retries = 0, maxTries = 10;

        while (dConnection->isBad() && retries++ < maxTries)
        {
            delete dConnection;

            cleanCovise();
            system("start_daemon");

            sleep(timeout);
            dConnection = new DaemonConn;
        }

        if (!dConnection->isBad())
        {
            cout << "WebSrv:  Daemon connected " << endl;
        }

        cout << "after " << retries << " of " << maxTries << " tries" << endl;

        if (master && retries < maxTries)
        {
            retries = 0;
            timeout = 4;
            maxTries = 40;

            char answer[64];

            dConnection->askState(answer);

            while (!strcmp(answer, "WAITING") && retries++ < maxTries)
            {
                cout << "WebSrv:: Waiting until daemon startup has finished ... " << endl;
                sleep(timeout);
                dConnection->askState(answer);
            }
        }
    }
}

void WebSrv::register_vrml(CMessage *msg)
{
    //CMessage* p_msg;
    char *tmp;
    char *buff;
    char *name;
    char *hostname;
    int cport, wport;

    if (msg->m_data)
    {
        //msg->m_conn->set_connid("generic.wrl");
        //cerr << endl << "$$$$$$ register_vrml : " << msg->m_data << endl;
        int l = strlen(msg->m_data) + 1;
        buff = new char[l];
        strcpy(buff, msg->m_data);
        name = strtok(buff, "(");
        hostname = strtok(NULL, ")");
        tmp = strtok(NULL, "#");
        tmp = strtok(NULL, "_");
        cport = atoi(tmp);
        tmp = strtok(NULL, "\n");
        wport = atoi(tmp);

        //cerr << "\n hostname = " << hostname << " cport= " << cport << " wport= " << wport << endl;

        WHost *host = get_host(hostname);
        if (host != NULL)
            host->set_ports(cport, wport);
        sprintf(buff, "%s(%s).cgi-rnd", name, hostname);
        msg->m_conn->set_connid(buff);
        //new vrml - update
        m_connList->broadcast_usr(" 3 /", msg->m_data);
    }
    else
        cerr << "\n Error trying to register a conn with a NULL id \n";
}

void WebSrv::mainLoop(void)
{
    Connection *p_conn;
    HMessage *p_sHMsg;
    Message *p_rMsg;
    char *p_txt;
    char *tmp;
    char buff[200];
    int quit;
    char *id;

    quit = 0;

    while (!quit)
    {

        p_conn = m_connList->wait_for_input();

        if (p_conn == NULL)
            continue;
        p_rMsg = p_conn->recv_msg();

        ////////////////////////////////////////////////////////////
        /////                                                     //
        ///// Switch across message types                         //
        /////                                                     //
        ////////////////////////////////////////////////////////////

        /********
            cerr << endl << "$$$$$$$ Web Server received a message : " << p_rMsg->m_type ;
            cerr << endl << " <<<<<<<<<<<<<<<<< \n";
            if(p_rMsg)  p_rMsg->print();
            else cerr << "-- Got an NULL msg --";
            cerr << "\n >>>>>>>>>>>>>>> \n";
      ***********/

        switch (p_rMsg->m_type)
        {
        case SOCKET_CLOSED:
        case CLOSE_SOCKET:
        case EMPTY:
        case C_CLOSE_SOCKET:
        case C_SOCKET_CLOSED:
            id = p_conn->get_connid();
            //if(id != NULL) cerr << "\n$$$$$ closing :" << id << endl;
            //else  cerr << "\n$$$$$ closing with null id !!" << endl;
            switch (p_conn->get_type())
            {
            case CON_HTTP:
                if (p_conn->is_dynamic_view())
                {
                    p_txt = p_conn->get_connid();
                    m_connList->remove_dynamic_view(p_txt);
                }
                break;
            case CON_COVISE:
                p_txt = p_conn->get_connid();
                if (p_txt != NULL)
                {
                    //remove aws if necessary
                    strcpy(buff, p_txt);
                    tmp = strtok(buff, "(");
                    tmp = strtok(NULL, ")");
                    if (tmp != NULL)
                        remove_host(tmp);
                    //remove vrml - update
                    m_connList->broadcast_usr(" 4 ", p_txt);
                }
                break;
            case CON_GENERIC:
                cerr << "WebSrv::mainLoop: CON_GENERIC not handled" << endl;
                break;
            }

            m_connList->remove(p_conn);
            //delete p_conn;
            break;
        case GET:
            //cerr << endl << " $$$$ Constructing a response for GET " << endl;
            head_get((HMessage *)p_rMsg, (HConnection *)p_conn, 1);
            break;
        case HEAD:
            //cerr << endl << " $$$$ Constructing a response for HEAD " << endl;
            head_get((HMessage *)p_rMsg, (HConnection *)p_conn, 0);
            break;
        case C_INIT:
            int cport, wport;
            WHost *host;
            CMessage *p_msg;

            host = add_host(p_rMsg->m_data);
            cport = host->get_cport();
            if (cport == 0)
            { // new aws
                p_msg = new CMessage;
                p_msg->m_type = C_MAKE_DATA_CONNECTION;
                p_msg->m_data = " ";
                p_msg->m_length = strlen(p_msg->m_data) + 1;
                p_rMsg->m_conn->send_msg(p_msg);

                delete p_msg;
            }
            else
            { // already started aws
                wport = host->get_wport();
                p_msg = new CMessage;
                sprintf(buff, "%d %d", cport, wport);
                p_msg->m_type = C_COMPLETE_DATA_CONNECTION;
                p_msg->m_data = &buff[0];
                p_msg->m_length = strlen(p_msg->m_data) + 1;
                p_rMsg->m_conn->send_msg(p_msg);

                delete p_msg;
            }
            break;
        case C_START:
            //cerr << endl << " $$$$ C_START received !!  " << endl;
            register_vrml((CMessage *)p_rMsg);
            break;
        case C_QUIT:
            quit = 1;
            break;
        case C_OBJECT_TRANSFER:
            //cerr << endl << "$$$$ C_OBJECT_TRANSFER msg received !!!\n";
            p_sHMsg = new HMessage(ST_200); // OK
            p_sHMsg->add_header(CONNECTION, "keep-alive");
            p_sHMsg->add_body(p_rMsg->m_data, p_rMsg->m_length, "en", M_XWRML_RENDERER);
            p_sHMsg->m_send_content = 1;
            p_txt = p_conn->get_connid();
            m_connList->broadcast_view(p_txt, p_sHMsg);
            p_sHMsg->m_data = NULL; //double reference
            delete p_sHMsg;
            break;

        case C_PARINFO:
            //cerr << endl << "$$$$ C_PARINFO msg received :" << p_rMsg->m_data << "------------\n";
            strcpy(buff, " 5 ");
            strcat(buff, p_rMsg->m_data);

            p_sHMsg = new HMessage(ST_200); // OK
            p_sHMsg->add_header(CONNECTION, "keep-alive");
            p_sHMsg->add_body(buff, strlen(buff), "en", M_TEXT_HTML);
            p_sHMsg->m_send_content = 1;

            p_txt = p_conn->get_connid();
            m_connList->broadcast_view(p_txt, p_sHMsg);
            p_sHMsg->m_data = NULL; //reference to a "static" string
            delete p_sHMsg;
            break;
        default:
            //cerr << endl << " web_srv ERROR - UNKNOWN message type : " << p_rMsg->m_type;
            //p_conn->sendError(ST_501,msg_array[p_rMsg->m_type]); // Not Implemented"
            break;
        } // end switch

        delete p_rMsg;

    } // end while  -> server loop
}

int main(int argc, char **argv)
{
    //Host *p_host;
    char buff[100];
    int http_port = 0;
    int covise_port = 0;
    WebSrv mySrv;

    signal(SIGPIPE, SIG_IGN);

    cerr << endl << " WARNING - This is a prototype version  !!! " << endl;

    if (argc > 2)
    {
        cerr << endl << " Error : correct usage - \"web_srv [port]\" !!!";
        exit(0);
    }

    if ((home_path = getenv("SERVER_PATH")) == NULL)
    {
        cerr << endl << " WARNING: SERVER_PATH not defined !!!\n";
        home_path = new char[200];
        strcpy(home_path, ".");
    }

    if (argc == 2)
    {
        mySrv.set_aws_port(atoi(argv[1]));
        CClientConnection *aws_conn = new CClientConnection(NULL, mySrv.get_aws_port(), mySrv.get_id());
        if (aws_conn->get_id() <= 0)
        {
            delete aws_conn;
            cerr << endl << " ERROR : connection to aws renderer on port " << mySrv.get_aws_port() << " could not be established !!!" << endl;
            return -1;
        }
        mySrv.set_aws_conn(aws_conn);
        if (mySrv.open_channel(&http_port, CON_HTTP) < 0)
        {
            cerr << endl << " ERROR : open HTTP connection  could not be established !!!" << endl;
            exit(0);
        }
        if (mySrv.open_channel(&covise_port, CON_COVISE) < 0)
        {
            cerr << endl << " ERROR : open COVISE connection  could not be established !!!" << endl;
            exit(0);
        }

        sprintf(buff, "%d %d", covise_port, http_port);

        CMessage *p_msg = new CMessage;
        p_msg->m_type = C_PORT;
        p_msg->m_data = &buff[0];
        p_msg->m_length = strlen(p_msg->m_data) + 1;

        aws_conn->send_msg(p_msg);

        delete p_msg;

        cerr << endl << " Auxiliary web server started on http_port: " << http_port << " and covise_port: " << covise_port << "\n with server_path = " << home_path << endl;
    }
    else
    {
#ifdef _AIRBUS
        system("xset s 1");
        system("xset s activate");
#endif

        char *http_env = getenv("HTTP_PORT");
        char *covise_env = getenv("COVISE_PORT");

        if ((http_env == NULL) || (covise_env == NULL))
        {
            cerr << endl << " ERROR : HTTP_PORT or COVISE_PORT environment variables not set !!!" << endl;
            exit(-1);
        }

        http_port = atoi(http_env);
        covise_port = atoi(covise_env);

        if ((http_port == 0) || (covise_port == 0))
        {
            cerr << endl << " ERROR : HTTP_PORT or COVISE_PORT environment variables not set !!!" << endl;
            exit(-1);
        }

        int max_tries = 10;
        int num_tries = 0;

        while (num_tries < max_tries && mySrv.open_channel(http_port, CON_HTTP) < 0)
        {
            num_tries++;
            sleep(8);
        }
        if (num_tries == max_tries)
        {
            cerr << endl << " ERROR : open HTTP connection  could not be established !!!" << endl;
            exit(0);
        }
        if (mySrv.open_channel(covise_port, CON_COVISE) < 0)
        {
            cerr << endl << " ERROR : open COVISE connection  could not be established !!!" << endl;
            exit(0);
        }

        cerr << endl << " Web server started on http_port: " << http_port << " and covise_port: " << covise_port << "\n with server_path = " << home_path << endl;
    }

    //Web server loop

    mySrv.mainLoop();

    cerr << endl << "Web Server is exitting !!! " << endl;

    return 0;
}
