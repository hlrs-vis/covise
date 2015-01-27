/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef __hpux
#include <stdlib.h>
#endif
#include "web_connect.h"
#include "web_srv.h"
#include <sys/types.h>
#include <errno.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include <stdlib.h>

#include <sys/types.h>
#include <sys/socket.h>

/***********************************************************************\ 
 **                                                                     **
 **   Connection  classes Routines                 Version: 1.1         **
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
 **                         **
 **                  ConnectionList                                     **
 **                                                                     **
 **   Copyright (C) 2001               **
 **                                    **
 **                                    **
 **                                    **
 **                                                                     **
 **                                                                     **
 **   Author       :                                 **
 **                                                                     **
 **   History      :                                                    **
 **                                                   **
 **                 **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#if defined __sgi || defined __hpux || defined _AIX
//extern "C" bzero(void *b, int length);
#include <strings.h>
#endif

#undef SHOWMSG
#undef DEBUG

#define SWAP_N(data, len)                                                                                                               \
    {                                                                                                                                   \
        register int i, num = (len) >> 2;                                                                                               \
        register unsigned int val;                                                                                                      \
        register unsigned int *udata = (unsigned int *)(data);                                                                          \
        for (i = 0; i < num; i++)                                                                                                       \
        {                                                                                                                               \
            val = udata[i];                                                                                                             \
            udata[i] = ((val & 0x000000ff) << 24) | ((val & 0x0000ff00) << 8) | ((val & 0x00ff0000) >> 8) | ((val & 0xff000000) >> 24); \
        }                                                                                                                               \
    }

////////////////////////////////////////////////////
//                                                //
//        Connection                              //
//                                                //
////////////////////////////////////////////////////

void Connection::set_connid(char *sid)
{

    if (m_connid != NULL)
        delete[] m_connid;
    if (sid != NULL)
    {
        m_connid = new char[strlen(sid) + 1 + 1];
        strcpy(m_connid, "/");
        strcat(m_connid, sid);
        //cerr << endl << "$$$$ registered : " << m_connid << endl;
    }
    else
        m_connid = NULL;
}

void Connection::close()
{
    char tmpstr[255];

    if (sock)
    {
        sprintf(tmpstr, "close port %d", sock->get_port());
        print_comment(__LINE__, __FILE__, tmpstr);
        if (remove_socket)
        {
            remove_socket(sock->get_id());
            remove_socket = NULL;
        }
        delete sock;
        sock = NULL;
    }
}

int Connection::receive(void *buf, unsigned nbyte)
{
    int retval;
    if (!sock)
        return 0;
    retval = sock->Read(buf, nbyte);

    return retval;
}

int Connection::send(const void *buf, unsigned nbyte)
{
    if (!sock)
        return 0;
    return sock->write(buf, nbyte);
}

int Connection::check_for_input(float time)
{
    fd_set fdread;
    int i;
    struct timeval timeout;

    // initialize the bit fields according to the existing sockets
    // so far only reads are of interest

    timeout.tv_sec = (int)time;
    timeout.tv_usec = (int)((time - timeout.tv_sec) * 1000000);

    if (has_message())
        return 1;

    FD_ZERO(&fdread);
    FD_SET(sock->get_id(), &fdread);

#ifdef __hpux9
    i = select(sock->get_id() + 1, (int *)&fdread, NULL, NULL, &timeout);
#else
    i = select(sock->get_id() + 1, &fdread, NULL, NULL, &timeout);
#endif

    // find the connection that has the read attempt

    if (i > 0)
        return (1);
    else if (i < 0)
    {
        switch (errno)
        {
        case EBADF:
            print_comment(__LINE__, __FILE__, "invalid descriptor for select");
            break;
        case EINTR:
            print_comment(__LINE__, __FILE__, "early signal for select");
            break;
        case EINVAL:
            print_comment(__LINE__, __FILE__, "invalid time limit for select");
            break;
        }
    }
    return 0;
}

////////////////////////////////////////////////////
//                                                //
//        HConnection                             //
//                                                //
////////////////////////////////////////////////////

HMessage *HConnection::recv_msg(void)
{
    int tmp_read, ret;
    char *read_buf_ptr;
    long total, rest;
    HMessage *p_msg = NULL;

    if (!sock)
        return NULL;

    ////// this is sending to anything else than stdin/out

    p_msg = new HMessage;

    p_msg->m_length = 0;
    p_msg->m_type = MSG_EMPTY;
    p_msg->m_conn = this;

    /*  reading all in one call avoids the expensive read system call */

    //bytes_read = 0;

    if (m_read_buff == NULL)
    {
        m_buffer_size = READ_BUFFER_SIZE;
        m_read_buff = new char[m_buffer_size + 1];
        m_bytes_read = 0;
    }

    // processing data read in previous call
    if (m_bytes_read > 0)
    {
        //cerr << "\n$$$$$ processing data from prev call: " << m_read_buff << endl;
        ret = p_msg->process_headers(m_read_buff);
        if (ret < 0) // invalid message
        {
            cerr << endl << "$$$$$ - invalid message !!!!!\n";
            p_msg->m_type = SOCKET_CLOSED;
            delete[] m_read_buff;
            m_read_buff = NULL;
            message_to_do = 0;
            return p_msg;
        }
    }

    //reading start line & headers
    while (!p_msg->m_eoh)
    {
        if (m_bytes_read == m_buffer_size)
        {
            read_buf_ptr = m_read_buff;
            m_buffer_size += READ_BUFFER_SIZE;
            m_read_buff = new char[m_buffer_size + 1];
            memcpy(m_read_buff, read_buf_ptr, m_bytes_read);
        }
        tmp_read = sock->Read(m_read_buff + m_bytes_read, m_buffer_size - m_bytes_read);
        if (tmp_read <= 0)
        {
            //cerr << endl << " $$$$$ Error no bytes read for headers : " << bytes_read;
            p_msg->m_type = SOCKET_CLOSED;
            delete[] m_read_buff;
            m_read_buff = NULL;
            message_to_do = 0;
            return p_msg;
        }
        m_bytes_read += tmp_read;
        m_read_buff[m_bytes_read] = '\0';
        ret = p_msg->process_headers(m_read_buff);
        if (ret < 0) // invalid message
        {
            cerr << endl << "$$$$$ - invalid message !!!!!\n";
            p_msg->m_type = SOCKET_CLOSED;
            delete[] m_read_buff;
            m_read_buff = NULL;
            message_to_do = 0;
            return p_msg;
        }
    }

    total = p_msg->m_index + p_msg->m_length;

    if (p_msg->m_length > 0) // p_msg->m_eom = 1;
    {
        while (m_bytes_read < total)
        {
            if (m_bytes_read == m_buffer_size)
            {
                read_buf_ptr = m_read_buff;
                m_buffer_size += READ_BUFFER_SIZE;
                m_read_buff = new char[m_buffer_size + 1];
                memcpy(m_read_buff, read_buf_ptr, m_bytes_read);
            }
            tmp_read = sock->Read(m_read_buff + m_bytes_read, m_buffer_size - m_bytes_read);
            if (tmp_read <= 0)
            {
                cerr << endl << " $$$$$ Error no bytes read for entity: " << m_bytes_read;
                p_msg->m_type = SOCKET_CLOSED;
                delete[] m_read_buff;
                m_read_buff = NULL;
                message_to_do = 0;
                return p_msg;
            }
            m_bytes_read += tmp_read;
        }
        m_read_buff[m_bytes_read] = '\0';
    }

    if (m_bytes_read < total)
    {
        cerr << endl << " $$$$$ Error : invalid length= " << m_bytes_read << " total= " << total << " m_length= " << p_msg->m_length << " !!!! \n";
        p_msg->m_type = SOCKET_CLOSED;
        delete[] m_read_buff;
        m_read_buff = NULL;
        message_to_do = 0;
        return p_msg;
    }

    if (p_msg->m_length > 0)
    {
        read_buf_ptr = &m_read_buff[p_msg->m_index];
        p_msg->m_data = new char[strlen(read_buf_ptr) + 1];
        strcpy(p_msg->m_data, read_buf_ptr);
    }

    if (m_bytes_read > total)
    {
        read_buf_ptr = m_read_buff;
        rest = m_bytes_read - total;
        m_buffer_size = rest + READ_BUFFER_SIZE;
        m_read_buff = new char[m_buffer_size + 1];
        memcpy(m_read_buff, read_buf_ptr + total, rest);
        m_read_buff[rest] = '\0';
        m_bytes_read = rest;
        delete[] read_buf_ptr;
        message_to_do = 1;
    }
    else
    {
        message_to_do = 0;
        delete[] m_read_buff;
        m_read_buff = NULL;
    }

    return p_msg;
}

int HConnection::send_file(FILE *fd)
{
    char buff[WRITE_BUFFER_SIZE + 1];
    int nb, no, total = 0;

    while (!feof(fd))
    {
        no = fread(buff, 1, WRITE_BUFFER_SIZE, fd);
        if (no < 0)
        {
            return -1;
        }
        else
        {
            nb = sock->write(buff, no);
            total += nb;
        }
    }
    return total;
}

int HConnection::send_file(char *name)
{
    int ret;
    FILE *fd;

    fd = fopen(name, "r");

    if (fd == NULL)
    {
        cerr << "$$$$ Error web_srv send_file(...): could not open web file "
             << name << " !!!!! " << endl;
        perror("Error : ");
        return -1;
    }

    ret = send_file(fd);
    if (ret < 0)
    {
        cerr << "$$$$ Error web_srv send_file(...): could not read web file "
             << name << " !!!!! " << endl;
        perror("Error : ");
        return -1;
    }
    fclose(fd);

    return ret;
}

int HConnection::send_msg(Message *msg)
{
    int tmp_val = 0, retval = 0, bytes_written = 0;
    int i, l, div, rest;
    char write_buff[WRITE_BUFFER_SIZE + 5];
    char *p_buff;
    HMessage *p_sMsg;

    if (!sock)
        return 0;

    p_sMsg = (HMessage *)msg;
    p_sMsg->add_header(CONNECTION, "close");
    p_sMsg->add_header(CACHE_CONTROL, "no-cache");
    p_sMsg->add_header(PRAGMA, "no-cache");

    cerr << endl << " $$$$ sending the message : -------------------\n";
    p_sMsg->print();
    cerr << endl << "-------------------------------" << endl;

    sprintf(write_buff, "%s\r\n", p_sMsg->m_start);
    bytes_written += strlen(p_sMsg->m_start) + 2;

    // write standard headers
    for (i = 0; i < MAX_HEADERS; i++)
    {
        if (p_sMsg->m_headers[i] != NULL)
        {
            l = strlen(p_sMsg->m_headers[i]->m_name) + 1 + strlen(p_sMsg->m_headers[i]->m_value) + 2;
            if ((bytes_written + l) > WRITE_BUFFER_SIZE)
            {
                tmp_val = sock->write(write_buff, bytes_written);
                if (tmp_val == COVISE_SOCKET_INVALID)
                    return -1;
                retval += tmp_val;
                sprintf(write_buff, "%s:%s\r\n", p_sMsg->m_headers[i]->m_name, p_sMsg->m_headers[i]->m_value);
                bytes_written = l;
            }
            else
            {
                p_buff = &write_buff[bytes_written];
                sprintf(p_buff, "%s:%s\r\n", p_sMsg->m_headers[i]->m_name, p_sMsg->m_headers[i]->m_value);
                bytes_written += l;
            }
        }
    }
    /*
      // write unknown headers

      p_sMsg->m_unknownHs->reset();
      while((tmp = p_sMsg->m_unknownHs->next())!=NULL)
      {
         l = strlen(tmp->m_name)+1+strlen(tmp->m_value)+2;
         if((bytes_written+l)>WRITE_BUFFER_SIZE)
         {
            tmp_val = sock->write(write_buff,bytes_written);
            retval += tmp_val;
   sprintf(write_buff,"%s:%s\r\n",tmp->m_name,tmp->m_value);
   bytes_written = l;
   }
   else
   {
   p_buff = &write_buff[bytes_written];
   sprintf(p_buff,"%s:%s\r\n",tmp->m_name,tmp->m_value);
   bytes_written += l;
   }

   }
   */
    p_buff = &write_buff[bytes_written];
    strcpy(p_buff, "\r\n"); // empty line
    bytes_written += 2;
    tmp_val = sock->write(write_buff, bytes_written);
    if (tmp_val == COVISE_SOCKET_INVALID)
        return -1;
    retval += tmp_val;

    bytes_written = 0;

    // write content
    if (!p_sMsg->m_send_content)
        return retval; // HEAD method

    if (p_sMsg->m_length > 0)
    {
        if (p_sMsg->m_indirect)
        {
            tmp_val = send_file(p_sMsg->m_fd);
            if (tmp_val < 0)
                return -1; // error reading file
            retval += tmp_val;
        } // m_indirect
        else
        {
            div = p_sMsg->m_length / WRITE_BUFFER_SIZE;
            rest = p_sMsg->m_length % WRITE_BUFFER_SIZE;

            //cerr << endl << " div= "<< div << " rest= " << rest;

            for (i = 0; i < div; i++)
            {
                p_buff = &p_sMsg->m_data[bytes_written];
                tmp_val = sock->write(p_buff, WRITE_BUFFER_SIZE);
                if (tmp_val == COVISE_SOCKET_INVALID)
                    return -1;
                retval += tmp_val;
                bytes_written += WRITE_BUFFER_SIZE;
            }

            if (rest)
            {
                p_buff = &p_sMsg->m_data[bytes_written];
                tmp_val = sock->write(p_buff, rest);
                if (tmp_val == COVISE_SOCKET_INVALID)
                    return -1;
                retval += tmp_val;
            }
        } // !m_indirect
    }

    return retval;
}

int HConnection::send_http_error(msg_type code, char *arg, int get_flag)
{
    HMessage *p_sMsg;
    char buff[2056];
    char *p_tmp;

    const char close_window[] = "<script>\n function closing(){ window.opener=self; self.close(); } </script> \n \
                             <meta http-equiv=refresh content=\"3; URL=javascript:void(closing())\">";
    p_sMsg = new HMessage(code);

    sprintf(buff, "<HTML><HEAD>%s<TITLE>%s %s</TITLE></HEAD>\n<BODY><H2>%s %s</H2>\n", close_window, msg_array[code], msg_txt[code], msg_array[code], msg_txt[code]);

    p_tmp = new char[strlen(buff) + 1];
    strcpy(p_tmp, buff);

    // add title
    p_sMsg->add_body(p_tmp, strlen(buff), "en", M_TEXT_HTML);

    if (msg_form[code] != NULL)
    {
        if (strchr(msg_form[code], '%'))
            sprintf(buff, msg_form[code], arg);
        else
            sprintf(buff, msg_form[code]);
        p_sMsg->add_body(buff); // add explications
    }

#ifndef _AIRBUS
    sprintf(buff, "<HR>\n<ADDRESS><A HREF=\"%s\">%s</A></ADDRESS>\n</BODY></HTML>\n", SERVER_ADDRESS, SERVER_SOFTWARE);
    p_sMsg->add_body(buff); // add response tail
#endif

    p_sMsg->m_send_content = get_flag;
    send_msg(p_sMsg);
    delete p_sMsg;

    return 0;
}

void HConnection::sendError(msg_type mt, char *txt)
{

    send_http_error(mt, txt);
}

////////////////////////////////////////////////////
//                                                //
//        CConnection                             //
//                                                //
////////////////////////////////////////////////////

CMessage *CConnection::recv_msg(void)
{
    int bytes_read, bytes_to_read, tmp_read;
    char *read_buf_ptr;
    int *int_read_buf;
    int data_length;
    char *read_data;
    CMessage *msg = NULL;

    if (!sock)
        return 0;

    msg = new CMessage;

    msg->m_sender = msg->m_length = 0;
    msg->m_send_type = UNDEFINED;
    msg->m_type = MSG_EMPTY;
    msg->m_conn = this;
    message_to_do = 0;

    /*  reading all in one call avoids the expensive read system call */

    while (bytes_to_process < 16)
    {
        tmp_read = sock->Read(read_buf + bytes_to_process, READ_BUFFER_SIZE - bytes_to_process);
        if (tmp_read == 0)
            return msg;
        if (tmp_read < 0)
        {
            msg->m_type = C_SOCKET_CLOSED;
            msg->m_conn = this;
            return msg; //tmp_read;
        }
        bytes_to_process += tmp_read;
    }

    read_buf_ptr = read_buf;

    while (1)
    {
        int_read_buf = (int *)read_buf_ptr;

        //cerr << endl << " !!! INITIALY +++ sender: " << int_read_buf[0] <<
        //                " +++ send_type: " << int_read_buf[1] <<
        //                " +++ msg_type: " << int_read_buf[2] <<
        //   " +++ length: " << int_read_buf[3] << endl ;

        if ((unsigned int)int_read_buf[2] > 1000) // dmg takes care about byteswpping, controller not
        {
            SWAP_N((unsigned int *)int_read_buf, 16);
            //cerr << endl << "=========== SWAP_N =============\n";
            //cerr << endl << " !!! AFTER SWAP +++ sender: " << int_read_buf[0] <<
            //              " +++ send_type: " << int_read_buf[1] <<
            //              " +++ msg_type: " << int_read_buf[2] <<
            // " +++ length: " << int_read_buf[3] << endl ;
        }

        msg->m_sender = int_read_buf[0];
        msg->m_send_type = sender_type(int_read_buf[1]);
        msg->m_type = msg_type(int_read_buf[2] + LAST_WMESSAGE + 2);
        msg->m_length = int_read_buf[3];

        bytes_to_process -= 4 * SIZEOF_IEEE_INT;
        read_buf_ptr += 4 * SIZEOF_IEEE_INT;
        if (msg->m_length > 0) // if msg->length == 0, no data will be received
        {
            if (msg->m_data)
            {
                delete[] msg -> m_data;
                msg->m_data = NULL;
            }
            // bring message data space to 16 byte alignment
            data_length = msg->m_length; // + ((msg->m_length % 16 != 0) * (16 - msg->m_length % 16));
            msg->m_data = new char[data_length + 20];
            //    cerr << "got: " << data_length << "   " << msg->m_length << endl;
            if (msg->m_length > bytes_to_process)
            {
                //cerr << endl << "$$$$$ Bytes available =" << bytes_to_process << " Bytes needed = " << msg->m_length;
                bytes_read = bytes_to_process;
                if (bytes_read != 0)
                    memmove(msg->m_data, read_buf_ptr, bytes_read);
                bytes_to_process = 0;
                bytes_to_read = msg->m_length - bytes_read;
                read_data = &msg->m_data[bytes_read];
                while (bytes_read < msg->m_length)
                {

                    tmp_read = sock->read(read_data, bytes_to_read);
                    // covise_time->mark(__LINE__, "nach weiterem sock->read(read_buf)");
                    bytes_read += tmp_read;
                    bytes_to_read -= tmp_read;
                    read_data = &msg->m_data[bytes_read];
                }
                return msg; //msg->m_length;
            }
            else if (msg->m_length < bytes_to_process)
            {
                //cerr << endl << "$$$$$ Bytes available :" << bytes_to_process;
                memcpy(msg->m_data, read_buf_ptr, msg->m_length);
                bytes_to_process -= msg->m_length;
                read_buf_ptr += msg->m_length;
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;

                //while(bytes_to_process < 16)
                //{
                //  bytes_to_process +=  sock->read(&read_buf_ptr[bytes_to_process],READ_BUFFER_SIZE - bytes_to_process);
                //}
                message_to_do = 1;
                //cerr << endl << "$$$$$ Bytes left :" << bytes_to_process << endl;
                //	        covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg; //msg->m_length;
            }
            else
            {
                memcpy(msg->m_data, read_buf_ptr, bytes_to_process);
                bytes_to_process = 0;
                //	            covise_time->mark(__LINE__, "    recv_msg: Ende");
                return msg; //msg->m_length;
            }
        }
        else //msg->length == 0, no data will be received
        {
            if (msg->m_data)
            {
                delete[] msg -> m_data;
                msg->m_data = NULL;
            }
            if (msg->m_length < bytes_to_process)
            {
                memmove(read_buf, read_buf_ptr, bytes_to_process);
                read_buf_ptr = read_buf;
                //while(bytes_to_process < 16)
                //{
                //   bytes_to_process +=  sock->read(&read_buf_ptr[bytes_to_process],READ_BUFFER_SIZE - bytes_to_process);
                //}
                message_to_do = 1;
            }
            return msg; //0;
        }
    }
}

int CConnection::send_msg(Message *msg)
{
    int retval = 0, tmp_bytes_written;
    char write_buf[WRITE_BUFFER_SIZE];
    int *write_buf_int;
    CMessage *p_sMsg;

    if (!sock)
        return 0;

    p_sMsg = (CMessage *)msg;

    write_buf_int = (int *)write_buf;
    write_buf_int[0] = sender_id;
    write_buf_int[1] = USERINTERFACE;
    write_buf_int[2] = p_sMsg->m_type - LAST_WMESSAGE - 2;
    write_buf_int[3] = p_sMsg->m_length;
#ifdef BYTESWAP
    SWAP_N((unsigned int *)write_buf_int, 16);
#endif
    if (p_sMsg->m_length == 0)
        retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT);
    else
    {
        if (p_sMsg->m_length < WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT)
        {
            memcpy(&write_buf[4 * SIZEOF_IEEE_INT], p_sMsg->m_data, p_sMsg->m_length);
            retval = sock->write(write_buf, 4 * SIZEOF_IEEE_INT + p_sMsg->m_length);
        }
        else
        {
            memcpy(&write_buf[16], p_sMsg->m_data, WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT);
            retval = sock->write(write_buf, WRITE_BUFFER_SIZE);
            tmp_bytes_written = sock->write(&p_sMsg->m_data[WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT], p_sMsg->m_length - (WRITE_BUFFER_SIZE - 4 * SIZEOF_IEEE_INT));
            if (tmp_bytes_written == COVISE_SOCKET_INVALID)
                return COVISE_SOCKET_INVALID;
            else
                retval += tmp_bytes_written;
        }
    }
    return retval;
}

void CConnection::inc_ref(void)
{
    m_view_ref++;
    if (m_view_ref == 1)
    {
        // send  SET_ACCESS 1 to the displayed VRML_renderer
        CMessage *p_msg = new CMessage;
        p_msg->m_type = C_SET_ACCESS;
        p_msg->m_data = "1";
        p_msg->m_length = strlen(p_msg->m_data) + 1;

        send_msg(p_msg);

        p_msg->m_data = NULL; //volatile reference
        delete p_msg;
    }
}

void CConnection::dec_ref(void)
{
    if (m_view_ref > 0)
        m_view_ref--;
    if (m_view_ref == 0)
    {
        // send  SET_ACCESS 0 to the displayed VRML_renderer
        CMessage *p_msg = new CMessage;
        p_msg->m_type = C_SET_ACCESS;
        p_msg->m_data = "0";
        p_msg->m_length = strlen(p_msg->m_data) + 1;

        send_msg(p_msg);

        p_msg->m_data = NULL; //volatile reference
        delete p_msg;
    }
}

////////////////////////////////////////////////////
//                                                //
//        HServerConnection                       //
//                                                //
////////////////////////////////////////////////////

HServerConnection::HServerConnection(int p, int id)
{
    sender_id = id;
    sock = new Socket(p);
    m_status = sock->get_id();
    port = p;
}

HServerConnection::HServerConnection(int *p, int id)
{
    sender_id = id;
    sock = new Socket(p);
    m_status = sock->get_id();
    port = *p;
}

int HServerConnection::accept(int wait)
{
    if (sock->accept(wait) != 0)
        return -1;
    //get_dataformat();
    return 0;
}

HServerConnection *HServerConnection::spawn_connection()
{
    HServerConnection *new_conn;
    Socket *tmp_sock;

    tmp_sock = sock->copy_and_accept();
    new_conn = new HServerConnection(tmp_sock);
    new_conn->port = port;
    new_conn->sender_id = sender_id;
    //new_conn->get_dataformat();
    return new_conn;
}

////////////////////////////////////////////////////
//                                                //
//        CServerConnection                       //
//                                                //
////////////////////////////////////////////////////

CServerConnection::CServerConnection(int p, int id)
{
    sender_id = id;
    sock = new Socket(p);
    m_status = sock->get_id();
    port = p;
}

CServerConnection::CServerConnection(int *p, int id)
{
    sender_id = id;
    sock = new Socket(p);
    m_status = sock->get_id();
    port = *p;
}

void CServerConnection::get_dataformat()
{
    char dataformat;

    if (sock->write(&df_local_machine, 1) == COVISE_SOCKET_INVALID)
        print_error(__LINE__, __FILE__,
                    "invalid socket in ServerConnection::accept");
    sock->read(&dataformat, 1);
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
}

int CServerConnection::accept(int wait)
{
    if (sock->accept(wait) != 0)
        return -1;
    get_dataformat();
    return 0;
}

CServerConnection *CServerConnection::spawn_connection()
{
    CServerConnection *new_conn;
    Socket *tmp_sock;

    tmp_sock = sock->copy_and_accept();
    new_conn = new CServerConnection(tmp_sock);
    new_conn->port = port;
    new_conn->sender_id = sender_id;
    new_conn->get_dataformat();
    return new_conn;
}

////////////////////////////////////////////////////
//                                                //
//        CClientConnection                        //
//                                                //
////////////////////////////////////////////////////

CClientConnection::CClientConnection(Host *h, int p, int id, int retries)
{
    char dataformat;

    lhost = NULL;
    if (h) // host is not local
        host = h;
    else // host is local (usually DataManagerConnection uses this)
        host = lhost = new Host("localhost");
    port = p;
    sender_id = id;
    sock = new Socket(host, port, retries);

    if (get_id() == -1)
        return; // connection not established
    sock->read(&dataformat, 1);
    if (dataformat != df_local_machine)
        if (df_local_machine != DF_IEEE)
            convert_to = DF_IEEE;
    if (sock->write(&df_local_machine, 1) == COVISE_SOCKET_INVALID)
        print_error(__LINE__, __FILE__, "invalid socket in new ClientConnection");
}

////////////////////////////////////////////////////
//                                                //
//        ConnectionList                          //
//                                                //
////////////////////////////////////////////////////

ConnectionList::ConnectionList()
{
    m_connlist = new Liste<Connection>(1);
    m_channels = new Liste<Connection>(1);

    m_maxfd = 0;
    FD_ZERO(&m_fdvar); // the field for the select call is initiallized
    return;
}

ConnectionList::ConnectionList(Connection *o_s)
{
    m_connlist = new Liste<Connection>(1);
    m_channels = new Liste<Connection>(1);
    FD_ZERO(&m_fdvar); // the field for the select call is initiallized
    o_s->listen();
    add_ch(o_s);

    return;
}

ConnectionList::~ConnectionList()
{
    //Connection *ptr;
    /*
      m_connlist->reset();
      while(ptr = m_connlist->next())
      {
         //ptr->close_inform();
         delete ptr;
      }
   */

    delete m_connlist;

    /*
      m_channels->reset();
      while(ptr = m_channels->next())
      {
         //ptr->close_inform();
         delete ptr;
      }
   */
    delete m_channels;

    return;
}

Connection *ConnectionList::get_conn(char *sid)
{
    Connection *ptr;

    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        if (strcmp(ptr->get_connid(), sid) == 0)
            return ptr;
    }
    return NULL;
}

void ConnectionList::add(Connection *c) // add a connection and update the
{ // field for the select call
    m_connlist->add(c);
    if (c->get_id() > m_maxfd)
        m_maxfd = c->get_id();
    FD_SET(c->get_id(), &m_fdvar);
    return;
}

void ConnectionList::remove(Connection *c) // remove a connection and update
{ // the field for the select call
    m_connlist->remove(c);
    FD_CLR(c->get_id(), &m_fdvar);
    return;
}

void ConnectionList::add_ch(Connection *c) // add a channel and update the
{ // field for the select call
    m_channels->add(c);
    if (c->get_id() > m_maxfd)
        m_maxfd = c->get_id();
    FD_SET(c->get_id(), &m_fdvar);
    return;
}

void ConnectionList::remove_ch(Connection *c) // remove a channel and update
{ // the field for the select call
    m_channels->remove(c);
    FD_CLR(c->get_id(), &m_fdvar);
    return;
}

#if 0
//  Check whether PPID==1 or no sockets left: prevent hanging
static void checkPPIDandFD(fd_set &actSet, int maxfd)
{
   if (getppid()==1)
   {
      cerr << "Process " << getpid()
         << " quit because parent died" << endl;
      exit(0);
   }

   int j,numFD=0;
   for(j = 0;j <= maxfd;j++)
      if(FD_ISSET(j,&actSet))
         numFD++;

   if (numFD==0)
   {
      cerr << "Process " << getpid()
         << " quit because all connections lost" << endl;
      exit(0);
   }

}
#endif

/// Wait for input infinitely - replaced by loop with timeouts
/// - check every 10 sec against hang aw 04/2000
Connection *
ConnectionList::wait_for_input()
{
    Connection *found = NULL;
    do
        found = check_for_input(10.0);
    while (!found);
    return found;
}

Connection *
ConnectionList::check_for_input(float time)
{
    fd_set fdread;
    int i, j;
    Connection *ptr;
    Connection *p_ch;
    struct timeval timeout;

    // if we already have a pending message, we return it
    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        if (ptr->has_message())
            return ptr;
    }

    // initialize timeout field
    timeout.tv_sec = (int)time;
    timeout.tv_usec = (int)((time - timeout.tv_sec) * 1000000);

    // initialize the bit fields according to the existing sockets
    // so far only reads are of interest
    FD_ZERO(&fdread);
    for (j = 0; j <= m_maxfd; j++)
        if (FD_ISSET(j, &m_fdvar))
            FD_SET(j, &fdread);

// wait for the next read attempt on one of the sockets

#ifdef __hpux9
    i = select(m_maxfd + 1, (int *)&fdread, NULL, NULL, &timeout);
#else
    i = select(m_maxfd + 1, &fdread, NULL, NULL, &timeout);
#endif

    //	print_comment(__LINE__, __FILE__, "something happened");

    /* 
   // nothing? this might be a hanger ... better check it!
   if (i==0)
   checkPPIDandFD(m_fdvar,m_maxfd);
   */
    // find the connection that has the read attempt
    if (i > 0)
    {
        // search if it's a channel
        m_channels->reset();
        while ((ptr = m_channels->next()))
        {
            if (FD_ISSET(ptr->get_id(), &fdread))
            {
                p_ch = (Connection *)ptr->spawn_connection();
                if (p_ch)
                {
                    this->add(p_ch);
                    return NULL;
                }
                else
                    cerr << " Error: channel cannot be established !!";
            }
        }
        // search if it's an established connection
        m_connlist->reset();
        while ((ptr = m_connlist->next()))
        {
            if (FD_ISSET(ptr->get_id(), &fdread))
                return ptr;
        }
    }
    else if (i < 0)
    {
        switch (errno)
        {
        case EBADF:
            print_comment(__LINE__, __FILE__, "invalid descriptor for select");
            break;
        case EINTR:
            print_comment(__LINE__, __FILE__, "early signal for select");
            break;
        case EINVAL:
            print_comment(__LINE__, __FILE__, "invalid time limit for select");
            break;
        }
    }
    return NULL;
}

char *ConnectionList::get_registered_users(char *safe)
{

    Connection *ptr;
    int length = 1000;
    int crtl = 3;
    int l;
    char *buff;
    char *p_buff;
    char *tmp;

    buff = new char[length + 1];
    strcpy(buff, " 1 ");
    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        if (ptr->get_type() == CON_COVISE)
        {
            tmp = ptr->get_connid();
            if (tmp != NULL)
            {
                strcpy(safe, tmp);
                l = strlen(tmp) + 1;
                if ((crtl + l) > length)
                {
                    length += 1000;
                    p_buff = new char[length + 1];
                    strcpy(p_buff, buff);
                    delete[] buff;
                    buff = p_buff;
                }
                strcat(buff, tmp);
                strcat(buff, "\n");
                crtl += l;
            }
        }
    }
    if (crtl == 3)
        strcpy(buff, " 1 No VRML_Renderer has been registered !\n");
    return buff;
}

char *ConnectionList::get_last_wrl(void)
{
    Connection *ptr;
    char *tmp;
    char *buff;
    int found = 0;

    buff = new char[100];

    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        tmp = ptr->get_connid();
        if (tmp != NULL)
        {
            strcpy(buff, tmp);
            found = 1;
        }
    }

    if (found)
        return buff;
    else
    {
        strcpy(buff, "/default.wrl");
        return buff;
    }
}

int ConnectionList::add_dynamic_view(char *id)
{
    Connection *ptr;
    int found;
    char cid[50];

    if (id == NULL)
        return 0;

    strcpy(cid, &id[4]); //skip "/app"

    m_connlist->reset();
    found = 0;
    while ((ptr = m_connlist->next()) && (!found))
    {
        if (strcmp(cid, ptr->get_connid()) == 0)
        {
            found = 1;
            ((CConnection *)ptr)->inc_ref();
        }
    }
    return found;
}

int ConnectionList::remove_dynamic_view(char *id)
{
    Connection *ptr;
    int found;
    char cid[50];

    if (id == NULL)
        return 0;

    strcpy(cid, &id[4]); //skip "/app"

    //cerr << endl << "$$$$$$ remove_dynamic_view : " << cid << endl;

    m_connlist->reset();
    found = 0;
    while ((ptr = m_connlist->next()) && (!found))
    {
        if (strcmp(cid, ptr->get_connid()) == 0)
        {
            found = 1;
            ((CConnection *)ptr)->dec_ref();
        }
    }
    return found;
}

int ConnectionList::broadcast_usr(char *act, char *usr)
{
    Connection *ptr;
    char *new_usr;

    new_usr = new char[strlen(act) + strlen(usr) + 2];

    strcpy(new_usr, act);
    strcat(new_usr, usr);
    strcat(new_usr, "\n");
    HMessage *p_sHMsg;
    p_sHMsg = new HMessage(ST_200); // OK
    p_sHMsg->add_header(CONNECTION, "keep-alive");
    p_sHMsg->add_body(new_usr, strlen(new_usr), "en", M_REGISTERED_USERS);
    p_sHMsg->m_send_content = 1;

    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        if (ptr->is_dynamic_usr())
        {
            ptr->send_msg(p_sHMsg);
        }
    }

    delete p_sHMsg;
    return 1;
}

int ConnectionList::broadcast_view(char *client_id, Message *p_Msg)
{
    Connection *ptr;
    char hid[50];

    strcpy(hid, "/app"); //add "/app..."
    strcat(hid, client_id);

    m_connlist->reset();
    while ((ptr = m_connlist->next()))
    {
        if (ptr->is_dynamic_view() && (strcmp(hid, ptr->get_connid()) == 0))
        {
            ptr->send_msg(p_Msg);
        }
    }

    return 1;
}
