/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define DEFINE_HTTP_MSGS
#include "web_msg.h"
#include "web_srv.h"
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
/***********************************************************************\ 
 **                                                                     **
 **   Message classes Routines                     Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Message                                **
 **                                                                     **
 **   Copyright (C) 2001     by                  **
 **                                              **
 **                                              **
 **                                              **
 **                                                                     **
 **                                                                     **
 **   Author       :                                   **
 **                                                                     **
 **   History      :                                                    **
 **                                                  **
 **                    **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#undef DEBUG

int HMessage::process_headers(char *buff)
{
    char *buf;
    char *next_l;
    char *tmp;
    //char end;
    int l, type;

    buf = &buff[m_index];

    if (m_index == 0) // start line
    {
        next_l = strchr(buf, '\n');
        if (next_l)
        {
            next_l[0] = '\0'; // over '\n'
            l = strlen(buf);
            m_start = new char[l + 1];
            strcpy(m_start, buf);

            // process request-line
            tmp = strtok(buf, " "); // get method
            if (tmp == NULL)
                return -1; // invalid msg !!!
            m_method = new char[strlen(tmp) + 1];
            strcpy(m_method, tmp);
            for (type = OPTIONS; type <= CONNECT; type++)
            {
                if (strcasecmp(msg_array[type], tmp) == 0)
                    break;
            }

            if (type <= CONNECT)
                m_type = (msg_type)type;
            else
                return -1; //m_type = MSG_UNKNOWN;

            tmp = strtok(NULL, " "); // get URI
            if (tmp == NULL)
                return -1; // invalid msg !!!
            m_URI = new char[strlen(tmp) + 1];
            strcpy(m_URI, tmp);

            tmp = strtok(NULL, "\r"); // get version
            if (tmp == NULL)
                return -1; // invalid msg !!!
            m_version = new char[strlen(tmp) + 1];
            strcpy(m_version, tmp);

            //msg->process_start_line();
            m_index += l + 1; // skip token & '\0'
            buf = &buff[m_index];
        }
        else
            return 1; // not enough bytes_read
    }

    while ((next_l = strchr(buf, '\n')) != NULL)
    {
        next_l[0] = '\0'; // over '\n'
        l = strlen(buf);

        if (buf[0] == '\r') // empty line - eof headers
        {
            m_eoh = 1;
            m_index += 2; // skip '\r\0'
            return 1; // continue
        }

        next_l = strchr(buf, ':');
        if (next_l)
            next_l[0] = '\0'; // over ':'
        else
            return -1; // invalid msg !!!

        //      cerr << endl << " $$$ Adding header : " << buf << ":" << &next_l[1];

        add_header(buf, &next_l[1]);

        m_index += l + 1; // skip token & '\0'
        buf = &buff[m_index];
    }
    return 1; // continue
}

void HMessage::add_start(char *start_line)
{
    m_start = start_line;
}

void HMessage::post_addH(header_type type)
{
    switch (type)
    {
    case CONTENT_LENGTH:
        m_length = atoi(m_headers[type]->m_value);
        break;
    default:
        break;
    }
    return;
}

int HMessage::add_header(header_type type, Header *hd)
{
    if (type != UNKNOWN)
        m_headers[type] = hd;
    //else m_unknownHs->add(hd);

    post_addH(type);

    return 0;
}

int HMessage::add_header(header_type type, char *value)
{
    Header *hd;

    hd = new Header();
    hd->Set(header_array[type], value);

    add_header(type, hd);

    return 0;
}

int HMessage::add_header(char *name, char *value)
{
    int type;
    Header *hd;

    for (type = CACHE_CONTROL; type < MAX_HEADERS; type++)
    {
        if (strcasecmp(header_array[type], name) == 0)
            break;
    }

    hd = new Header(name, value);

    add_header((header_type)type, hd);

    return 0;
}

void HMessage::add_body(char *content, int length, char *lang, int type)
{
    char length_s[100];

    if (m_data != NULL)
        delete[] m_data;
    if (m_headers[CONTENT_LANGUAGE] != NULL)
    {
        delete m_headers[CONTENT_LANGUAGE];
        m_headers[CONTENT_LANGUAGE] = NULL;
    }
    if (m_headers[CONTENT_TYPE] != NULL)
    {
        delete m_headers[CONTENT_TYPE];
        m_headers[CONTENT_TYPE] = NULL;
    }
    if (m_headers[CONTENT_LENGTH] != NULL)
    {
        delete m_headers[CONTENT_LENGTH];
        m_headers[CONTENT_LENGTH] = NULL;
    }

    m_data = content;
    m_length = length;

    sprintf(length_s, "%d", m_length);
    add_header(CONTENT_LENGTH, length_s);

    add_header(CONTENT_LANGUAGE, lang);
    add_header(CONTENT_TYPE, URI_type_array[type]);
}

void HMessage::add_body(char *content)
{
    char length_s[10];
    char *p_tmp;
    int add_l;

    if (m_data == NULL)
    {
        cerr << endl << "Error: first add body with previous function  !!!";
        //add_body(content,"en","text/html");
        return;
    }

    if (m_headers[CONTENT_LENGTH] != NULL)
    {
        delete m_headers[CONTENT_LENGTH];
        m_headers[CONTENT_LENGTH] = NULL;
    }

    add_l = strlen(content);

    p_tmp = new char[m_length + add_l + 1];

    memcpy(p_tmp, m_data, m_length);
    delete[] m_data;

    memcpy(&p_tmp[m_length], content, add_l + 1);

    m_data = p_tmp;
    m_length = m_length + add_l;

    sprintf(length_s, "%d", m_length);
    add_header(CONTENT_LENGTH, length_s);
}

void HMessage::process_start_line(void)
{
    char *tmp;
    int type;

    if (m_start == NULL)
        return;

    tmp = strtok(m_start, " "); // get method
    m_method = new char[strlen(tmp) + 1];
    strcpy(m_method, tmp);

    for (type = OPTIONS; type <= CONNECT; type++)
    {
        if (strcasecmp(msg_array[type], tmp) == 0)
            break;
    }

    if (type <= CONNECT)
        m_type = (msg_type)type;
    else
        m_type = MSG_UNKNOWN;

    tmp = strtok(NULL, "\r"); // get URI

    m_URI = new char[strlen(tmp) + 1];
    strcpy(m_URI, tmp);

    tmp = strtok(NULL, "\r"); // get version

    m_version = new char[strlen(tmp) + 1];
    strcpy(m_version, tmp);
}

Header *HMessage::get_header(header_type type)
{
    if ((type >= 0) && (type < UNKNOWN))
        return m_headers[type];

    return NULL;
}

void HMessage::print(void)
{
    int i;
    char *content;

    cerr << " Message type : " << msg_array[m_type] << endl;
    cerr << m_start << endl;
    for (i = 0; i < MAX_HEADERS; i++)
    {
        if (m_headers[i] != NULL)
            m_headers[i]->print();
    }
    //m_unknownHs->reset();
    //while((tmp = m_unknownHs->next())!=NULL) tmp->print();

    if (m_data)
    {
        content = new char[m_length + 1];
        memcpy(content, m_data, m_length);
        content[m_length] = '\0';
        if (m_length > 1000)
            cerr << endl << " Content will not be printed !!!!\n";
        else
            cerr << content;
    }
    else
        cerr << " NULL content ";
}

void CMessage::print(void)
{
    cerr << " Message_type: " << m_type << " from Sender: " << m_sender << " Send_type: " << m_send_type << endl << "-------------------------------\n";

    //cerr << m_data;
    if (m_data)
        cerr << endl << " Content will not be printed !!!!\n";
    else
        cerr << " NULL content ";
}
