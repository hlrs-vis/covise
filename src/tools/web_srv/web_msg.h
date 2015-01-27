/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_MESSAGE_H
#define EC_MESSAGE_H
#include <iostream>
#include <covise/covise_global.h>
#include <util/coString.h>
#include "header.h"
#include "web_list.h"
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

#ifndef _HAS_UINT
#ifndef _WIN32
#include <sys/types.h>
#ifdef __linux__
typedef u_int32_t uint32;
#else
typedef uint32_t uint32;
#endif
#else
typedef unsigned long int uint32;
#endif
#endif
/***********************************************************************\ 
 **                                                                     **
 **   Message classes                              Version: 1.1         **
 **                                                                     **
 **                                                                     **
 **   Description  : The basic message structure as well as ways to     **
 **                  initialize messages easily are provided.           **
 **                  Subclasses for special types of messages           **
 **                  can be introduced.                                 **
 **                                                                     **
 **   Classes      : Message                              **
 **                                                                     **
 **   Copyright (C) 2001     by Vircinity                 **
 **                                               **
 **                             NöbelStr 15                          **
 **                             7000 Stuttgart                       **
 **                  HOSTID                                             **
 **                                                                     **
 **   Author       :                                  **
 **                                                                     **
 **   History      :                                                    **
 **                                                    **
 **                        **
 **                        **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

class Connection;

// IDs for all messages that go between processes are fixed here

enum msg_type
{
    MSG_EMPTY = 0, //  0
    OPTIONS, //  1      --- first request message
    GET, //  2
    HEAD, //  3
    POST, //  4
    PUT, //  5
    DELETE, //  6
    TRACE, //  7
    CONNECT, //  8      --- last request message
    ST_100, //  9   // "Continue"
    ST_101, // 10   // "Switching Protocols"
    ST_200, // 11   // "OK"
    ST_201, // 12   // "Created"
    ST_202, // 13   // "Accepted"
    ST_203, // 14   // "Non-Authoritative Information"
    ST_204, // 15   // "No Content"
    ST_205, // 16   // "Reset Content"
    ST_206, // 17   // "Partial Content"
    ST_300, // 18   // "Multiple Choices"
    ST_301, // 19   // "Moved Permanently"
    ST_302, // 20   // "Found"
    ST_303, // 21   // "See Other"
    ST_304, // 22   // "Not Modified"
    ST_305, // 23   // "Use Proxy"
    ST_306, // 24   // UNUSED
    ST_307, // 25   // "Temporary Redirect"
    ST_400, // 26   // "Bad Request"
    ST_401, // 27   // "Unauthorized"
    ST_402, // 28   // "Payment Required"
    ST_403, // 29   // "Forbidden"
    ST_404, // 30   // "Not Found"
    ST_405, // 31   // "Method Not Allowed"
    ST_406, // 32   // "Not Acceptable"
    ST_407, // 33   // "Proxy Authentication Required"
    ST_408, // 34   // "Request Time-out"
    ST_409, // 35   // "Conflict"
    ST_410, // 36   // "Gone"
    ST_411, // 37   // "Length Required"
    ST_412, // 38   // "Precondition Failed"
    ST_413, // 39   // "Request Entity Too Large"
    ST_414, // 40   // "Request-URI Too Large"
    ST_415, // 41   // "Unsupported Media Type"
    ST_416, // 42   // "Requested Range not satisfiable"
    ST_417, // 43   // "Expectation Failed"
    ST_500, // 44   // "Internal Server Error"
    ST_501, // 45   // "Not Implemented"
    ST_502, // 46   // "Bad Gateway"
    ST_503, // 47   // "Service Unavailable"
    ST_504, // 48   // "Gateway Time-out"
    ST_505, // 49   // "HTTP Version not supported"
    ST_EXT, // 50   // AVAILABLE FOR EXTENSION
    MSG_FAILED, // 51
    MSG_OK, // 52
    CLOSE_SOCKET, // 53
    SOCKET_CLOSED, // 54
    QUIT, // 55
    START, // 56
    MSG_ERROR, // 57
    MSG_WARNING, // 58
    MSG_INFO, // 59
    MSG_UNKNOWN, // 60
    LAST_WMESSAGE, // 61  //  LAST_WMESSAGE
    C_EMPTY, // -1
    C_MSG_FAILED, //  0
    C_MSG_OK, //  1
    C_INIT, //  2
    C_FINISHED, //  3
    C_SEND, //  4
    C_ALLOC, //  5
    C_UI, //  6
    C_APP_CONTACT_DM, //  7
    C_DM_CONTACT_DM, //  8
    C_SHM_MALLOC, //  9
    C_SHM_MALLOC_LIST, // 10
    C_MALLOC_OK, // 11
    C_MALLOC_LIST_OK, // 12
    C_MALLOC_FAILED, // 13
    C_PREPARE_CONTACT, // 14
    C_PREPARE_CONTACT_DM, // 15
    C_PORT, // 16
    C_GET_SHM_KEY, // 17
    C_NEW_OBJECT, // 18
    C_GET_OBJECT, // 19
    C_REGISTER_TYPE, // 20
    C_NEW_SDS, // 21
    C_SEND_ID, // 22
    C_ASK_FOR_OBJECT, // 23
    C_OBJECT_FOUND, // 24
    C_OBJECT_NOT_FOUND, // 25
    C_HAS_OBJECT_CHANGED, // 26
    C_OBJECT_UPDATE, // 27
    C_OBJECT_TRANSFER, // 28
    C_OBJECT_FOLLOWS, // 29
    C_OBJECT_OK, // 30
    C_CLOSE_SOCKET, // 31
    C_DESTROY_OBJECT, // 32
    C_CTRL_DESTROY_OBJECT, // 33
    C_QUIT, // 34
    C_START, // 35
    C_COVISE_ERROR, // 36
    C_INOBJ, // 37
    C_OUTOBJ, // 38
    C_OBJECT_NO_LONGER_USED, // 39
    C_SET_ACCESS, // 40
    C_FINALL, // 41
    C_ADD_OBJECT, // 42
    C_DELETE_OBJECT, // 43
    C_NEW_OBJECT_VERSION, // 44
    C_RENDER, // 45
    C_WAIT_CONTACT, // 46
    C_PARINFO, // 47
    C_MAKE_DATA_CONNECTION, // 48
    C_COMPLETE_DATA_CONNECTION, //49
    C_SHM_FREE, // 50
    C_GET_TRANSFER_PORT, // 51
    C_TRANSFER_PORT, // 52
    C_CONNECT_TRANSFERMANAGER, // 53
    C_STDINOUT_EMPTY, // 54
    C_WARNING, // 55
    C_INFO, // 56
    C_REPLACE_OBJECT, // 57
    C_PLOT, // 58
    C_GET_LIST_OF_INTERFACES, // 59
    C_USR1, // 60
    C_USR2, // 61
    C_USR3, // 62
    C_USR4, // 63
    C_NEW_OBJECT_OK, // 64
    C_NEW_OBJECT_FAILED, // 65
    C_NEW_OBJECT_SHM_MALLOC_LIST, //66
    C_REQ_UI, // 67
    C_NEW_PART_ADDED, // 68
    C_SENDING_NEW_PART, // 69
    C_FINPART, // 70
    C_NEW_PART_AVAILABLE, // 71
    C_OBJECT_ON_HOSTS, // 72
    C_OBJECT_FOLLOWS_CONT, // 73
    C_CRB_EXEC, // 74
    C_COVISE_STOP_PIPELINE, // 75
    C_PREPARE_CONTACT_MODULE, // 76
    C_MODULE_CONTACT_MODULE, // 77
    C_SEND_APPL_PROCID, // 78
    C_INTERFACE_LIST, // 79
    C_MODULE_LIST, // 80
    C_HOSTID, // 81
    C_MODULE_STARTED, // 82
    C_GET_USER, // 83
    C_SOCKET_CLOSED, // 84
    C_NEW_COVISED, // 85
    C_USER_LIST, // 86
    C_STARTUP_INFO, // 87
    C_CO_MODULE, // 88
    C_WRITE_SCRIPT, // 89
    C_CRB, // 90
    C_GENERIC, // 91
    C_RENDER_MODULE, // 92
    C_FEEDBACK, // 93
    C_VRB_CONTACT, // 94
    C_VRB_CONNECT_TO_COVISE, // 95
    C_VRB_CHECK_COVER, // 96
    C_END_IMM_CB, //97
    C_NEW_DESK, //98
    C_VRB_SET_USERINFO, //99
    C_VRB_GET_ID, //100
    C_VRB_SET_GROUP, //101
    C_VRB_QUIT, //102
    LAST_DUMMY_MESSAGE //103
};

// TODO
enum sender_type
{
    UNDEFINED = 0,
    CONTROLLER,
    DATAMANAGER,
    USERINTERFACE,
    RENDERER,
    APPLICATIONMODULE,
    TRANSFERMANAGER,
    SIMPLEPROCESS,
    SIMPLECONTROLLER,
    STDINOUT,
    COVISED
};

#ifdef DEFINE_HTTP_MSGS

char *msg_array[] = {
    "MSG_EMPTY", //  0
    "OPTIONS", //  1      --- first request message
    "GET", //  2
    "HEAD", //  3
    "POST", //  4
    "PUT", //  5
    "DELETE", //  6
    "TRACE", //  7
    "CONNECT", //  8      --- last request message
    "100", //  9   // "Continue"
    "101", // 10   // "Switching Protocols"
    "200", // 11   // "OK"
    "201", // 12   // "Created"
    "202", // 13   // "Accepted"
    "203", // 14   // "Non-Authoritative Information"
    "204", // 15   // "No Content"
    "205", // 16   // "Reset Content"
    "206", // 17   // "Partial Content"
    "300", // 18   // "Multiple Choices"
    "301", // 19   // "Moved Permanently"
    "302", // 20   // "Found"
    "303", // 21   // "See Other"
    "304", // 22   // "Not Modified"
    "305", // 23   // "Use Proxy"
    "306", // 24   // UNUSED
    "307", // 25   // "Temporary Redirect"
    "400", // 26   // "Bad Request"
    "401", // 27   // "Unauthorized"
    "402", // 28   // "Payment Required"
    "403", // 29   // "Forbidden"
    "404", // 30   // "Not Found"
    "405", // 31   // "Method Not Allowed"
    "406", // 32   // "Not Acceptable"
    "407", // 33   // "Proxy Authentication Required"
    "408", // 34   // "Request Time-out"
    "409", // 35   // "Conflict"
    "410", // 36   // "Gone"
    "411", // 37   // "Length Required"
    "412", // 38   // "Precondition Failed"
    "413", // 39   // "Request Entity Too Large"
    "414", // 40   // "Request-URI Too Large"
    "415", // 41   // "Unsupported Media Type"
    "416", // 42   // "Requested Range not satisfiable"
    "417", // 43   // "Expectation Failed"
    "500", // 44   // "Internal Server Error"
    "501", // 45   // "Not Implemented"
    "502", // 46   // "Bad Gateway"
    "503", // 47   // "Service Unavailable"
    "504", // 48   // "Gateway Time-out"
    "505", // 49   // "HTTP Version not supported"
    "EXT", // 50   // AVAILABLE FOR EXTENSION
    "MSG_FAILED", // 51
    "MSG_OK", // 52
    "CLOSE_SOCKET", // 53
    "SOCKET_CLOSED", // 54
    "QUIT", // 55
    "START", // 56
    "MSG_ERROR", // 57
    "MSG_WARNING", // 58
    "MSG_INFO", // 59
    "MSG_UNKNOWN", // 60
    "LAST_WMESSAGE" // 61
};

char *msg_txt[] = {
    "MSG_EMPTY", //  0
    "OPTIONS", //  1      --- 1st request message
    "GET", //  2
    "HEAD", //  3
    "POST", //  4
    "PUT", //  5
    "DELETE", //  6
    "TRACE", //  7
    "CONNECT", //  8      --- last request message
    "Continue", //  9   // 100 -- 1st status msg.
    "Switching Protocols", // 10   // 101
    "OK", // 11   // 200
    "Created", // 12   // 201
    "Accepted", // 13   // 202
    "Non-Authoritative Information", // 14   // 203
    "No Content", // 15   // 204
    "Reset Content", // 16   // 205
    "Partial Content", // 17   // 206
    "Multiple Choices", // 18   // 300
    "Moved Permanently", // 19   // 301
    "Found", // 20   // 302
    "See Other", // 21   // 303
    "Not Modified", // 22   // 304
    "Use Proxy", // 23   // 305
    "UNUSED", // 24   // UNUSED
    "Temporary Redirect", // 25   // 307
    "Bad Request", // 26   // 400
    "Unauthorized", // 27   // 401
    "Payment Required", // 28   // 402
    "Forbidden", // 29   // 403
    "Not Found", // 30   // 404
    "Method Not Allowed", // 31   // 405
    "Not Acceptable", // 32   // 406
    "Proxy Authentication Required", // 33   // 407
    "Request Time-out", // 34   // 408
    "Conflict", // 35   // 409
    "Gone", // 36   // 410
    "Length Required", // 37   // 411
    "Precondition Failed", // 38   // 412
    "Request Entity Too Large", // 39   // 413
    "Request-URI Too Large", // 40   // 414
    "Unsupported Media Type", // 41   // 415
    "Requested Range not satisfiable", // 42   // 416
    "Expectation Failed", // 43   // 417
    "Internal Server Error", // 44   // 500
    "Not Implemented", // 45   // 501
    "Bad Gateway", // 46   // 502
    "Service Unavailable", // 47   // 503
    "Gateway Time-out", // 48   // 504
    "HTTP Version not supported", // 49   // 505
    "EXT", // 50   // AVAILABLE FOR EXTENSION
    "MSG_FAILED", // 51
    "MSG_OK", // 52
    "CLOSE_SOCKET", // 53
    "SOCKET_CLOSED", // 54
    "QUIT", // 55
    "START", // 56
    "MSG_ERROR", // 57
    "MSG_WARNING", // 58
    "MSG_INFO", // 59
    "MSG_UNKNOWN", // 60
    "LAST_WMESSAGE" // 61
};

char *msg_form[] = {
    NULL, //  0
    NULL, //  1      --- 1st request message
    NULL, //  2
    NULL, //  3
    NULL, //  4
    NULL, //  5
    NULL, //  6
    NULL, //  7
    NULL, //  8      --- last request message
    NULL, //  9   // 100 -- 1st status msg.
    NULL, // 10   // 101
    NULL, // 11   // 200
    NULL, // 12   // 201
    NULL, // 13   // 202
    NULL, // 14   // 203
    NULL, // 15   // 204
    NULL, // 16   // 205
    NULL, // 17   // 206
    NULL, // 18   // 300
    NULL, // 19   // 301
    "The actual URL is '%.80s'.\n", // 20   // 302
    NULL, // 21   // 303
    NULL, // 22   // 304
    NULL, // 23   // 305
    NULL, // 24   // UNUSED
    NULL, // 25   // 307
    // 26   // 400
    "Your request has bad syntax or is inherently impossible to satisfy.\n",
    // 27   // 401
    "Authorization required for the URL '%.80s'.\n",
    NULL, // 28   // 402
    // 29   // 403
    "You do not have permission to get URL '%.80s' from this server.\n",
    // 30   // 404
    "The requested URL '%.80s' was not found on this server.\n",
    NULL, // 31   // 405
    NULL, // 32   // 406
    NULL, // 33   // 407
    // 34   // 408
    "No request appeared within a reasonable time period.\n",
    NULL, // 35   // 409
    NULL, // 36   // 410
    NULL, // 37   // 411
    NULL, // 38   // 412
    NULL, // 39   // 413
    NULL, // 40   // 414
    // 41   // 415
    "Serving requests for '%.80s' media type is not implemented by this server.\n",
    NULL, // 42   // 416
    NULL, // 43   // 417
    // 44   // 500
    "There was an unusual problem serving the requested URL '%.80s'.\n",
    // 45   // 501
    "The requested method '%.80s' is not implemented by this server.\n",
    NULL, // 46   // 502
    // 47   // 503
    "The requested URL '%.80s' is temporarily overloaded.  Please try again later.\n",
    NULL, // 48   // 504
    NULL, // 49   // 505
    NULL, // 50   // AVAILABLE FOR EXTENSION
    NULL, // 51
    NULL, // 52
    NULL, // 53
    NULL, // 54
    NULL, // 55
    NULL, // 56
    NULL, // 57
    NULL, // 58
    NULL, // 59
    NULL, // 60
    NULL // 61   // LAST WMESSAGE
};

#else
extern char *msg_array[];
extern char *msg_txt[];
extern char *msg_form[];
#endif

// IDs for the data type encoding (for IPC)

const int NONE = 0;

const int SIZEOF_IEEE_CHAR = 1;
const int SIZEOF_IEEE_SHORT = 2;
const int SIZEOF_IEEE_INT = 4;
const int SIZEOF_IEEE_LONG = 4;
const int SIZEOF_IEEE_FLOAT = 4;
const int SIZEOF_IEEE_DOUBLE = 8;
const int SIZEOF_ALIGNMENT = 8;

class Header;

class Message // class for generic messages
{

public:
    msg_type m_type; // type of the message
    Connection *m_conn; // connection at which message has been rcv

    int m_length; // length of the message body in bytes
    char *m_data; // pointer to the data of the message

    Message()
    {
        m_type = MSG_EMPTY;
        m_conn = NULL;
        m_length = 0;
        m_data = NULL;
    };

    virtual ~Message()
    {
        if (m_data)
            delete[] m_data;
        //if(m_conn) delete m_conn;
        m_type = MSG_EMPTY;
    };

    virtual void print(void)
    {
        cerr << " Generic Message - don't use it !!! ";
    };
};

class HMessage : public Message
{ // class for HTTP messages
public:
    char *m_start; // request-line | status-line
    // headers of the message
    Header *m_headers[MAX_HEADERS + 1];
    Liste<Header> *m_unknownHs; // unknown headers

    char *m_method;
    char *m_URI;
    char *m_version;

    int m_eoh;
    int m_eom;
    int m_index;

    int m_send_content;
    int m_indirect;
    FILE *m_fd;

    // empty initialization:
    void init_headers(void)
    {
        int i;
        for (i = 0; i < MAX_HEADERS + 1; i++)
            m_headers[i] = NULL;
        //m_unknownHs = new Liste<Header>;
    }

    void remove_headers(void)
    {
        int i;
        for (i = 0; i < MAX_HEADERS + 1; i++)
        {
            if (m_headers[i] != NULL)
            {
                delete m_headers[i];
            }
        }
        //delete m_unknownHs;
        //m_unknownHs = NULL;
    }

    HMessage()
    {
        m_start = NULL;
        m_version = NULL;
        m_URI = NULL;

        m_eoh = 0;
        m_eom = 0;
        m_index = 0;

        m_send_content = 1;
        m_indirect = 0;
        m_fd = NULL;

        init_headers();
    };

    HMessage(msg_type type)
    {
        m_type = type;
        m_version = NULL;
        m_URI = NULL;

        m_eoh = 0;
        m_eom = 0;
        m_index = 0;

        m_send_content = 1;
        m_indirect = 0;
        m_fd = NULL;

        init_headers();
        int l = strlen(msg_txt[m_type]) + 16;
        m_start = new char[l];
        sprintf(m_start, "HTTP/1.1 %s %s", msg_array[m_type], msg_txt[m_type]);
    };

    ~HMessage()
    {
        if (m_start)
            delete m_start;
        if (m_version)
            delete m_version;
        if (m_URI)
            delete m_URI;
        remove_headers();
    };

    int process_headers(char *buff);

    void add_start(char *start_line);
    void post_addH(header_type type);
    int add_header(header_type type, Header *hd);
    int add_header(header_type type, char *value);
    int add_header(char *name, char *value);

    void add_body(char *content, int length, char *lang, int type);
    void add_body(char *part);

    void process_start_line(void);
    Header *get_header(header_type type);
    Liste<Header> *get_unknowns(void)
    {
        return m_unknownHs;
    };
    void print(void);
};

class CMessage : public Message
{ // class for Covise messages
public:
    int m_sender; // sender of message (max. 3bytes)
    sender_type m_send_type; // type of sender

    CMessage()
    {
        ;
    };

    void print(void);
};
#endif
