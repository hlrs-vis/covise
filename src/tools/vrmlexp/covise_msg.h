/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_MESSAGE_H
#define EC_MESSAGE_H

#include "covise_byteswap.h"

//#include "CharString.h"

#ifdef _WIN64
#define __WORDSIZE 64
#endif

/*
 $Log:  $
 * Revision 1.4  1994/03/23  18:07:03  zrf30125
 * Modifications for multiple Shared Memory segments have been finished
 * (not yet for Cray)
 *
 * Revision 1.3  93/10/11  09:22:19  zrhk0125
 * new types DM_CONTACT_DM and APP_CONTACT_DM included
 *
 * Revision 1.2  93/10/08  19:18:06  zrhk0125
 * data type sizes introduced
* some fixed type sizes with sizeof calls replaced
 *
 * Revision 1.1  93/09/25  20:47:03  zrhk0125
 * Initial revision
 *
 */

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
 **   Classes      : Message, ShmMessage                                **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                        HOSTID                                             **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  15.04.93  Ver 1.0                                  **
 **                  15.04.93  Ver 1.1 new Messages and type added      **
 **                                    sender and send_type added       **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

class DataManagerProcess;
class Connection;
//class  ShmPtr;

// IDs for all messages that go between processes are fixed here

enum covise_msg_type
{
    EMPTY = -1, // -1
    MSG_FAILED, //  0
    MSG_OK, //  1
    INIT, //  2
    FINISHED, //  3
    SEND, //  4
    ALLOC, //  5
    UI, //  6
    APP_CONTACT_DM, //  7
    DM_CONTACT_DM, //  8
    SHM_MALLOC, //  9
    SHM_MALLOC_LIST, // 10
    MALLOC_OK, // 11
    MALLOC_LIST_OK, // 12
    MALLOC_FAILED, // 13
    PREPARE_CONTACT, // 14
    PREPARE_CONTACT_DM, // 15
    PORT, // 16
    GET_SHM_KEY, // 17
    NEW_OBJECT, // 18
    GET_OBJECT, // 19
    REGISTER_TYPE, // 20
    NEW_SDS, // 21
    SEND_ID, // 22
    ASK_FOR_OBJECT, // 23
    OBJECT_FOUND, // 24
    OBJECT_NOT_FOUND, // 25
    HAS_OBJECT_CHANGED, // 26
    OBJECT_UPDATE, // 27
    OBJECT_TRANSFER, // 28
    OBJECT_FOLLOWS, // 29
    OBJECT_OK, // 30
    CLOSE_SOCKET, // 31
    DESTROY_OBJECT, // 32
    CTRL_DESTROY_OBJECT, // 33
    QUIT, // 34
    START, // 35
    COVISE_ERROR, // 36
    INOBJ, // 37
    OUTOBJ, // 38
    OBJECT_NO_LONGER_USED, // 39
    SET_ACCESS, // 40
    FINALL, // 41
    ADD_OBJECT, // 42
    DELETE_OBJECT, // 43
    NEW_OBJECT_VERSION, // 44
    RENDER, // 45
    WAIT_CONTACT, // 46
    PARINFO, // 47
    MAKE_DATA_CONNECTION, // 48
    COMPLETE_DATA_CONNECTION, // 49
    SHM_FREE, // 50
    GET_TRANSFER_PORT, // 51
    TRANSFER_PORT, // 52
    CONNECT_TRANSFERMANAGER, // 53
    STDINOUT_EMPTY, // 54
    WARNING, // 55
    INFO, // 56
    REPLACE_OBJECT, // 57
    PLOT, // 58
    GET_LIST_OF_INTERFACES, // 59
    USR1, // 60
    USR2, // 61
    USR3, // 62
    USR4, // 63
    NEW_OBJECT_OK, // 64
    NEW_OBJECT_FAILED, // 65
    NEW_OBJECT_SHM_MALLOC_LIST, // 66
    REQ_UI, // 67
    NEW_PART_ADDED, // 68
    SENDING_NEW_PART, // 69
    FINPART, // 70
    NEW_PART_AVAILABLE, // 71
    OBJECT_ON_HOSTS, // 72
    OBJECT_FOLLOWS_CONT, // 73
    CRB_EXEC, // 74
    COVISE_STOP_PIPELINE, // 75
    PREPARE_CONTACT_MODULE, // 76
    MODULE_CONTACT_MODULE, // 77
    SEND_APPL_PROCID, // 78
    INTERFACE_LIST, // 79
    MODULE_LIST, // 80
    HOSTID, // 81
    MODULE_STARTED, // 82
    GET_USER, // 83
    SOCKET_CLOSED, // 84
    NEW_COVISED, // 85
    USER_LIST, // 86
    STARTUP_INFO, // 87
    CO_MODULE, // 88
    WRITE_SCRIPT, // 89
    CRB, // 90
    GENERIC, // 91
    RENDER_MODULE, // 92
    FEEDBACK, // 93
    VRB_CONTACT, // 94
    VRB_CONNECT_TO_COVISE, // 95
    VRB_CHECK_COVER, // 96
    END_IMM_CB, // 97
    NEW_DESK, // 98
    VRB_SET_USERINFO, // 99
    VRB_GET_ID, // 100
    VRB_SET_GROUP, // 101
    VRB_QUIT, // 102
    VRB_SET_MASTER, // 103
    VRB_GUI, // 104
    VRB_CLOSE_VRB_CONNECTION, // 105
    VRB_REQUEST_FILE, // 106
    VRB_SEND_FILE, // 107
    VRB_CURRENT_FILE, // 108
    CRB_QUIT, // 109
    REMOVED_HOST, // 110
    START_COVER_SLAVE, // 111
    VRB_REGISTRY_ENTRY_CHANGED, // 112
    VRB_REGISTRY_ENTRY_DELETED, // 113
    VRB_REGISTRY_SUBSCRIBE_CLASS, // 114
    VRB_REGISTRY_SUBSCRIBE_VARIABLE, // 115
    VRB_REGISTRY_CREATE_ENTRY, // 116
    VRB_REGISTRY_SET_VALUE, // 117
    VRB_REGISTRY_DELETE_ENTRY, // 118
    VRB_REGISTRY_UNSUBSCRIBE_CLASS, // 119
    VRB_REGISTRY_UNSUBSCRIBE_VARIABLE, // 120
    SYNCHRONIZED_ACTION, // 121
    ACCESSGRID_DAEMON, // 122
    TABLET_UI, // 123
    QUERY_DATA_PATH, // 124
    SEND_DATA_PATH, // 125
    LAST_DUMMY_MESSAGE
};
#ifdef DEFINE_MSG_TYPES
const char *covise_msg_types_array[] = {
    "FAILED", //  0
    "OK", //  1
    "INIT", //  2
    "FINISHED", //  3
    "SEND", //  4
    "ALLOC", //  5
    "UI", //  6
    "APP_CONTACT_DM", //  7
    "DM_CONTACT_DM", //  8
    "SHM_MALLOC", //  9
    "SHM_MALLOC_LIST", // 10
    "MALLOC_OK", // 11
    "MALLOC_LIST_OK", // 12
    "MALLOC_FAILED", // 13
    "PREPARE_CONTACT", // 14
    "PREPARE_CONTACT_DM", // 15
    "PORT", // 16
    "GET_SHM_KEY", // 17
    "NEW_OBJECT", // 18
    "GET_OBJECT", // 19
    "REGISTER_TYPE", // 20
    "NEW_SDS", // 21
    "SEND_ID", // 22
    "ASK_FOR_OBJECT", // 23
    "OBJECT_FOUND", // 24
    "OBJECT_NOT_FOUND", // 25
    "HAS_OBJECT_CHANGED", // 26
    "OBJECT_UPDATE", // 27
    "OBJECT_TRANSFER", // 28
    "OBJECT_FOLLOWS", // 29
    "OBJECT_OK", // 30
    "CLOSE_SOCKET", // 31
    "DESTROY_OBJECT", // 32
    "CTRL_DESTROY_OBJECT", // 33
    "QUIT", // 34
    "START", // 35
    "COVISE_ERROR", // 36
    "INOBJ", // 37
    "OUTOBJ", // 38
    "OBJECT_NO_LONGER_USED", // 39
    "SET_ACCESS", // 40
    "FINALL", // 41
    "ADD_OBJECT", // 42
    "DELETE_OBJECT", // 43
    "NEW_OBJECT_VERSION", // 44
    "RENDER", // 45
    "WAIT_CONTACT", // 46
    "PARINFO", // 47
    "MAKE_DATA_CONNECTION", // 48
    "COMPLETE_DATA_CONNECTION", // 49
    "SHM_FREE", // 50
    "GET_TRANSFER_PORT", // 51
    "TRANSFER_PORT", // 52
    "CONNECT_TRANSFERMANAGER", // 53
    "STDINOUT_EMPTY", // 54
    "WARNING", // 55
    "INFO", // 56
    "REPLACE_OBJECT", // 57
    "PLOT", // 58
    "GET_LIST_OF_INTERFACES", // 59
    "USR1", // 60
    "USR2", // 61
    "USR3", // 62
    "USR4", // 63
    "NEW_OBJECT_OK", // 64
    "NEW_OBJECT_FAILED", // 65
    "NEW_OBJECT_SHM_MALLOC_LIST", // 66
    "REQ_UI", // 67
    "NEW_PART_ADDED", // 68
    "SENDING_NEW_PART", // 69
    "FINPART", // 70
    "NEW_PART_AVAILABLE", // 71
    "OBJECT_ON_HOSTS", // 72
    "OBJECT_FOLLOWS_CONT", // 73
    "CRB_EXEC", // 74
    "COVISE_STOP_PIPELINE", // 75
    "PREPARE_CONTACT_MODULE", // 76
    "MODULE_CONTACT_MODULE", // 77
    "SEND_APPL_PROCID", // 78
    "INTERFACE_LIST", // 79
    "MODULE_LIST", // 80
    "HOSTID", // 81
    "MODULE_STARTED", // 82
    "GET_USER", // 83
    "SOCKET_CLOSED", // 84
    "NEW_COVISED", // 85
    "USER_LIST", // 86
    "STARTUP_INFO", // 87
    "CO_MODULE", // 88
    "WRITE_SCRIPT", // 89
    "CRB", // 90
    "GENERIC", // 91
    "RENDER_MODULE", // 92
    "FEEDBACK", // 93
    "VRB_CONTACT", // 94
    "VRB_CONNECT_TO_COVISE", // 95
    "VRB_CHECK_COVER", // 96
    "END_IMM_CB", // 97
    "NEW_DESK", // 98
    "VRB_SET_USERINFO", // 99
    "VRB_GET_ID", // 100
    "VRB_SET_GROUP", // 101
    "VRB_QUIT", // 102
    "VRB_SET_MASTER", // 103
    "VRB_GUI", // 104
    "VRB_CLOSE_VRB_CONNECTION", // 105
    "VRB_REQUEST_FILE", // 106
    "VRB_SEND_FILE", // 107
    "VRB_CURRENT_FILE", // 108
    "CRB_QUIT", // 109
    "REMOVED_HOST", // 110
    "START_COVER_SLAVE", // 111
    "VRB_REGISTRY_ENTRY_CHANGED", // 112
    "VRB_REGISTRY_ENTRY_DELETED", // 113
    "VRB_REGISTRY_SUBSCRIBE_CLASS", // 114
    "VRB_REGISTRY_SUBSCRIBE_VARIABLE", // 115
    "VRB_REGISTRY_CREATE_ENTRY", // 116
    "VRB_REGISTRY_SET_VALUE", // 117
    "VRB_REGISTRY_DELETE_ENTRY", // 118
    "VRB_REGISTRY_UNSUBSCRIBE_CLASS", // 119
    "VRB_REGISTRY_UNSUBSCRIBE_VARIABLE", // 120
    "SYNCHRONIZED_ACTION", // 121
    "ACCESSGRID_DAEMON", // 122
    "TABLET_UI", // 123
    "QUERY_DATA_PATH", // 124
    "SEND_DATA_PATH", // 125
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME",
    "GIVE_ME_A_NAME"
};
#else
extern const char *covise_msg_types_array[];
#endif

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

enum access_type
{
    ACC_DENIED = 0x0,
    ACC_NONE = 0x1,
    ACC_READ_ONLY = 0x2,
    ACC_WRITE_ONLY = 0x4,
    ACC_READ_AND_WRITE = 0x8,
    ACC_READ_WRITE_DESTROY = 0x10,
    ACC_REMOTE_DATA_MANAGER = 0x20
};

enum colormap_type
{
    RGBAX,
    VIRVO
};

// IDs for the data type encoding (for IPC)

const int NONE = 0;
//const int CHAR            =  1;
//const int SHORT           =  2;
//const int INT             =  3;
//const int LONG            =  4;
//const int FLOAT           =  5;
//const int DOUBLE          =  6;
//const int CHARSHMPTR      =  7;
//const int SHORTSHMPTR     =  8;
//const int INTSHMPTR       =  9;
//const int LONGSHMPTR      = 10;
//const int FLOATSHMPTR     = 11;
//const int DOUBLESHMPTR    = 12;
const int CHARSHMARRAY = 13;
const int SHORTSHMARRAY = 14;
const int INTSHMARRAY = 15;
#ifdef __GNUC__
#define LONGSHMARRAY 16
#else
const int LONGSHMARRAY = 16;
#endif
const int FLOATSHMARRAY = 17;
const int DOUBLESHMARRAY = 18;
const int EMPTYCHARSHMARRAY = 13 | 0x80;
const int EMPTYSHORTSHMARRAY = 14 | 0x80;
const int EMPTYINTSHMARRAY = 15 | 0x80;
const int EMPTYLONGSHMARRAY = 16 | 0x80;
const int EMPTYFLOATSHMARRAY = 17 | 0x80;
const int EMPTYDOUBLESHMARRAY = 18 | 0x80;
const int SHMPTRARRAY = 19;
const int CHARSHM = 20;
const int SHORTSHM = 21;
const int INTSHM = 22;
#ifdef __GNUC__
#define LONGSHM 23
#else
const int LONGSHM = 23;
#endif
const int FLOATSHM = 24;
const int DOUBLESHM = 25;
const int USERDEFINED = 26;
const int SHMPTR = 27;
const int COVISE_OBJECTID = 28;
const int DISTROBJ = 29;
const int STRINGSHMARRAY = 30;
const int STRING = 31; // CHARPTR == STRING
const int UNKNOWN = 37;
const int COVISE_NULLPTR = 38;
const int COVISE_OPTIONAL = 39;
const int I_SLIDER = 40;
const int F_SLIDER = 41;
const int PER_FACE = 42;
const int PER_VERTEX = 43;
const int OVERALL = 44;
const int FLOAT_SLIDER = 45;
const int FLOAT_VECTOR = 46;
const int COVISE_BOOLEAN = 47;
const int BROWSER = 48;
const int CHOICE = 49;
const int FLOAT_SCALAR = 50;
const int COMMAND = 51;
const int MMPANEL = 52;
const int TEXT = 53;
const int TIMER = 54;
const int PASSWD = 55;
const int CLI = 56;
const int ARRAYSET = 57;
// do not exceed 127 (see EMPTY... =   | 0x80)
const int COLORMAP_MSG = 58;
const int INT_SLIDER = 59;
const int INT_SCALAR = 60;
const int INT_VECTOR = 61;

const int SIZEOF_IEEE_CHAR = 1;
const int SIZEOF_IEEE_SHORT = 2;
const int SIZEOF_IEEE_INT = 4;
const int SIZEOF_IEEE_LONG = 4;
const int SIZEOF_IEEE_FLOAT = 4;
const int SIZEOF_IEEE_DOUBLE = 8;
const int SIZEOF_ALIGNMENT = 8;

const int START_EVEN = 0;
const int START_ODD = 4;

const int MSG_NOCOPY = 0;
const int MSG_COPY = 1;

const int OBJ_OVERWRITE = 0;
const int OBJ_NO_OVERWRITE = 1;

const int SET_CREATE = 0;

typedef long data_type;

#ifdef BYTESWAP

inline void swap_byte(unsigned int &byte) // only if necessary
{
    byteSwap(byte);
}

// only if necessary
inline void swap_bytes(unsigned int *bytes, int no)
{
    byteSwap(bytes, no);
}

inline void swap_short_byte(unsigned short &byte) // only if necessary
{
    byteSwap(byte);
}

// only if necessary
inline void swap_short_bytes(unsigned short *bytes, int no)
{
    byteSwap(bytes, no);
}

#else
inline void swap_byte(unsigned int){};
inline void swap_bytes(unsigned int *, int){};
inline void swap_short_byte(unsigned short){};
inline void swap_short_bytes(unsigned short *, int){};
#endif

class TokenBuffer;

class Message // class for messages
{
public:
    //    static int new_count;
    //    static int delete_count;
    int sender; // sender of message (max. 3bytes)
    sender_type send_type; // type of sender
    covise_msg_type type; // type of the message
    int length; // length of the message in byte
    char *data; // pointer to the data of the message
    Connection *conn; // connection at which message has been received (if so)
    // empty initialization:
    Message()
        : sender(-1)
        , send_type(UNDEFINED)
        , type(EMPTY)
        , length(0)
        , data(0L)
        , conn(0L)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };

    Message(TokenBuffer *);

    Message(TokenBuffer &);

    Message(Connection *c)
        : sender(-1)
        , send_type(UNDEFINED)
        , type(EMPTY)
        , length(0)
        , data(0L)
        , conn(c)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };
    // initialization with data only (for sending):
    //Message(ShmPtr *ptr);                       // initialization with result of shmalloc
    Message(covise_msg_type m, char *d, int cp = MSG_COPY)
        : type(m)
        , conn(0L)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        if (d)
            length = (int)strlen(d) + 1;
        else
            length = 0;
        if (cp == MSG_NOCOPY || d == 0L)
        {
            data = d;
        }
        else
        {
            data = new char[length];
            memcpy(data, d, length);
        }
        print();
    };
    Message(covise_msg_type m, int l, char *d, int cp = MSG_COPY)
        : type(m)
        , length(l)
        , conn(0L)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        if (cp == MSG_NOCOPY || d == 0L)
        {
            data = d;
        }
        else
        {
            data = new char[length];
            memcpy(data, d, length);
        }
        print();
    };
    Message(const Message &); // copy constructor
    ~Message()
    {
        data = 0L;
        // do NOT delete this pointer here - some apps take over the buffer!!
    };
    Message &operator=(const Message &); // assignment
    void delete_data()
    {
        delete[] data;
        data = NULL;
    };
    char *extract_data();
    char *get_part(char *data = 0L); // parse data message
    void print();
};

class TokenBuffer // class for tokens
{
private:
    int buflen; // number of allocated bytes
    int length; // number of used bytes
    char *data; // pointer to the tokens
    char *currdata; // pointer to the tokens

    void incbuf(int size = 100);

public:
    TokenBuffer()
    {
        buflen = length = 0;
        data = currdata = NULL;
    }
    TokenBuffer(int al)
    {
        buflen = al;
        length = 0;
        data = currdata = new char[al];
    }
    ~TokenBuffer()
    {
#ifndef _WIN32
        if (buflen)
            delete[] data;
#endif
    }
    void delete_data();
    TokenBuffer(Message *msg)
    {
        buflen = 0;
        length = msg->length;
        data = currdata = msg->data;
    }
    TokenBuffer(const char *dat, int len)
    {
        buflen = 0;
        length = len;
        data = currdata = (char *)dat;
    }
    const char *getBinary(int n)
    {
        const char *c = currdata;
        currdata += n;
        return c;
    };
    void addBinary(const char *buf, int n)
    {
        if (buflen < length + n + 1)
            incbuf(n + 40);
        memcpy(currdata, buf, n);
        currdata += n;
        length += n;
    }
    const char *allocBinary(int n)
    {
        if (buflen < length + n + 1)
            incbuf(n + 40);
        const char *buf = currdata;
        currdata += n;
        length += n;
        return buf;
    }

    TokenBuffer &operator<<(const uint64_t i);
    TokenBuffer &operator<<(const uint32_t i);
    TokenBuffer &operator<<(const int i)
    {
        return (*this << (uint32_t)i);
    };
    TokenBuffer &operator<<(const char c)
    {
        if (buflen < length + 2)
            incbuf();
        *currdata = c;
        currdata++;
        length++;
        return (*this);
    }
    TokenBuffer &operator<<(const float f);
    TokenBuffer &operator<<(const char *c)
    {
        int l = ((int)strlen(c) + 1);
        if (buflen < length + l + 1)
            incbuf(l * 10);
        strcpy(currdata, c);
        currdata += l;
        length += l;
        return (*this);
    }
    /* TokenBuffer& operator << (CharString s)
      {
         int l=s.length()+1;
         if(buflen<length+l+1)
            incbuf(l*10);strcpy(currdata,(char *) s);
         currdata+=l; length+=l;
         return(*this);
      }*/
    TokenBuffer &operator<<(TokenBuffer *t);
    TokenBuffer &operator<<(TokenBuffer t)
    {
        if (buflen < length + t.get_length() + 1)
            incbuf(t.get_length() * 4);
        memcpy(currdata, t.get_data(), t.get_length());
        currdata += t.get_length();
        length += t.get_length();
        return (*this);
    }
    TokenBuffer &operator>>(uint64_t &i);
    TokenBuffer &operator>>(uint32_t &i);
    TokenBuffer &operator>>(int &i)
    {
        return (*this >> *((uint32_t *)&i));
    }
    TokenBuffer &operator>>(char &c)
    {
        c = *(char *)currdata;
        currdata++;
        return (*this);
    }
    TokenBuffer &operator>>(float &f);
    TokenBuffer &operator>>(char *&c)
    {
        char *end = data + length - 1;
        c = currdata;
        while (*currdata)
        {
            currdata++;
            if (currdata > end)
            {
                cerr << "string not terminated within range" << endl;
                *end = '\0';
                return (*this);
            }
        }
        currdata++;
        return (*this);
    }
    uint32_t get_int_token();
    char get_char_token()
    {
        char ret = *(char *)currdata;
        currdata++;
        return (ret);
    };
    float get_float_token();
    char *get_charp_token()
    {
        char *ret = currdata;
        while (*currdata)
            currdata++;
        currdata++;
        return (ret);
    };
    int get_length()
    {
        return (length);
    };
    const char *get_data()
    {
        return (data);
    };
    void reset()
    {
        currdata = data;
        length = 0;
        if (data)
            data[0] = 0;
    };
};

// #include "covise_shmalloc.h"

class ShmMessage : public Message // message especially for memory allocation
{
public: // at the datamanager
    // constructor with encoding into data field:
    ShmMessage(data_type d, long count)
        : Message()
    {
        conn = 0L;
        type = SHM_MALLOC;
        length = sizeof(data_type) + sizeof(long);
        data = new char[length];
        *(data_type *)data = d;
        *(long *)(&data[sizeof(data_type)]) = count;
    };
    ShmMessage(data_type *d, long *count, int no);
    ShmMessage(char *n, int t, data_type *d, long *count, int no);
    ShmMessage()
        : Message(){};
    //    ~ShmMessage() {
    //	delete [] data;
    //	data = 0L;
    //    };  // destructor
    int process_new_object_list(DataManagerProcess *dmgr);
    int process_list(DataManagerProcess *dmgr);
    data_type get_data_type() // data type of msg
    {
        return *(data_type *)data;
    };
    long get_count() // length of msg
    {
        return *(long *)(data + sizeof(data_type));
    };
    int get_seq_no()
    {
        if (type == MALLOC_OK)
            return *(int *)(data);
        else
            return -1;
    };
    int get_offset()
    {
        if (type == MALLOC_OK)
            return *(int *)(data + sizeof(data_type));
        else
            return -1;
    };
};

class Param
{
    friend class CtlMessage;
    char *name;
    int type;
    int no;

public:
    Param(const char *na, int t, int n)
    {
        name = new char[strlen(na) + 1];
        strcpy(name, na);
        type = t;
        no = n;
    }
    char *get_name() const
    {
        return name;
    }
    int no_of_items()
    {
        return no;
    }
    int get_type()
    {
        return type;
    }
    virtual ~Param()
    {
        delete[] name;
    }
};

class ParamFloatScalar : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamFloatScalar(const char *na, char *l)
        : Param(na, FLOAT_SCALAR, 1)
    {
        list = new char[strlen(l) + 1];
        strcpy(list, l);
    }
    ParamFloatScalar(const char *na, float val)
        : Param(na, FLOAT_SCALAR, 1)
    {
        char *buf = new char[255];
        sprintf(buf, "%f", val);
        list = new char[strlen(buf) + 1];
        strcpy(list, buf);
        delete[] buf;
    }
    ~ParamFloatScalar()
    {
        delete[] list;
    }
};

class ParamIntScalar : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamIntScalar(const char *na, char *l)
        : Param(na, INT_SCALAR, 1)
    {
        list = new char[strlen(l) + 1];
        strcpy(list, l);
    }
    ParamIntScalar(const char *na, long val)
        : Param(na, INT_SCALAR, 1)
    {
        char *buf = new char[255];
        sprintf(buf, "%ld", val);
        list = new char[strlen(buf) + 1];
        strcpy(list, buf);
        delete[] buf;
    }
    ~ParamIntScalar()
    {
        delete[] list;
    }
};

class ParamChoice : public Param
{
    friend class CtlMessage;
    char **list;
    int sel;

public:
    ParamChoice(const char *na, int num, int s, char **l)
        : Param(na, CHOICE, num)
    {
        sel = s;
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamChoice(const char *na, int num, char **l, int s)
        : Param(na, CHOICE, num)
    {
        sel = s;
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ~ParamChoice()
    {
        for (int j = 0; j < no_of_items(); j++)
        {
            delete[] list[j];
        }
        delete[] list;
    }
};

class ParamFloatVector : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamFloatVector(const char *na, int num, char **l)
        : Param(na, FLOAT_VECTOR, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamFloatVector(const char *na, int num, float *l)
        : Param(na, FLOAT_VECTOR, num)
    {
        char *buf = new char[255];
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            sprintf(buf, "%f", *(l + j));
            list[j] = new char[strlen(buf) + 1];
            strcpy(list[j], buf);
        }
        delete[] buf;
    }
    ~ParamFloatVector()
    {
        for (int j = 0; j < no_of_items(); j++)
        {
            delete[] list[j];
        }
        delete[] list;
    }
};

class ParamIntVector : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamIntVector(const char *na, int num, char **l)
        : Param(na, INT_VECTOR, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamIntVector(const char *na, int num, long *l)
        : Param(na, INT_VECTOR, num)
    {
        char *buf = new char[255];
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            sprintf(buf, "%ld", *(l + j));
            list[j] = new char[strlen(buf) + 1];
            strcpy(list[j], buf);
        }
        delete[] buf;
    }
    ~ParamIntVector()
    {
        for (int j = 0; j < no_of_items(); j++)
        {
            delete[] list[j];
        }
        delete[] list;
    }
};

class ParamBrowser : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamBrowser(const char *na, int num, char **l)
        : Param(na, BROWSER, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamBrowser(const char *na, char *file, char *wildcard)
        : Param(na, BROWSER, 2)
    {
        list = new char *[2];
        list[0] = new char[strlen(file) + 1];
        strcpy(list[0], file);
        list[1] = new char[strlen(wildcard) + 1];
        strcpy(list[1], wildcard);
    }
    ~ParamBrowser()
    {
        for (int j = 0; j < no_of_items(); j++)
        {
            delete[] list[j];
        }
        delete[] list;
    }
};

class ParamString : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamString(const char *na, char *l)
        : Param(na, STRING, 1)
    {
        list = new char[strlen(l) + 1];
        strcpy(list, l);
    }
    char *get_length()
    {
        return (char *)strlen(list);
    };
    ~ParamString()
    {
        delete[] list;
    }
};

class ParamText : public Param
{
    friend class CtlMessage;
    char **list;
    int line_num;
    int length;

public:
    ParamText(const char *na, char **l, int lineno, int len)
        : Param(na, TEXT, lineno)
    {
        for (int j = 0; j < lineno; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
        line_num = lineno;
        length = len;
    }
    int get_line_number()
    {
        return line_num;
    };
    int get_length()
    {
        return length;
    };
    ~ParamText()
    {
        for (int j = 0; j < no_of_items(); j++)
        {
            delete[] list[j];
        }
        delete[] list;
    }
};

class ParamBoolean : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamBoolean(const char *na, char *l)
        : Param(na, COVISE_BOOLEAN, 1)
    {
        list = new char[strlen(l) + 1];
        strcpy(list, l);
    }
    ParamBoolean(const char *na, int val)
        : Param(na, COVISE_BOOLEAN, 1)
    {
        if (val == 0)
        {
            list = new char[strlen("FALSE") + 1];
            strcpy(list, "FALSE");
        }
        else
        {
            list = new char[strlen("TRUE") + 1];
            strcpy(list, "TRUE");
        }
    }
    ~ParamBoolean()
    {
        delete[] list;
    }
};

class ParamFloatSlider : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamFloatSlider(const char *na, int num, char **l)
        : Param(na, FLOAT_SLIDER, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamFloatSlider(const char *na, float min, float max, float val)
        : Param(na, FLOAT_SLIDER, 3)
    {
        char *buf = new char[255];
        list = new char *[3];
        sprintf(buf, "%f", min);
        list[0] = new char[strlen(buf) + 1];
        strcpy(list[0], buf);
        sprintf(buf, "%f", max);
        list[1] = new char[strlen(buf) + 1];
        strcpy(list[1], buf);
        sprintf(buf, "%f", val);
        list[2] = new char[strlen(buf) + 1];
        strcpy(list[2], buf);
        delete[] buf;
    }
    ~ParamFloatSlider()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

class ParamIntSlider : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamIntSlider(const char *na, int num, char **l)
        : Param(na, INT_SLIDER, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamIntSlider(const char *na, long min, long max, long val)
        : Param(na, INT_SLIDER, 3)
    {
        char *buf = new char[255];
        list = new char *[3];
        sprintf(buf, "%ld", min);
        list[0] = new char[strlen(buf) + 1];
        strcpy(list[0], buf);
        sprintf(buf, "%ld", max);
        list[1] = new char[strlen(buf) + 1];
        strcpy(list[1], buf);
        sprintf(buf, "%ld", val);
        list[2] = new char[strlen(buf) + 1];
        strcpy(list[2], buf);
        delete[] buf;
    }
    ~ParamIntSlider()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

class ParamMMPanel : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamMMPanel(const char *na, int num, char **l)
        : Param(na, MMPANEL, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamMMPanel(const char *na, char *text1, char *text2,
                 long min, long max, long val, char *b1, char *b2,
                 char *b3, char *b4, char *b5, char *b6, char *b7)
        : Param(na, MMPANEL, 12)
    {
        char *buf = new char[255];
        list = new char *[12];
        list[0] = new char[strlen(text1) + 1];
        strcpy(list[0], text1);
        list[1] = new char[strlen(text2) + 1];
        strcpy(list[1], text1);
        sprintf(buf, "%ld", min);
        list[2] = new char[strlen(buf) + 1];
        strcpy(list[2], buf);
        sprintf(buf, "%ld", max);
        list[3] = new char[strlen(buf) + 1];
        strcpy(list[3], buf);
        sprintf(buf, "%ld", val);
        list[4] = new char[strlen(buf) + 1];
        strcpy(list[4], buf);
        list[5] = new char[strlen(b1) + 1];
        strcpy(list[5], buf);
        list[6] = new char[strlen(b2) + 1];
        strcpy(list[6], buf);
        list[7] = new char[strlen(b3) + 1];
        strcpy(list[7], buf);
        list[8] = new char[strlen(b4) + 1];
        strcpy(list[8], buf);
        list[9] = new char[strlen(b5) + 1];
        strcpy(list[9], buf);
        list[10] = new char[strlen(b6) + 1];
        strcpy(list[10], buf);
        list[11] = new char[strlen(b7) + 1];
        strcpy(list[11], buf);

        delete[] buf;
    }
    ~ParamMMPanel()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete list[j];
        delete[] list;
    }
};

class ParamTimer : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamTimer(const char *na, int num, char **l)
        : Param(na, TIMER, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamTimer(const char *na, long start, long delta, long state)
        : Param(na, TIMER, 3)
    {
        char *buf = new char[255];
        list = new char *[3];
        sprintf(buf, "%ld", start);
        list[0] = new char[strlen(buf) + 1];
        strcpy(list[0], buf);
        sprintf(buf, "%ld", delta);
        list[1] = new char[strlen(buf) + 1];
        strcpy(list[1], buf);
        sprintf(buf, "%ld", state);
        list[2] = new char[strlen(buf) + 1];
        strcpy(list[2], buf);
        delete[] buf;
    }
    ~ParamTimer()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

class ParamPasswd : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamPasswd(const char *na, int num, char **l)
        : Param(na, PASSWD, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }
    ParamPasswd(const char *na, char *host, char *user, char *passwd)
        : Param(na, PASSWD, 3)
    {
        list = new char *[3];
        list[0] = new char[strlen(host) + 1];
        strcpy(list[0], host);
        list[1] = new char[strlen(user) + 1];
        strcpy(list[1], user);
        list[2] = new char[strlen(passwd) + 1];
        strcpy(list[2], passwd);
    }
    ~ParamPasswd()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

class ParamCli : public Param
{
    friend class CtlMessage;
    char *list;

public:
    ParamCli(const char *na, char *l)
        : Param(na, CLI, 1)
    {
        list = new char[strlen(l) + 1];
        strcpy(list, l);
    }
    char *get_length()
    {
        return (char *)strlen(list);
    };

    ~ParamCli()
    {
        delete[] list;
    }
};

class ParamArrayset : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamArrayset(const char *na, int num, char **l)
        : Param(na, ARRAYSET, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }

    ~ParamArrayset()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

class ParamColormap : public Param
{
    friend class CtlMessage;
    char **list;

public:
    ParamColormap(const char *na, int num, char **l)
        : Param(na, COLORMAP_MSG, num)
    {
        list = new char *[num];
        for (int j = 0; j < num; j++)
        {
            list[j] = new char[strlen(l[j]) + 1];
            strcpy(list[j], l[j]);
        }
    }

    ~ParamColormap()
    {
        for (int j = 0; j < no_of_items(); j++)
            delete[] list[j];
        delete[] list;
    }
};

// holds everything necessary to work
// easily with messages from the controller
// to an application module
class CtlMessage : public Message
{
private:
    char *m_name;
    char *h_name;
    char *inst_no;
    int no_of_objects;
    int no_of_save_objects;
    int no_of_release_objects;
    int no_of_params_in;
    int no_of_params_out;
    int MAX_OUT_PAR;
    char **port_names;
    char **object_names;
    char **object_types;
    int *port_connected;
    char **save_names;
    char **release_names;
    Param **params_in;
    Param **params_out;
    void init_list();

public:
    CtlMessage(Message *m)
        : Message()
    {
        sender = m->sender;
        send_type = m->send_type;
        type = m->type;
        length = m->length;
        data = new char[strlen(m->data) + 1];
        strcpy(data, m->data);
        conn = m->conn; // never use this

        m_name = h_name = 0L;
        inst_no = 0L;
        object_types = port_names = object_names = save_names = release_names = 0L;
        no_of_save_objects = no_of_release_objects = 0;
        no_of_params_in = no_of_params_out = 0;
        MAX_OUT_PAR = 100;
        params_out = new Param *[MAX_OUT_PAR];
        init_list();
    }
    ~CtlMessage();

    void get_header(char **m, char **h, char **inst)
    {
        *m = m_name;
        *h = h_name, *inst = inst_no;
    };

    int get_scalar_param(const char *, long *);
    int get_scalar_param(const char *, float *);
    int get_vector_param(const char *, int, long *);
    int get_vector_param(const char *, int, float *);
    int get_string_param(const char *, char **);
    int get_text_param(const char *, char ***, int *line_num);
    int get_boolean_param(const char *, int *);
    int get_slider_param(const char *, long *min, long *max, long *value);
    int get_slider_param(const char *, float *min, float *max, float *value);
    int get_choice_param(const char *, int *);
    int get_choice_param(const char *, char **);
    int get_browser_param(const char *, char **);
    int get_mmpanel_param(const char *name, char **text1, char **text2,
                          long *min, long *max, long *val,
                          int *b1, int *b2, int *b3, int *b4, int *b5, int *b6,
                          int *b7);
    int get_timer_param(const char *, long *start, long *delta, long *state);
    int get_passwd_param(const char *, char **host, char **user, char **passwd);
    int get_cli_param(const char *, char **command);
    int get_arrayset_param(const char *, char **buf);
    int get_colormap_param(const char *, float *min, float *max, int *len, colormap_type *type);

    int set_scalar_param(const char *, long);
    int set_scalar_param(const char *, float);
    int set_vector_param(const char *, int num, long *);
    int set_vector_param(const char *, int num, float *);
    int set_string_param(const char *, char *);
    int set_text_param(const char *, char *, int);
    int set_boolean_param(const char *, int);
    int set_slider_param(const char *, long min, long max, long value);
    int set_slider_param(const char *, float min, float max, float value);
    int set_choice_param(const char *, int, char **, int);
    int set_browser_param(const char *, char *, char *);
    int set_mmpanel_param(const char *name, char *text1, char *text2, long min, long max, long val,
                          int b1, int b2, int b3, int b4, int b5, int b6, int b7);
    int set_timer_param(const char *, long start, long delta, long state);
    int set_passwd_param(const char *, char *host, char *user, char *passwd);
    int set_cli_param(const char *, char *result);

    char *get_object_name(const char *name);
    char *get_object_type(const char *name);

    int set_save_object(const char *name);
    int set_release_object(const char *name);

    int create_finpart_message();
    int create_finall_message();

    int is_port_connected(const char *name);
};

class RenderMessage : public Message // holds everything necessary to work
{
    // easily with messages from the controller
    // to the renderer
    char *m_name;
    char *h_name;
    char *inst_no;
    void init_list();

public:
    int no_of_objects;
    char **object_names;
    RenderMessage(Message *m)
        : Message()
    {
        sender = m->sender;
        send_type = m->send_type;
        type = m->type;
        length = m->length;
        data = m->data;
        m->data = 0L;
        conn = m->conn;
        delete m;
        m_name = h_name = 0L;
        inst_no = 0L;
        no_of_objects = 0;
        object_names = 0L;
        init_list();
    }
    ~RenderMessage();
};
#endif
