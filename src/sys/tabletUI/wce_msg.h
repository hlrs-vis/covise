/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WCE_MESSAGE_H
#define WCE_MESSAGE_H

#include <string.h>
#include <stdio.h>
#include <iostream>

#include <util/byteswap.h>
#include <util/coTypes.h>

namespace covise
{

class DataManagerProcess;
class Connection;
class coShmPtr;

// IDs for all messages that go between processes are fixed here

enum covise_msg_type
{
    COVISE_MESSAGE_EMPTY = -1, // -1
    COVISE_MESSAGE_MSG_FAILED, //  0
    COVISE_MESSAGE_MSG_OK, //  1
    COVISE_MESSAGE_INIT, //  2
    COVISE_MESSAGE_FINISHED, //  3
    COVISE_MESSAGE_SEND, //  4
    COVISE_MESSAGE_ALLOC, //  5
    COVISE_MESSAGE_UI, //  6
    COVISE_MESSAGE_APP_CONTACT_DM, //  7
    COVISE_MESSAGE_DM_CONTACT_DM, //  8
    COVISE_MESSAGE_SHM_MALLOC, //  9
    COVISE_MESSAGE_SHM_MALLOC_LIST, // 10
    COVISE_MESSAGE_MALLOC_OK, // 11
    COVISE_MESSAGE_MALLOC_LIST_OK, // 12
    COVISE_MESSAGE_MALLOC_FAILED, // 13
    COVISE_MESSAGE_PREPARE_CONTACT, // 14
    COVISE_MESSAGE_PREPARE_CONTACT_DM, // 15
    COVISE_MESSAGE_PORT, // 16
    COVISE_MESSAGE_GET_SHM_KEY, // 17
    COVISE_MESSAGE_NEW_OBJECT, // 18
    COVISE_MESSAGE_GET_OBJECT, // 19
    COVISE_MESSAGE_REGISTER_TYPE, // 20
    COVISE_MESSAGE_NEW_SDS, // 21
    COVISE_MESSAGE_SEND_ID, // 22
    COVISE_MESSAGE_ASK_FOR_OBJECT, // 23
    COVISE_MESSAGE_OBJECT_FOUND, // 24
    COVISE_MESSAGE_OBJECT_NOT_FOUND, // 25
    COVISE_MESSAGE_HAS_OBJECT_CHANGED, // 26
    COVISE_MESSAGE_OBJECT_UPDATE, // 27
    COVISE_MESSAGE_OBJECT_TRANSFER, // 28
    COVISE_MESSAGE_OBJECT_FOLLOWS, // 29
    COVISE_MESSAGE_OBJECT_OK, // 30
    COVISE_MESSAGE_CLOSE_SOCKET, // 31
    COVISE_MESSAGE_DESTROY_OBJECT, // 32
    COVISE_MESSAGE_CTRL_DESTROY_OBJECT, // 33
    COVISE_MESSAGE_QUIT, // 34
    COVISE_MESSAGE_START, // 35
    COVISE_MESSAGE_COVISE_ERROR, // 36
    COVISE_MESSAGE_INOBJ, // 37
    COVISE_MESSAGE_OUTOBJ, // 38
    COVISE_MESSAGE_OBJECT_NO_LONGER_USED, // 39
    COVISE_MESSAGE_SET_ACCESS, // 40
    COVISE_MESSAGE_FINALL, // 41
    COVISE_MESSAGE_ADD_OBJECT, // 42
    COVISE_MESSAGE_DELETE_OBJECT, // 43
    COVISE_MESSAGE_NEW_OBJECT_VERSION, // 44
    COVISE_MESSAGE_RENDER, // 45
    COVISE_MESSAGE_WAIT_CONTACT, // 46
    COVISE_MESSAGE_PARINFO, // 47
    COVISE_MESSAGE_MAKE_DATA_CONNECTION, // 48
    COVISE_MESSAGE_COMPLETE_DATA_CONNECTION, // 49
    COVISE_MESSAGE_SHM_FREE, // 50
    COVISE_MESSAGE_GET_TRANSFER_PORT, // 51
    COVISE_MESSAGE_TRANSFER_PORT, // 52
    COVISE_MESSAGE_CONNECT_TRANSFERMANAGER, // 53
    COVISE_MESSAGE_STDINOUT_EMPTY, // 54
    COVISE_MESSAGE_WARNING, // 55
    COVISE_MESSAGE_INFO, // 56
    COVISE_MESSAGE_REPLACE_OBJECT, // 57
    COVISE_MESSAGE_PLOT, // 58
    COVISE_MESSAGE_GET_LIST_OF_INTERFACES, // 59
    COVISE_MESSAGE_USR1, // 60
    COVISE_MESSAGE_USR2, // 61
    COVISE_MESSAGE_USR3, // 62
    COVISE_MESSAGE_USR4, // 63
    COVISE_MESSAGE_NEW_OBJECT_OK, // 64
    COVISE_MESSAGE_NEW_OBJECT_FAILED, // 65
    COVISE_MESSAGE_NEW_OBJECT_SHM_MALLOC_LIST, // 66
    COVISE_MESSAGE_REQ_UI, // 67
    COVISE_MESSAGE_NEW_PART_ADDED, // 68
    COVISE_MESSAGE_SENDING_NEW_PART, // 69
    COVISE_MESSAGE_FINPART, // 70
    COVISE_MESSAGE_NEW_PART_AVAILABLE, // 71
    COVISE_MESSAGE_OBJECT_ON_HOSTS, // 72
    COVISE_MESSAGE_OBJECT_FOLLOWS_CONT, // 73
    COVISE_MESSAGE_CRB_EXEC, // 74
    COVISE_MESSAGE_COVISE_STOP_PIPELINE, // 75
    COVISE_MESSAGE_PREPARE_CONTACT_MODULE, // 76
    COVISE_MESSAGE_MODULE_CONTACT_MODULE, // 77
    COVISE_MESSAGE_SEND_APPL_PROCID, // 78
    COVISE_MESSAGE_INTERFACE_LIST, // 79
    COVISE_MESSAGE_MODULE_LIST, // 80
    COVISE_MESSAGE_HOSTID, // 81
    COVISE_MESSAGE_MODULE_STARTED, // 82
    COVISE_MESSAGE_GET_USER, // 83
    COVISE_MESSAGE_SOCKET_CLOSED, // 84
    COVISE_MESSAGE_NEW_COVISED, // 85
    COVISE_MESSAGE_USER_LIST, // 86
    COVISE_MESSAGE_STARTUP_INFO, // 87
    COVISE_MESSAGE_CO_MODULE, // 88
    COVISE_MESSAGE_WRITE_SCRIPT, // 89
    COVISE_MESSAGE_CRB, // 90
    COVISE_MESSAGE_GENERIC, // 91
    COVISE_MESSAGE_RENDER_MODULE, // 92
    COVISE_MESSAGE_FEEDBACK, // 93
    COVISE_MESSAGE_VRB_CONTACT, // 94
    COVISE_MESSAGE_VRB_CONNECT_TO_COVISE, // 95
    COVISE_MESSAGE_VRB_CHECK_COVER, // 96
    COVISE_MESSAGE_END_IMM_CB, // 97
    COVISE_MESSAGE_NEW_DESK, // 98
    COVISE_MESSAGE_VRB_SET_USERINFO, // 99
    COVISE_MESSAGE_VRB_GET_ID, // 100
    COVISE_MESSAGE_VRB_SET_GROUP, // 101
    COVISE_MESSAGE_VRB_QUIT, // 102
    COVISE_MESSAGE_VRB_SET_MASTER, // 103
    COVISE_MESSAGE_VRB_GUI, // 104
    COVISE_MESSAGE_VRB_CLOSE_VRB_CONNECTION, // 105
    COVISE_MESSAGE_VRB_REQUEST_FILE, // 106
    COVISE_MESSAGE_VRB_SEND_FILE, // 107
    COVISE_MESSAGE_VRB_CURRENT_FILE, // 108
    COVISE_MESSAGE_CRB_QUIT, // 109
    COVISE_MESSAGE_REMOVED_HOST, // 110
    COVISE_MESSAGE_START_COVER_SLAVE, // 111
    COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED, // 112
    COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED, // 113
    COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS, // 114
    COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE, // 115
    COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY, // 116
    COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE, // 117
    COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY, // 118
    COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS, // 119
    COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE, // 120
    COVISE_MESSAGE_SYNCHRONIZED_ACTION, // 121
    COVISE_MESSAGE_ACCESSGRID_DAEMON, // 122
    COVISE_MESSAGE_TABLET_UI, // 123
    COVISE_MESSAGE_QUERY_DATA_PATH, // 124
    COVISE_MESSAGE_SEND_DATA_PATH, // 125
    COVISE_MESSAGE_VRB_FB_RQ, // 126
    COVISE_MESSAGE_VRB_FB_SET, // 127
    COVISE_MESSAGE_VRB_FB_REMREQ, // 128
    COVISE_MESSAGE_UPDATE_LOADED_MAPNAME, // 129
    COVISE_MESSAGE_LAST_DUMMY_MESSAGE // 130
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
const int COLOR_MSG = 62;
const int COLORMAPCHOICE_MSG = 63;
const int MATERIAL_MSG = 64;

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
        , type(COVISE_MESSAGE_EMPTY)
        , length(0)
        , data(NULL)
        , conn(NULL)
        , mustDelete(false)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };

    Message(TokenBuffer *);

    Message(TokenBuffer &);

    Message(Connection *c)
        : sender(-1)
        , send_type(UNDEFINED)
        , type(COVISE_MESSAGE_EMPTY)
        , length(0)
        , data(NULL)
        , conn(c)
        , mustDelete(false)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };
    // initialization with data only (for sending):
    Message(coShmPtr *ptr); // initialization with result of shmalloc
    Message(covise_msg_type m, std::string str)
        : sender(-1)
        , send_type(UNDEFINED)
        , type(m)
        , length(0)
        , data(NULL)
        , conn(NULL)
        , mustDelete(true)
    {
        if (!str.empty())
        {
            length = str.length() + 1;
            data = new char[length];
            memcpy(data, str.c_str(), length);
        }
        print();
    };
    Message(covise_msg_type m, const char *d, int cp = MSG_COPY)
        : sender(-1)
        , send_type(UNDEFINED)
        , type(m)
        , length(0)
        , data(NULL)
        , conn(NULL)
        , mustDelete(false)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        if (d)
            length = strlen(d) + 1;
        else
            length = 0;
        if (cp == MSG_NOCOPY || d == NULL)
        {
            data = (char *)d;
        }
        else
        {
            data = new char[length];
            memcpy(data, d, length);
            mustDelete = true;
        }
        print();
    };
    Message(covise_msg_type m, int l, char *d, int cp = MSG_COPY)
        : sender(-1)
        , send_type(UNDEFINED)
        , type(m)
        , length(l)
        , data(NULL)
        , conn(NULL)
        , mustDelete(false)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        if (cp == MSG_NOCOPY || d == NULL)
        {
            data = d;
        }
        else
        {
            data = new char[length];
            memcpy(data, d, length);
            mustDelete = true;
        }
        print();
    };
    Message(const Message &); // copy constructor
    ~Message()
    {
        if (mustDelete)
            delete[] data;

        data = NULL;
        // do NOT delete this pointer here - some apps take over the buffer!!
    };
    Message &operator=(const Message &); // assignment
    void delete_data()
    {
        delete[] data;
        data = NULL;
    };
    char *extract_data();
    char *get_part(char *data = NULL); // parse data message
    void print();

private:
    bool mustDelete;
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
#ifndef WIN32 // it does not work on win32 as size_t == int
//TokenBuffer& operator << (const size_t s){return (*this<<(uint64_t)s);}
#endif
    TokenBuffer &operator<<(const uint32_t i);
    TokenBuffer &operator<<(const int i)
    {
        return (*this << (uint32_t)i);
    }
    TokenBuffer &operator<<(const std::string &s);
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
    TokenBuffer &operator<<(const double f);
    TokenBuffer &operator<<(const char *c)
    {
        int l = (strlen(c) + 1);
        if (buflen < length + l + 1)
            incbuf(l * 10);
        strcpy(currdata, c);
        currdata += l;
        length += l;
        return (*this);
    }
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
#ifndef WIN32 // it does not work on win32 as size_t == int
//TokenBuffer& operator >> (size_t &s){uint64_t i; *this>>i; s=i; return *this; }
#endif
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
    TokenBuffer &operator>>(double &f);
    TokenBuffer &operator>>(std::string &s);
    TokenBuffer &operator>>(char *&c)
    {
        char *end = data + length - 1;
        c = currdata;
        while (*currdata)
        {
            currdata++;
            if (currdata > end)
            {
                std::cerr << "string not terminated within range" << std::endl;
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
}

#endif
