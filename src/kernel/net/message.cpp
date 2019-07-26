/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "message.h"

#include <util/coErr.h>
#include <util/byteswap.h>
#include "tokenbuffer.h"

#define DEFINE_MSG_TYPES
#include "message_types.h"
#include <cassert>
/*
 $Log: covise_msg.C,v $
Revision 1.3  1994/03/23  18:07:03  zrf30125
Modifications for multiple Shared Memory segments have been finished
(not yet for Cray)

Revision 1.2  93/10/08  19:19:15  zrhk0125
some fixed type sizes with sizeof calls replaced

Revision 1.1  93/09/25  20:47:21  zrhk0125
Initial revision
*/

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
 **   Classes      : Message, ShmMessage                                **
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
 **                  15.04.93  Ver 1.1 adopted to shm-malloc handling   **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise {
MessageBase::MessageBase()
	: conn(nullptr)
{

}
MessageBase::MessageBase(TokenBuffer& tb)
{
    data = tb.getData();
	//printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
}
MessageBase::MessageBase(DataHandle& dh)
{
    data =dh;
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
}

MessageBase::~MessageBase()
{
}


Message::Message(TokenBuffer &t)
    :MessageBase(t)
	,type(Message::EMPTY)
{
}

void Message::print()
{
#ifdef DEBUG
    fprintf(stderr, "Message: this=%p, sender=%d, sender_type=%d\n", this, sender, send_type);
    fprintf(stderr, "  type=%s (%d), length=%d, conn=%p\n",
            (type >= 0 && type < sizeof(covise_msg_types_array) / sizeof(covise_msg_types_array[0])) ? covise_msg_types_array[type] : (type == -1 ? "EMPTY" : "(invalid)"),
            type, length, conn);
#endif
}

Message::Message(const Message &src)
{
    //    printf("+ in message no. %d for %x, line %d\n", new_count++, this, __LINE__);
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
    sender = src.sender;
    send_type = src.send_type;
    type = src.type;
    int length = src.data.length();
    char *c = new char[length];
    memcpy(c, src.data.data(), length);
    data = DataHandle(c, length);
    conn = src.conn;
    print();
}
Message::Message(int message_type, const DataHandle &dh)
    :sender(-1)
    ,send_type(Message::UNDEFINED)
    ,type(message_type)
{
    data = dh;
}
Message &Message::operator=(const Message &src)
{
    //    printf("+ in message no. %d for %x, line %d\n", new_count++, this, __LINE__);
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);

    // Check against self-assignment
    if (&src != this)
    {
        // always cope these
        sender = src.sender;
        send_type = src.send_type;
        type = src.type;
        conn = src.conn;

        data = DataHandle(src.data.length());
        memcpy(data.accessData(), src.data.data(), data.length());

    }
    print();
    return *this;
}

void Message::copyAndReuseData(const Message& src)
{
    sender = src.sender;
    send_type = src.send_type;
    type = src.type;
    conn = src.conn;
    data = src.data;
}
//char *Message::extract_data()
//{
//    char *tmpdata = data;
//    data = NULL;
//    return tmpdata;
//}

bool isVrbMessageType(int type)
{
    switch (type)
    {
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    case COVISE_MESSAGE_VRB_SEND_FILE:
    case COVISE_MESSAGE_VRB_CURRENT_FILE:
    case COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE: // Set Registry value
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS:
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
    case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
    case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
    case COVISE_MESSAGE_VRB_CONTACT:
    case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    case COVISE_MESSAGE_RENDER:
    case COVISE_MESSAGE_RENDER_MODULE: // send Message to all others in same group
    case COVISE_MESSAGE_VRB_CHECK_COVER:
    case COVISE_MESSAGE_VRB_GET_ID:
    case COVISE_MESSAGE_VRB_SET_GROUP:
    case COVISE_MESSAGE_VRB_SET_MASTER:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_VRB_FB_RQ:
    case COVISE_MESSAGE_VRB_FB_REMREQ:
        return true;

    default:
        return false;
    }

    return false;
}

} // namespace covise
