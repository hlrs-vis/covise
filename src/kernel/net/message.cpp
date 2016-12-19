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

using namespace covise;

Message::Message(TokenBuffer *t)
    : type(Message::EMPTY)
    , conn(NULL)
    , mustDelete(false)
{
    length = t->get_length();
    data = (char *)t->get_data();
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
    print();
}

Message::Message(TokenBuffer &t)
    : type(Message::EMPTY)
    , conn(NULL)
    , mustDelete(false)
{
    length = t.get_length();
    data = (char *)t.get_data();
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
    print();
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
    length = src.length;
    data = new char[length];
    memcpy(data, src.data, length);
    conn = src.conn;
    print();
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
        length = src.length;
        conn = src.conn;

        // copy data (if existent)
        delete[] data;
        data = new char[length];
        if (length && src.data)
            memcpy(data, src.data, length);
    }
    print();
    return *this;
}

char *Message::extract_data()
{
    char *tmpdata = data;
    data = NULL;
    return tmpdata;
}
