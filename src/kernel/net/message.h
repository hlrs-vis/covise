/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_MESSAGE_H
#define EC_MESSAGE_H

#include <string.h>
#include <stdio.h>
#include <iostream>

#include <util/coExport.h>
#include <util/byteswap.h>

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

namespace covise
{

const int MSG_NOCOPY = 0;
const int MSG_COPY = 1;

class Connection;

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
class NETEXPORT MessageBase
{
public:
	int length; // length of the message in byte
	char* data = nullptr; // pointer to the data of the message
	Connection* conn; // connection at which message has been received (if so)
	MessageBase();
	MessageBase(TokenBuffer*);
	MessageBase(const TokenBuffer&);
	~MessageBase();

	virtual void print() = 0;
	void delete_data()
	{
		if (mustDelete)
		{
			delete[] data;
			data = NULL;
		}
	};
	char* takeData();

protected:
	bool mustDelete;
	
};
class NETEXPORT Message : public MessageBase// class for messages
{
public:
    // message types
    enum Type
    {
        EMPTY = -1,
        HOSTID = 81,
        SOCKET_CLOSED = 84,
        CLOSE_SOCKET = 31,
        STDINOUT_EMPTY = 54,
        UI = 6,
        RENDER = 45,
    };

    enum SenderType
    {
        UNDEFINED = 0,
        STDINOUT = 9
    };

    //    static int new_count;
    //    static int delete_count;
    int sender; // sender of message (max. 3bytes)
    int send_type; // type of sender
    int type; // type of the message

    // empty initialization:
    Message()
        : sender(-1)
        , send_type(Message::UNDEFINED)
        , type(Message::EMPTY)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };
	Message(TokenBuffer* t);
	Message(const TokenBuffer& t);
    Message(Connection *c)
        : sender(-1)
        , send_type(Message::UNDEFINED)
        , type(Message::EMPTY)
    {
		conn = c;
		//printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        print();
    };
    // initialization with data only (for sending):
    Message(int message_type, const std::string &str = std::string())
        : sender(-1)
        , send_type(Message::UNDEFINED)
        , type(message_type)
    {
		mustDelete = true;
		if (!str.empty())
        {
            length = (int)str.length() + 1;
            data = new char[length];
            memcpy(data, str.c_str(), length);
        }
        print();
    };
    Message(int message_type, const char *d, int cp)
        : sender(-1)
        , send_type(Message::UNDEFINED)
        , type(message_type)
    {
        //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
        if (d)
            length = (int)strlen(d) + 1;
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
    Message(int message_type, int l, char *d, int cp = MSG_COPY)
        : sender(-1)
        , send_type(Message::UNDEFINED)
        , type(message_type)
    {
		length = l;
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

    Message &operator=(const Message &); // assignment
    void delete_data()
    {
        delete[] data;
        data = NULL;
    };
    char *extract_data();
    void print() override;

};
}
#endif
