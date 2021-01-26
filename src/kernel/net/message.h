/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_MESSAGE_H
#define EC_MESSAGE_H

#include "dataHandle.h"

#include <util/coExport.h>
#include <util/byteswap.h>

#include <string>

#ifdef _WIN64
#define __WORDSIZE 64
#endif

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
    DataHandle data;
    Connection* conn; // connection at which message has been received (if so)
	MessageBase();
	MessageBase(TokenBuffer& tb);
    MessageBase(DataHandle& dh);
	virtual ~MessageBase() = default;
	virtual void print() = 0;
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

    int sender = -1; // sender of message (max. 3bytes)
    int send_type = UNDEFINED; // type of sender
    int type = EMPTY; // type of the message

    // empty initialization:
    Message();
	explicit Message(TokenBuffer& t);
    explicit Message(Connection *c);

    // initialization with data only (for sending):
    explicit Message(int message_type, const std::string &str = std::string());

    Message(int message_type, const DataHandle& dh);

    Message(const Message &); // copy constructor
    //copies data
    Message &operator=(const Message &src); // assignment
    //does not copy data
    void copyAndReuseData(const Message& src);
    //char *extract_data();
    void print() override;

};
}
#endif
