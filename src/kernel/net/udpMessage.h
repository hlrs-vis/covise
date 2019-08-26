/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef UDP_MESSAGE_H
#define UDP_MESSAGE_H

#include <string.h>
#include <stdio.h>
#include <iostream>

#include <util/coExport.h>
#include <util/byteswap.h>
#include "message.h"
#include "udp_message_types.h"
//
//#ifdef _WIN64
//#define __WORDSIZE 64
//#endif

//protocol for udp messages between vrb and clients (OpenCOVER or external renderer)
//basically this is a covise message with smaler header
//
namespace covise
{
	class TokenBuffer;
}
namespace vrb
{

const int MSG_NOCOPY = 0;
const int MSG_COPY = 1;



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



class NETEXPORT UdpMessage : public covise::MessageBase// class for messages
{
public:
	udp_msg_type type; //type of message
    mutable int sender = -1; // sender of message, < 0: invalid, 0 = server, > 0 clients 
	char* m_ip = new char[16];

    UdpMessage()
        : sender(-1)
		, type(udp_msg_type::EMPTY)
    {
        print();
    };

    UdpMessage(covise::TokenBuffer &tb);

	UdpMessage(covise::TokenBuffer&tb, udp_msg_type type);
    
    UdpMessage(const UdpMessage &); // copy constructor
    ~UdpMessage()
    {
		delete[] m_ip;
    };
    UdpMessage &operator=(const UdpMessage &); // assignment

    void print();

;
};
}
#endif
