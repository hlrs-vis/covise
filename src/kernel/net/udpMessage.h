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


//basically this is a covise message with smaler header using udp protocoll
//
namespace covise
{
class TokenBuffer;

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
