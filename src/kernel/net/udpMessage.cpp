/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "udpMessage.h"

#include <util/coErr.h>
#include <util/byteswap.h>
#include "tokenbuffer.h"


using namespace covise;

UdpMessage::UdpMessage(TokenBuffer &t)
    :MessageBase(t)
	,type(udp_msg_type::EMPTY)

{
}

UdpMessage::UdpMessage(covise::TokenBuffer& tb, udp_msg_type t)
	:UdpMessage(tb)
{
	type = t;
}

void UdpMessage::print()
{
#ifdef DEBUG
	cerr <<" udpMessage m_type = " udp_msg_types_vector[(int)m_type] << " m_sender = " << sender << " length = " << length << endl;
#endif
}

UdpMessage::UdpMessage(const UdpMessage &src)
{
    type = src.type;
    data = DataHandle(src.data.length());
    if (src.data.length() && src.data.data())
        memcpy(data.accessData(), src.data.data(), data.length());
    print();
}

UdpMessage &UdpMessage::operator=(const UdpMessage &src)
{
    //    printf("+ in message no. %d for %x, line %d\n", new_count++, this, __LINE__);
    //printf("+ in message no. %d for %p, line %d, m_type %d (%s)\n", 0, this, __LINE__, m_type, covise_msg_types_array[m_type]);

    // Check against self-assignment
    if (&src != this)
    {
        // always cope these
        sender = src.sender;
        type = src.type;

        data = DataHandle(src.data.length());

        if (src.data.length() && src.data.data())
            memcpy(data.accessData(), src.data.data(), data.length());
    }
    print();
    return *this;
}

