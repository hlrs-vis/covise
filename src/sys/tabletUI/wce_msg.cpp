/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define DEFINE_MSG_TYPES
#include "wce_msg.h"

#define LOGINFO(x)
#include <util/byteswap.h>

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

Message::Message(covise::TokenBuffer *t)
    : type(COVISE_MESSAGE_EMPTY)
    , conn(NULL)
    , mustDelete(false)
{
    length = t->get_length();
    data = (char *)t->get_data();
    //printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
    print();
}

Message::Message(covise::TokenBuffer &t)
    : type(COVISE_MESSAGE_EMPTY)
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

covise::TokenBuffer &covise::TokenBuffer::operator<<(const uint64_t i)
{
    if (buflen < length + 9)
        incbuf();
    *currdata = i & 0xff;
    currdata++;
    *currdata = (i >> 8) & 0xff;
    currdata++;
    *currdata = (i >> 16) & 0xff;
    currdata++;
    *currdata = (i >> 24) & 0xff;
    currdata++;
    *currdata = (i >> 32) & 0xff;
    currdata++;
    *currdata = (i >> 40) & 0xff;
    currdata++;
    *currdata = (i >> 48) & 0xff;
    currdata++;
    *currdata = (i >> 56) & 0xff;
    currdata++;
    length += 8;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator<<(const uint32_t i)
{
    if (buflen < length + 5)
        incbuf();
    *currdata = i & 0x000000ff;
    currdata++;
    *currdata = (i & 0x0000ff00) >> 8;
    currdata++;
    *currdata = (i & 0x00ff0000) >> 16;
    currdata++;
    *currdata = (i & 0xff000000) >> 24;
    currdata++;
    length += 4;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator<<(const std::string &s)
{
    uint32_t slen = s.length() + 1;
    if (buflen < length + slen)
        incbuf(slen);
    memcpy(currdata, s.c_str(), slen);
    currdata += slen;
    length += slen;
    return *this;
}

covise::TokenBuffer &covise::TokenBuffer::operator<<(covise::TokenBuffer *t)
{
    if (buflen < length + t->get_length() + 1)
        incbuf(t->get_length() * 4);
    memcpy(currdata, t->get_data(), t->get_length());
    currdata += t->get_length();
    length += t->get_length();
    return (*this);
}

void covise::TokenBuffer::incbuf(int size)
{
    buflen += size;
    char *nb = new char[buflen];
    if (data)
        memcpy(nb, data, length);
    delete[] data;
    data = nb;
    currdata = data + length;
}

void covise::TokenBuffer::delete_data()
{
    if (buflen)
        delete[] data;
    buflen = 0;
    length = 0;
    data = NULL;
    currdata = NULL;
}

covise::TokenBuffer &covise::TokenBuffer::operator<<(const double f)
{
    const uint64_t *i = (const uint64_t *)(void *)&f;

    if (buflen < length + 8)
        incbuf();

    *currdata = *i & 0x00000000000000ffLL;
    currdata++;
    *currdata = (*i & 0x000000000000ff00LL) >> 8;
    currdata++;
    *currdata = (*i & 0x0000000000ff0000LL) >> 16;
    currdata++;
    *currdata = (*i & 0x00000000ff000000LL) >> 24;
    currdata++;
    *currdata = (*i & 0x000000ff00000000LL) >> 32;
    currdata++;
    *currdata = (*i & 0x0000ff0000000000LL) >> 40;
    currdata++;
    *currdata = (*i & 0x00ff000000000000LL) >> 48;
    currdata++;
    *currdata = (*i & 0xff00000000000000LL) >> 56;
    currdata++;

    length += 8;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator>>(double &f)
{
    uint64_t *i = (uint64_t *)(void *)&f;
    *i = *(unsigned char *)currdata;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 8;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 16;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 24;
    currdata++;
    *i |= (((uint64_t) * (unsigned char *)currdata)) << 32;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
    currdata++;
    *i |= ((uint64_t)(*(unsigned char *)currdata)) << 56;
    currdata++;
    length += 8;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator<<(const float f)
{
    const uint32_t *i = (const uint32_t *)(void *)&f;
    if (buflen < length + 4)
        incbuf();
    *currdata = *i & 0x000000ff;
    currdata++;
    *currdata = (*i & 0x0000ff00) >> 8;
    currdata++;
    *currdata = (*i & 0x00ff0000) >> 16;
    currdata++;
    *currdata = (*i & 0xff000000) >> 24;
    currdata++;

    length += 4;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator>>(float &f)
{

    uint32_t *i = (uint32_t *)(void *)&f;
    *i = *(unsigned char *)currdata;
    currdata++;
    *i |= (*(unsigned char *)currdata) << 8;
    currdata++;
    *i |= (*(unsigned char *)currdata) << 16;
    currdata++;
    *i |= (*(unsigned char *)currdata) << 24;
    currdata++;
    length += 4;
    return (*this);
}

float covise::TokenBuffer::get_float_token()
{

    uint32_t i;
    i = *(unsigned char *)currdata;
    currdata++;
    i |= (*(unsigned char *)currdata) << 8;
    currdata++;
    i |= (*(unsigned char *)currdata) << 16;
    currdata++;
    i |= (*(unsigned char *)currdata) << 24;
    currdata++;

    length += 4;
    return (*((float *)(void *)&i));
}

uint32_t covise::TokenBuffer::get_int_token()
{

    uint32_t i;
    i = *(unsigned char *)currdata;
    currdata++;
    i |= (*(unsigned char *)currdata) << 8;
    currdata++;
    i |= (*(unsigned char *)currdata) << 16;
    currdata++;
    i |= (*(unsigned char *)currdata) << 24;
    currdata++;

    length += 4;
    return (i);
}

covise::TokenBuffer &covise::TokenBuffer::operator>>(uint32_t &i)
{

    i = *(unsigned char *)currdata;
    currdata++;
    i |= (*(unsigned char *)currdata) << 8;
    currdata++;
    i |= (*(unsigned char *)currdata) << 16;
    currdata++;
    i |= (*(unsigned char *)currdata) << 24;
    currdata++;

    length += 4;
    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator>>(uint64_t &i)
{

    i = *(unsigned char *)currdata;
    currdata++;
    i |= (*(unsigned char *)currdata) << 8;
    currdata++;
    i |= (*(unsigned char *)currdata) << 16;
    currdata++;
    i |= (*(unsigned char *)currdata) << 24;
    currdata++;
    i |= ((uint64_t)(*(unsigned char *)currdata)) << 32;
    currdata++;
    i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
    currdata++;
    i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
    currdata++;
    i |= ((uint64_t)(*(unsigned char *)currdata)) << 56;
    currdata++;

    length += 8;

    return (*this);
}

covise::TokenBuffer &covise::TokenBuffer::operator>>(std::string &s)
{
    const char *c;
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
    s = c;
    return (*this);
}

char *Message::get_part(char *chdata)
{
    static char *data_ptr = NULL;
    char *part, *part_ptr;
    int i;

    if (chdata)
        data_ptr = chdata;
    if (data_ptr == NULL)
    {
        LOGINFO("no data to get part of");
        return NULL;
    }
    for (i = 0; data_ptr[i] != '\n'; i++)
        ;
    part_ptr = part = new char[i + 1];
    while ((*part_ptr++ = *data_ptr++) != '\n')
        ;
    *(part_ptr - 1) = '\0';
    return part;
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
