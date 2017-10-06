/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cassert>

#include "tokenbuffer.h"
#include "message.h"
#include "message_types.h"

#include <util/coErr.h>
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

TokenBuffer::TokenBuffer(Message *msg, bool nbo)
{
    assert(msg);
    if (msg->type == COVISE_MESSAGE_SOCKET_CLOSED)
    {
        std::cerr << "TokenBuffer: cannot handle SOCKET_CLOSED message" << std::endl;
    }
    buflen = 0;
    length = msg->length;
    data = currdata = msg->data;
    networkByteOrder = nbo;
}

TokenBuffer::TokenBuffer(const char *dat, int len, bool nbo)
{
    buflen = 0;
    length = len;
    data = currdata = (char *)dat;
    networkByteOrder = nbo;
}

TokenBuffer &TokenBuffer::operator<<(const uint64_t i)
{
    if (buflen < length + 9)
        incbuf();
    if (networkByteOrder)
    {
        *currdata = (i >> 56) & 0xff;
        currdata++;
        *currdata = (i >> 48) & 0xff;
        currdata++;
        *currdata = (i >> 40) & 0xff;
        currdata++;
        *currdata = (i >> 32) & 0xff;
        currdata++;
        *currdata = (i >> 24) & 0xff;
        currdata++;
        *currdata = (i >> 16) & 0xff;
        currdata++;
        *currdata = (i >> 8) & 0xff;
        currdata++;
        *currdata = i & 0xff;
        currdata++;
    }
    else
    {
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
    }
    length += 8;
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const uint32_t i)
{
    if (networkByteOrder)
    {
        if (buflen < length + 5)
            incbuf();
        *currdata = (i & 0xff000000) >> 24;
        currdata++;
        *currdata = (i & 0x00ff0000) >> 16;
        currdata++;
        *currdata = (i & 0x0000ff00) >> 8;
        currdata++;
        *currdata = i & 0x000000ff;
        currdata++;
    }
    else
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
    }
    length += 4;
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const std::string &s)
{
    int slen = (int)s.length() + 1;
    if (buflen < length + slen)
        incbuf(slen);
    memcpy(currdata, s.c_str(), slen);
    currdata += slen;
    length += slen;
    return *this;
}

TokenBuffer &TokenBuffer::operator<<(const TokenBuffer *t)
{
    if (buflen < length + t->get_length() + 1)
        incbuf(t->get_length() * 4);
    memcpy(currdata, t->get_data(), t->get_length());
    currdata += t->get_length();
    length += t->get_length();
    return (*this);
}

void TokenBuffer::incbuf(int size)
{
    buflen += size;
    char *nb = new char[buflen];
    if (data)
        memcpy(nb, data, length);
    delete[] data;
    data = nb;
    currdata = data + length;
}

void TokenBuffer::delete_data()
{
    if (buflen)
        delete[] data;
    buflen = 0;
    length = 0;
    data = NULL;
    currdata = NULL;
}

TokenBuffer::~TokenBuffer()
{
    if (buflen)
        delete[] data;
}
TokenBuffer &TokenBuffer::operator<<(const double f)
{
    const uint64_t *i = (const uint64_t *)(void *)&f;

    if (buflen < length + 8)
        incbuf();

    if (networkByteOrder)
    {

        *currdata = char((*i & 0xff00000000000000LL) >> 56);
        currdata++;
        *currdata = char((*i & 0x00ff000000000000LL) >> 48);
        currdata++;
        *currdata = char((*i & 0x0000ff0000000000LL) >> 40);
        currdata++;
        *currdata = char((*i & 0x000000ff00000000LL) >> 32);
        currdata++;
        *currdata = char((*i & 0x00000000ff000000LL) >> 24);
        currdata++;
        *currdata = char((*i & 0x0000000000ff0000LL) >> 16);
        currdata++;
        *currdata = char((*i & 0x000000000000ff00LL) >> 8);
        currdata++;
        *currdata = char(*i & 0x00000000000000ffLL);
        currdata++;
    }
    else
    {
        *currdata = char(*i & 0x00000000000000ffLL);
        currdata++;
        *currdata = char((*i & 0x000000000000ff00LL) >> 8);
        currdata++;
        *currdata = char((*i & 0x0000000000ff0000LL) >> 16);
        currdata++;
        *currdata = char((*i & 0x00000000ff000000LL) >> 24);
        currdata++;
        *currdata = char((*i & 0x000000ff00000000LL) >> 32);
        currdata++;
        *currdata = char((*i & 0x0000ff0000000000LL) >> 40);
        currdata++;
        *currdata = char((*i & 0x00ff000000000000LL) >> 48);
        currdata++;
        *currdata = char((*i & 0xff00000000000000LL) >> 56);
        currdata++;
    }

    length += 8;
    return (*this);
}

#define CHECK(size_type, error_ret)                                                                                       \
    do                                                                                                                    \
    {                                                                                                                     \
        if (currdata + sizeof(size_type) > data + length)                                                                 \
        {                                                                                                                 \
            std::cerr << "TokenBuffer: read past end (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;               \
            std::cerr << "  required: " << sizeof(size_type) << ", available: " << data + length - currdata << std::endl; \
            assert(0 == "read past end");                                                                                 \
            return error_ret;                                                                                             \
        }                                                                                                                 \
    } while (false)

TokenBuffer &TokenBuffer::operator>>(double &f)
{
    f = 0.;
    CHECK(f, *this);

    uint64_t *i = (uint64_t *)(void *)&f;
    if (networkByteOrder)
    {
        *i = ((uint64_t)(*(unsigned char *)currdata)) << 56;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 32;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 24;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 16;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 8;
        currdata++;
        *i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        *i = *(unsigned char *)currdata;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 8;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 16;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 24;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 32;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
        currdata++;
        *i |= ((uint64_t)(*(unsigned char *)currdata)) << 56;
        currdata++;
    }
    length += 8;
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const float f)
{
    const uint32_t *i = (const uint32_t *)(void *)&f;
    if (buflen < length + 4)
        incbuf();
    if (networkByteOrder)
    {
        if (buflen < length + 5)
            incbuf();
        *currdata = (*i & 0xff000000) >> 24;
        currdata++;
        *currdata = (*i & 0x00ff0000) >> 16;
        currdata++;
        *currdata = (*i & 0x0000ff00) >> 8;
        currdata++;
        *currdata = *i & 0x000000ff;
        currdata++;
    }
    else
    {
        if (buflen < length + 5)
            incbuf();
        *currdata = *i & 0x000000ff;
        currdata++;
        *currdata = (*i & 0x0000ff00) >> 8;
        currdata++;
        *currdata = (*i & 0x00ff0000) >> 16;
        currdata++;
        *currdata = (*i & 0xff000000) >> 24;
        currdata++;
    }

    length += 4;
    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(float &f)
{
    f = 0.f;
    CHECK(f, *this);

    uint32_t *i = (uint32_t *)(void *)&f;
    if (networkByteOrder)
    {
        *i = (*(unsigned char *)currdata) << 24;
        currdata++;
        *i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        *i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        *i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        *i = *(unsigned char *)currdata;
        currdata++;
        *i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        *i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        *i |= (*(unsigned char *)currdata) << 24;
        currdata++;
    }
    length += 4;
    return (*this);
}

float TokenBuffer::get_float_token()
{
    CHECK(float, 0.f);

    uint32_t i;
    if (networkByteOrder)
    {
        i = (*(unsigned char *)currdata) << 24;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        i = *(unsigned char *)currdata;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 24;
        currdata++;
    }

    length += 4;
    return (*((float *)(void *)&i));
}

uint32_t TokenBuffer::get_int_token()
{
    CHECK(uint32_t, 0);

    uint32_t i;
    if (networkByteOrder)
    {
        i = (*(unsigned char *)currdata) << 24;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        i = *(unsigned char *)currdata;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 24;
        currdata++;
    }

    length += 4;
    return (i);
}

TokenBuffer &TokenBuffer::operator>>(uint32_t &i)
{
    i = 0;
    CHECK(i, *this);

    if (networkByteOrder)
    {
        i = (*(unsigned char *)currdata) << 24;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        i = *(unsigned char *)currdata;
        currdata++;
        i |= (*(unsigned char *)currdata) << 8;
        currdata++;
        i |= (*(unsigned char *)currdata) << 16;
        currdata++;
        i |= (*(unsigned char *)currdata) << 24;
        currdata++;
    }

    length += 4;
    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(uint64_t &i)
{
    i = 0;
    CHECK(i, *this);
    if (networkByteOrder)
    {
        i = ((uint64_t)(*(unsigned char *)currdata)) << 56;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 32;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 24;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 16;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 8;
        currdata++;
        i |= *(unsigned char *)currdata;
        currdata++;
    }
    else
    {
        i = *(unsigned char *)currdata;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 8;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 16;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 24;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 32;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 40;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 48;
        currdata++;
        i |= ((uint64_t)(*(unsigned char *)currdata)) << 56;
        currdata++;
    }

    length += 8;

    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(std::string &s)
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

void TokenBuffer::rewind()
{
    currdata = data;
}
