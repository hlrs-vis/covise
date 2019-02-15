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

//#define TB_DEBUG // define to enable debugging
#define TB_DEBUG_TAG // include whether debug mode is enabled in first byte of TokenBuffer, for interoperability with debug-enabled TokenBuffer's

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


TokenBuffer::TokenBuffer(const Message *msg, bool nbo)
{
#ifndef TB_DEBUG_TAG
#ifdef TB_DEBUG
    debug = true;
#endif
#endif
    //std::cerr << "new TokenBuffer(Message) " << this << ": debug=" << debug << std::endl;
    assert(msg);
    if (msg->type == COVISE_MESSAGE_SOCKET_CLOSED)
    {
        //std::cerr << "TokenBuffer: cannot handle SOCKET_CLOSED message" << std::endl;
    }
    buflen = 0;
    length = msg->length;
    data = currdata = msg->data;
    networkByteOrder = nbo;

    rewind();
}

TokenBuffer::TokenBuffer(const char *dat, int len, bool nbo)
{
#ifndef TB_DEBUG_TAG
#ifdef TB_DEBUG
    debug = true;
#endif
#endif
    //std::cerr << "new TokenBuffer(data,len) " << this << ": debug=" << debug << std::endl;

    buflen = 0;
    length = len;
    data = currdata = (char *)dat;
    networkByteOrder = nbo;

    rewind();
}

TokenBuffer::TokenBuffer()
{
#ifdef TB_DEBUG
    debug = true;
#endif
    //std::cerr << "new TokenBuffer() " << this << ": debug=" << debug << std::endl;

    buflen = length = 0;
    data = currdata = NULL;
    networkByteOrder = false;
}

TokenBuffer::TokenBuffer(bool nbo)
{
#ifdef TB_DEBUG
    debug = true;
#endif
    //std::cerr << "new TokenBuffer() " << this << ": debug=" << debug << std::endl;

    buflen = length = 0;
    data = currdata = NULL;
    networkByteOrder = nbo;
}

TokenBuffer::TokenBuffer(int al, bool nbo)
{
#ifdef TB_DEBUG
    debug = true;
#endif
    //std::cerr << "new TokenBuffer(size) " << this << ": debug=" << debug << std::endl;

#ifndef TB_DEBUG_TAG
    buflen = al+1;
    length = 1;
#else
    buflen = al;
    length = 0;
#endif
    data = currdata = new char[al];
#ifdef TB_DEBUG_TAG
    if (al >= 1)
    {
        data[0] = debug;
        ++currdata;
        ++length;
    }
#endif
    networkByteOrder = nbo;
}


const char *TokenBuffer::getBinary(int n)
{
    checktype(TbBinary);
    const char *c = currdata;
    currdata += n;
    return c;
}

void TokenBuffer::addBinary(const char *buf, int n)
{
    puttype(TbBinary);
    if (buflen < length + n + 1)
        incbuf(n + 40);
    memcpy(currdata, buf, n);
    currdata += n;
    length += n;
}

const char *TokenBuffer::allocBinary(int n)
{
    puttype(TbBinary);
    if (buflen < length + n + 1)
        incbuf(n + 40);
    const char *buf = currdata;
    currdata += n;
    length += n;
    return buf;
}

TokenBuffer &TokenBuffer::operator<<(const int i)
{
    *this << (uint32_t)i;
    return *this;
}

TokenBuffer &TokenBuffer::operator>>(char *&c)
{
    checktype(TbString);
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

TokenBuffer &TokenBuffer::operator<<(const char c)
{
    puttype(TbChar);
    if (buflen < length + 2)
        incbuf();
    *currdata = c;
    currdata++;
    length++;
    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(unsigned char &c)
{
    checktype(TbChar);
    CHECK(c, *this);
    c = *(unsigned char *)currdata;
    currdata++;
    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(char &c)
{
    checktype(TbChar);
    CHECK(c, *this);
    c = *(char *)currdata;
    currdata++;
    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(int &i)
{
    uint32_t u32;
    *this >> u32;
    i = *reinterpret_cast<int *>(&u32);
    return *this;
}


TokenBuffer &TokenBuffer::operator<<(const char *c)
{
    puttype(TbString);
    int l = int(strlen(c) + 1);
    if (buflen < length + l + 1)
        incbuf(l * 10);
    strcpy(currdata, c);
    currdata += l;
    length += l;
    return (*this);
}

TokenBuffer &TokenBuffer::operator=(TokenBuffer &&other)
{
    delete_data();

	data = other.data;
	other.data = nullptr;
	currdata = other.currdata;
	other.currdata = nullptr;
	buflen = other.buflen;
	other.buflen = 0;
	length = other.length;
	other.length = 0;
	networkByteOrder = other.networkByteOrder;
	other.networkByteOrder = false;

	return *this;
}
void TokenBuffer::copy(const TokenBuffer &other) {
    delete_data();
    char *nb = new char[other.length];
	memcpy(nb, other.get_data(), other.get_length());
	length = other.length;
    buflen = other.get_length();
    data = nb;
	networkByteOrder = other.networkByteOrder;
}

TokenBuffer &TokenBuffer::operator>>(bool &b)
{
    checktype(TbBool);

    char byte = 0;
    CHECK(byte, *this);
    (*this) >> byte;
    b = byte>0;
    return *this;
}

TokenBuffer &TokenBuffer::operator<<(const bool b)
{
    puttype(TbBool);

    char byte = b?1:0;
    return (*this) << byte;
}

TokenBuffer &TokenBuffer::operator<<(const uint64_t i)
{
    puttype(TbInt64);

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
    puttype(TbInt32);

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
    puttype(TbString);

    int slen = (int)s.length() + 1;
    if (buflen < length + slen)
        incbuf(slen);
    memcpy(currdata, s.c_str(), slen);
    currdata += slen;
    length += slen;
    return *this;
}

TokenBuffer &TokenBuffer::operator<<(const TokenBuffer &t)
{
    puttype(TbTB);

    uint32_t l = t.get_length();
    *this << l;

    if (buflen < length + l)
        incbuf(l);
    memcpy(currdata, t.get_data(), l);
    currdata += l;

    return *this;
}

TokenBuffer &TokenBuffer::operator>>(TokenBuffer &tb)
{
    checktype(TbTB);

    tb.delete_data();

    uint32_t l = 0;
    *this >> l;

    tb.data = new char[l];
    tb.length = l;
    tb.buflen = l;
    memcpy(tb.data, currdata, l);

    tb.rewind();

    return *this;
}

void TokenBuffer::puttype(TokenBuffer::Types t)
{
#ifdef TB_DEBUG
    //std::cerr << "TokenBuffer " << this << ", puttype: " << t << ", length=" << length << std::endl;
    assert(debug);

    if (buflen+1 >= length)
        incbuf();
    *currdata = (char)t;
    ++currdata;
    ++length;
    assert(data+length == currdata);
#else
    assert(!debug);
#endif
}

bool TokenBuffer::checktype(TokenBuffer::Types t)
{
    if (debug)
    {
        char tt = 0;
        CHECK(tt, false);
        //std::cerr << "TokenBuffer " << this << ", checktype: expecting " << t << ", have " << (int)*currdata << std::endl;
        assert(data+length > currdata);

        if (*currdata != t)
        {
            std::cerr << "TokenBuffer::checktype: ERROR: expecting " << t << ", have " << (int)*currdata << std::endl;
            assert(*currdata == t);
            abort();
            return false;
        }
        ++currdata;
    }

    return true;
}

void TokenBuffer::incbuf(int size)
{
    assert((buflen==0 && !data) || (buflen>0 && data));
    assert(buflen>0 || length==0);
    assert(!data || data+length == currdata);

    buflen += size;
#ifdef TB_DEBUG_TAG
    if (!data)
        buflen += 1;
#endif
    char *nb = new char[buflen];
    if (data)
    {
        memcpy(nb, data, length);
    }
    else
    {
#ifdef TB_DEBUG_TAG
        nb[0] = debug;
        length = 1;
#else
        length = 0;
#endif
    }

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
    puttype(TbDouble);

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

TokenBuffer &TokenBuffer::operator>>(double &f)
{
    checktype(TbDouble);

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
    puttype(TbFloat);

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
    checktype(TbFloat);

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
    checktype(TbFloat);
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

char *TokenBuffer::get_charp_token()
{
    checktype(TbString);
    char *ret = currdata;
    while (*currdata)
        currdata++;
    currdata++;
    return (ret);
}

uint32_t TokenBuffer::get_int_token()
{
    checktype(TbInt32);
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

char TokenBuffer::get_char_token()
{
    checktype(TbChar);
    char ret = *(char *)currdata;
    currdata++;
    return (ret);
}

TokenBuffer &TokenBuffer::operator>>(uint32_t &i)
{
    checktype(TbInt32);
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
    checktype(TbInt64);
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
    checktype(TbString);
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
    if (currdata)
    {
#ifdef TB_DEBUG_TAG
        debug = *currdata;
        ++currdata;
#endif
    }

    //std::cerr << "rewind TokenBuffer " << this << ": debug=" << debug << std::endl;
}

void TokenBuffer::reset()
{
    currdata = data;
    length = 0;
#ifdef TB_DEBUG
    debug = true;
#else
    debug = false;
#endif
#ifdef TB_DEBUG_TAG
    if (data && buflen > length)
    {
        data[length] = debug;
        ++currdata;
        ++length;
    }
#endif
    if (data && buflen > length)
        data[length] = 0;

    assert(!data || data+length == currdata);

    //std::cerr << "reset TokenBuffer " << this << ": debug=" << debug << std::endl;
}
