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

#define TB_DEBUG // define to enable debugging
#define TB_DEBUG_TAG // include whether debug mode is enabled in first byte of TokenBuffer, for interoperability with debug-enabled TokenBuffer's

#define CHECK(size_type, error_ret)                                                                                       \
    do                                                                                                                    \
    {                                                                                                                     \
        if (currdata + sizeof(size_type) > data.end())                                                                 \
        {                                                                                                                 \
            std::cerr << "TokenBuffer: read past end (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;               \
            std::cerr << "  required: " << sizeof(size_type) << ", available: " << data.end() - currdata << std::endl; \
            assert(0 == "read past end");                                                                                 \
            abort();                                                                                                      \
            return error_ret;                                                                                             \
        }                                                                                                                 \
    } while (false)



TokenBuffer::TokenBuffer(const MessageBase *msg, bool nbo)
{
#ifndef TB_DEBUG_TAG
#ifdef TB_DEBUG
    debug = true;
#endif
#endif
    //std::cerr << "new TokenBuffer(Message) " << this << ": debug=" << debug << std::endl;
    assert(msg);

    buflen = 0;
    data = msg->data;
    currdata = data.accessData();
    networkByteOrder = nbo;

    rewind();
}
TokenBuffer::TokenBuffer(const DataHandle& dh, bool nbo)
{
#ifndef TB_DEBUG_TAG
#ifdef TB_DEBUG
    debug = true;
#endif
#endif
    buflen = 0;
    data = dh;
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
    data = DataHandle{ (char*)dat, len, false };
    currdata = (char *)dat;
    networkByteOrder = nbo;

    rewind();
}

TokenBuffer::TokenBuffer()
{
#ifdef TB_DEBUG
    debug = true;
#endif
    //std::cerr << "new TokenBuffer() " << this << ": debug=" << debug << std::endl;

    networkByteOrder = false;
}

TokenBuffer::TokenBuffer(bool nbo)
{
#ifdef TB_DEBUG
    debug = true;
#endif
    //std::cerr << "new TokenBuffer() " << this << ": debug=" << debug << std::endl;
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
    data = DataHandle(new char[al+1], 0);
    if (al >= 1)
    {
        data.accessData()[0] = debug;
        currdata = data.accessData() + 1;
        ++length;
    }
#else
    buflen = al;
    data = DataHandle(new char[al], 0);
    currdata = data.accessData();
#endif

    networkByteOrder = nbo;
}


const DataHandle& covise::TokenBuffer::getData()
{
    return data;
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
    if (buflen < data.length() + n + 1)
        incbuf(n + 40);
    memcpy(currdata, buf, n);
    currdata += n;
    data.incLength(n);
}

const char *TokenBuffer::allocBinary(int n)
{
    puttype(TbBinary);
    if (buflen < data.length() + n + 1)
        incbuf(n + 40);
    const char *buf = currdata;
    currdata += n;
    data.incLength(n);
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
    char *end = data.accessData() + data.length() - 1;
    c = currdata;
    while (*currdata)
    {
        currdata++;
        if (currdata > end)
        {
            std::cerr << "TokenBuffer: string not terminated within range" << std::endl;
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
    if (buflen < data.length() + 2)
        incbuf();
    *currdata = c;
    currdata++;
    data.incLength(1);
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
    int l = 1;
    if (c)
        l = int(strlen(c) + 1);
    if (buflen < data.length() + l + 1)
        incbuf(l * 10);
    strcpy(currdata, c ? c : "");
    currdata += l;
    data.incLength(l);
    return (*this);
}

TokenBuffer &TokenBuffer::operator=(const TokenBuffer &other)
{
	data = other.data;
	currdata = other.currdata;
	buflen = other.buflen;
	networkByteOrder = other.networkByteOrder;
	return *this;
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

    if (buflen < data.length() + 9)
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
    data.incLength(8);
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const uint32_t i)
{
    puttype(TbInt32);

    if (networkByteOrder)
    {
        if (buflen < data.length() + 5)
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
        if (buflen < data.length() + 5)
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
    data.incLength(4);
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const std::string &s)
{
    puttype(TbString);

    int slen = (int)s.length() + 1;
    if (buflen < data.length() + slen)
        incbuf(slen);
    memcpy(currdata, s.c_str(), slen);
    currdata += slen;
    data.incLength(slen);
    return *this;
}

TokenBuffer& TokenBuffer::operator<<(const DataHandle& d)
{
    puttype(TbTB);
    *this << d.length();
    this->addBinary(d.data(), d.length());
    return *this;
}

TokenBuffer& TokenBuffer::operator>>(DataHandle& d)
{
    checktype(TbTB);
    int l;
    *this >> l;
    //d = data;
    //d.movePtr(currdata - data.data());
    //d.setLength(l);
    //this->getBinary(l);

    char* c = new char[l];
    memcpy(c, this->getBinary(l), l);
    d = DataHandle(c, l);
    return *this;
}

TokenBuffer &TokenBuffer::operator<<(const TokenBuffer &t)
{
    *this << t.data;
    return *this;
}

TokenBuffer &TokenBuffer::operator>>(TokenBuffer &tb)
{
    *this >> tb.data;
    tb.rewind();
    return *this;
}

void TokenBuffer::puttype(TokenBuffer::Types t)
{
#ifdef TB_DEBUG
    //std::cerr << "TokenBuffer " << this << ", puttype: " << t << ", length=" << length << std::endl;
    assert(debug);

    if (buflen <= data.length() + 1)
        incbuf();
    *currdata = (char)t;
    ++currdata;
    data.incLength(1);
    assert(data.end() == currdata);
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
        assert(data.end() > currdata);

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
    assert((buflen==0 && !data.data()) || (buflen>0 && data.data()));
    assert(!data.data() || data.end() == currdata);

    buflen += size;
#ifdef TB_DEBUG_TAG
    if (!data.data())
        buflen += 1;
#endif
    DataHandle nb(buflen);
    if (data.data())
    {
        memcpy(nb.accessData(), data.data(), data.length());
        nb.setLength(data.length());
    }
    else
    {
#ifdef TB_DEBUG_TAG
        nb.accessData()[0] = debug;
        nb.setLength(1);

#else
        nb.setLength(0);
#endif
    }
    data = nb;
    currdata = data.end();
}

TokenBuffer::~TokenBuffer()
{
}
TokenBuffer &TokenBuffer::operator<<(const double f)
{
    puttype(TbDouble);

    const uint64_t *i = (const uint64_t *)(void *)&f;

    if (buflen < data.length() + 8)
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

    data.incLength(8);
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
    return (*this);
}

TokenBuffer &TokenBuffer::operator<<(const float f)
{
    puttype(TbFloat);

    const uint32_t *i = (const uint32_t *)(void *)&f;
    if (buflen < data.length() + 4)
        incbuf();
    if (networkByteOrder)
    {
        if (buflen < data.length() + 5)
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
        if (buflen < data.length() + 5)
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

    data.incLength(4);
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

    data.incLength(8);

    return (*this);
}

TokenBuffer &TokenBuffer::operator>>(std::string &s)
{
    checktype(TbString);
    char *end = data.end() - 1;
    const char *c = currdata;
    while (*currdata)
    {
        currdata++;
        if (currdata > end)
        {
            std::cerr << "TokenBuffer: string not terminated within range" << std::endl;
            s = c;
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
    currdata = data.accessData();
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
    data = DataHandle();
    currdata = data.accessData();
    buflen = 0;
#ifdef TB_DEBUG
    debug = true;
#else
    debug = false;
#endif
        assert(!data.data() || data.end() == currdata);

    //std::cerr << "reset TokenBuffer " << this << ": debug=" << debug << std::endl;
}
