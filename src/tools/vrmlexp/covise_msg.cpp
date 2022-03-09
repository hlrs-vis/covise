/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTypes.h"
#define DEFINE_MSG_TYPES
#include "covise_msg.h"
//#include "covise_shm.h"
//#include "covise_process.h"

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

#undef DEBUG
MessageBase::MessageBase()
	: conn(nullptr)
{

}
MessageBase::MessageBase(TokenBuffer& tb)
{
	data = tb.getData();
	//printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
}
MessageBase::MessageBase(DataHandle& dh)
{
	data = dh;
	//printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
}

MessageBase::~MessageBase()
{
}


Message::Message(TokenBuffer& t)
	:MessageBase(t)
	, type(Message::EMPTY)
{
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

Message::Message(const Message& src)
{
	//    printf("+ in message no. %d for %x, line %d\n", new_count++, this, __LINE__);
	//printf("+ in message no. %d for %p, line %d, type %d (%s)\n", 0, this, __LINE__, type, covise_msg_types_array[type]);
	sender = src.sender;
	send_type = src.send_type;
	type = src.type;
	int length = src.data.length();
	char* c = new char[length];
	memcpy(c, src.data.data(), length);
	data = DataHandle(c, length);
	conn = src.conn;
	print();
}
Message::Message(int message_type, const DataHandle& dh)
	:sender(-1)
	, send_type(Message::UNDEFINED)
	, type(message_type)
{
	data = dh;
}
Message& Message::operator=(const Message& src)
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
		conn = src.conn;

		data = DataHandle(src.data.length());
		memcpy(data.accessData(), src.data.data(), data.length());

	}
	print();
	return *this;
}

void Message::copyAndReuseData(const Message& src)
{
	sender = src.sender;
	send_type = src.send_type;
	type = src.type;
	conn = src.conn;
	data = src.data;
}
//char *Message::extract_data()
//{
//    char *tmpdata = data;
//    data = NULL;
//    return tmpdata;
//}
/*
bool isVrbMessageType(int type)
{
    switch (type)
    {
    case COVISE_MESSAGE_VRB_REQUEST_FILE:
    case COVISE_MESSAGE_VRB_SEND_FILE:
    case DEPRECATED_COVISE_MESSAGE_VRB_CURRENT_FILE:
    case COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE: // Set Registry value
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS:
    case COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE:
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS:
    case COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE:
    case COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY:
    case COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY:
    case COVISE_MESSAGE_VRB_CONTACT:
    case COVISE_MESSAGE_VRB_CONNECT_TO_COVISE:
    case COVISE_MESSAGE_VRB_SET_USERINFO:
    case COVISE_MESSAGE_RENDER:
    case COVISE_MESSAGE_RENDER_MODULE: // send Message to all others in same group
    case COVISE_MESSAGE_VRB_CHECK_COVER:
    case COVISE_MESSAGE_VRB_GET_ID:
    case COVISE_MESSAGE_VRB_SET_GROUP:
    case COVISE_MESSAGE_VRB_SET_MASTER:
    case COVISE_MESSAGE_SOCKET_CLOSED:
    case COVISE_MESSAGE_CLOSE_SOCKET:
    case COVISE_MESSAGE_VRB_FB_RQ:
    case COVISE_MESSAGE_VRB_FB_REMREQ:
        return true;

    default:
        return false;
    }

    return false;
}

*/
DataHandle::DataHandle()
	:m_ManagedData(nullptr)
	, m_length(0)
	, m_dataPtr(nullptr)
{
}

DataHandle::~DataHandle()
{
}

DataHandle::DataHandle(char* data, const size_t length, bool doDelete)
	: DataHandle(data, static_cast<int>(length), doDelete)
{
}
DataHandle::DataHandle(char* data, const int length, bool doDelete)
	: m_length(length)
{
	if (doDelete)
	{
		m_ManagedData.reset(data, std::default_delete<char[]>());
		m_dataPtr = data;
	}
	else
	{
		m_dataPtr = data;
	}
}

DataHandle::DataHandle(size_t size)
{
	m_ManagedData.reset(new char[size], std::default_delete<char[]>());
	m_dataPtr = m_ManagedData.get();
	m_length = static_cast<int>(size);
}


const char* DataHandle::data() const
{
	if (m_ManagedData.get() && m_dataPtr)
	{
		checkPtr();
		return m_dataPtr;
	}
	else if (m_dataPtr && !m_ManagedData.get())
	{
		return m_dataPtr;
	}
	else if (!m_dataPtr && m_ManagedData.get())
	{
		return m_ManagedData.get();
	}
	else
	{
		return nullptr;
	}
}
char* DataHandle::accessData()
{
	return const_cast<char*>(data());
}

const int DataHandle::length() const
{
	return m_length;
}


const char* DataHandle::end() const
{
	return data() + m_length;
}

char* DataHandle::end()
{
	return accessData() + m_length;
}

void DataHandle::setLength(const int l)
{
	m_length = l;
}

void DataHandle::incLength(const int inc)
{
	m_length += inc;
	assert(m_length > sizeof(*m_ManagedData));
}

void DataHandle::movePtr(int amount)
{
	m_dataPtr += amount;
	checkPtr();
}

void DataHandle::checkPtr() const
{

}


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



TokenBuffer::TokenBuffer(const MessageBase* msg, bool nbo)
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


TokenBuffer::TokenBuffer(const char* dat, int len, bool nbo)
{
#ifndef TB_DEBUG_TAG
#ifdef TB_DEBUG
	debug = true;
#endif
#endif
	//std::cerr << "new TokenBuffer(data,len) " << this << ": debug=" << debug << std::endl;

	buflen = 0;
	data = DataHandle{ (char*)dat, len, false };
	currdata = (char*)dat;
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
	buflen = al + 1;
	data = DataHandle(new char[al + 1], 0);
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


const DataHandle& TokenBuffer::getData()
{
	return data;
}

const char* TokenBuffer::getBinary(int n)
{
	checktype(TbBinary);
	const char* c = currdata;
	currdata += n;
	return c;
}

void TokenBuffer::addBinary(const char* buf, int n)
{
	puttype(TbBinary);
	if (buflen < data.length() + n + 1)
		incbuf(n + 40);
	memcpy(currdata, buf, n);
	currdata += n;
	data.incLength(n);
}

const char* TokenBuffer::allocBinary(int n)
{
	puttype(TbBinary);
	if (buflen < data.length() + n + 1)
		incbuf(n + 40);
	const char* buf = currdata;
	currdata += n;
	data.incLength(n);
	return buf;
}

TokenBuffer& TokenBuffer::operator<<(const int i)
{
	*this << (uint32_t)i;
	return *this;
}

TokenBuffer& TokenBuffer::operator>>(char*& c)
{
	checktype(TbString);
	char* end = data.accessData() + data.length() - 1;
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

TokenBuffer& TokenBuffer::operator<<(const char c)
{
	puttype(TbChar);
	if (buflen < data.length() + 2)
		incbuf();
	*currdata = c;
	currdata++;
	data.incLength(1);
	return (*this);
}

TokenBuffer& TokenBuffer::operator>>(unsigned char& c)
{
	checktype(TbChar);
	CHECK(c, *this);
	c = *(unsigned char*)currdata;
	currdata++;
	return (*this);
}

TokenBuffer& TokenBuffer::operator>>(char& c)
{
	checktype(TbChar);
	CHECK(c, *this);
	c = *(char*)currdata;
	currdata++;
	return (*this);
}

TokenBuffer& TokenBuffer::operator>>(int& i)
{
	uint32_t u32;
	*this >> u32;
	i = *reinterpret_cast<int*>(&u32);
	return *this;
}


TokenBuffer& TokenBuffer::operator<<(const char* c)
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

TokenBuffer& TokenBuffer::operator=(const TokenBuffer& other)
{
	data = other.data;
	currdata = other.currdata;
	buflen = other.buflen;
	networkByteOrder = other.networkByteOrder;
	return *this;
}

TokenBuffer& TokenBuffer::operator>>(bool& b)
{
    checktype(TbBool);

    char byte = 0;
    CHECK(byte, *this);
    byte = *(char*)currdata;
    currdata++;
    b = byte > 0;
    return *this;
}

TokenBuffer& TokenBuffer::operator<<(const bool b)
{
    puttype(TbBool);

    if (buflen < data.length() + 2)
        incbuf();
    *currdata = b ? 1 : 0;
    currdata++;
    data.incLength(1);
    return *this;
}

TokenBuffer& TokenBuffer::operator<<(const uint64_t i)
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

TokenBuffer& TokenBuffer::operator<<(const uint32_t i)
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

TokenBuffer& TokenBuffer::operator<<(const std::string& s)
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

TokenBuffer& TokenBuffer::operator<<(const TokenBuffer& t)
{
	*this << t.data;
	return *this;
}

TokenBuffer& TokenBuffer::operator>>(TokenBuffer& tb)
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
	assert((buflen == 0 && !data.data()) || (buflen > 0 && data.data()));
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
TokenBuffer& TokenBuffer::operator<<(const double f)
{
	puttype(TbDouble);

	const uint64_t* i = (const uint64_t*)(void*)&f;

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

TokenBuffer& TokenBuffer::operator>>(double& f)
{
	checktype(TbDouble);

	f = 0.;
	CHECK(f, *this);

	uint64_t* i = (uint64_t*)(void*)&f;
	if (networkByteOrder)
	{
		*i = ((uint64_t)(*(unsigned char*)currdata)) << 56;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 48;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 40;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 32;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 24;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 16;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 8;
		currdata++;
		*i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		*i = *(unsigned char*)currdata;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 8;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 16;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 24;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 32;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 40;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 48;
		currdata++;
		*i |= ((uint64_t)(*(unsigned char*)currdata)) << 56;
		currdata++;
	}
	return (*this);
}

TokenBuffer& TokenBuffer::operator<<(const float f)
{
	puttype(TbFloat);

	const uint32_t* i = (const uint32_t*)(void*)&f;
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

TokenBuffer& TokenBuffer::operator>>(float& f)
{
	checktype(TbFloat);

	f = 0.f;
	CHECK(f, *this);

	uint32_t* i = (uint32_t*)(void*)&f;
	if (networkByteOrder)
	{
		*i = (*(unsigned char*)currdata) << 24;
		currdata++;
		*i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		*i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		*i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		*i = *(unsigned char*)currdata;
		currdata++;
		*i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		*i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		*i |= (*(unsigned char*)currdata) << 24;
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
		i = (*(unsigned char*)currdata) << 24;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		i = *(unsigned char*)currdata;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 24;
		currdata++;
	}
	return (*((float*)(void*)&i));
}

char* TokenBuffer::get_charp_token()
{
	checktype(TbString);
	char* ret = currdata;
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
		i = (*(unsigned char*)currdata) << 24;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		i = *(unsigned char*)currdata;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 24;
		currdata++;
	}
	return (i);
}

char TokenBuffer::get_char_token()
{
	checktype(TbChar);
	char ret = *(char*)currdata;
	currdata++;
	return (ret);
}

TokenBuffer& TokenBuffer::operator>>(uint32_t& i)
{
	checktype(TbInt32);
	i = 0;
	CHECK(i, *this);

	if (networkByteOrder)
	{
		i = (*(unsigned char*)currdata) << 24;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		i = *(unsigned char*)currdata;
		currdata++;
		i |= (*(unsigned char*)currdata) << 8;
		currdata++;
		i |= (*(unsigned char*)currdata) << 16;
		currdata++;
		i |= (*(unsigned char*)currdata) << 24;
		currdata++;
	}
	return (*this);
}

TokenBuffer& TokenBuffer::operator>>(uint64_t& i)
{
	checktype(TbInt64);
	i = 0;
	CHECK(i, *this);
	if (networkByteOrder)
	{
		i = ((uint64_t)(*(unsigned char*)currdata)) << 56;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 48;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 40;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 32;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 24;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 16;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 8;
		currdata++;
		i |= *(unsigned char*)currdata;
		currdata++;
	}
	else
	{
		i = *(unsigned char*)currdata;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 8;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 16;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 24;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 32;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 40;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 48;
		currdata++;
		i |= ((uint64_t)(*(unsigned char*)currdata)) << 56;
		currdata++;
	}

	data.incLength(8);

	return (*this);
}

TokenBuffer& TokenBuffer::operator>>(std::string& s)
{
	checktype(TbString);
	char* end = data.end() - 1;
	const char* c = currdata;
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
/*
ShmMessage::ShmMessage(data_type *d, long *count, int no)
    : Message()
{
    int i, j;
#ifdef DEBUG
    char tmp_str[255];
#endif

    conn = 0L;
    type = SHM_MALLOC_LIST;
#ifdef DEBUG
    sprintf(tmp_str, "ShmMessage::ShmMessage");
    //print_comment(__LINE__, __FILE__, tmp_str);
    sprintf(tmp_str, "no: %d", no);
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
    length = no * (sizeof(data_type) + sizeof(long));
    data = new char[length];
    for (i = 0, j = 0; i < no; i++)
    {
        *(data_type *)(data + j) = d[i];
#ifdef DEBUG
        sprintf(tmp_str, "data_type: %d", *(data_type *)(data + j));
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
        j += sizeof(data_type);
        *(long *)(data + j) = count[i];
#ifdef DEBUG
        sprintf(tmp_str, "size: %d", *(long *)(data + j));
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
        j += sizeof(long);
    }
#ifdef DEBUG
    sprintf(tmp_str, "j: %d", j);
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
}

ShmMessage::ShmMessage(char *name, int otype, data_type *d, long *count, int no)
    : Message()
{
    int i, j;
    char *tmp_data;
    int start_data;
#ifdef DEBUG
    char tmp_str[255];
#endif

    conn = 0L;
    type = NEW_OBJECT_SHM_MALLOC_LIST;
#ifdef DEBUG
    sprintf(tmp_str, "ShmMessage::ShmMessage");
    //print_comment(__LINE__, __FILE__, tmp_str);
    sprintf(tmp_str, "no: %d", no);
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
    length = (int)strlen(name) + 1;
    if (length % SIZEOF_ALIGNMENT)
        start_data = sizeof(long) + (length / SIZEOF_ALIGNMENT + 1) * SIZEOF_ALIGNMENT;
    else
        start_data = sizeof(long) + length;
    length = sizeof(long) + start_data + no * (sizeof(data_type) + sizeof(long));
    data = new char[length];
    *(int *)data = otype;
    strcpy(&data[sizeof(int)], name);
    tmp_data = data + start_data;
    for (i = 0, j = 0; i < no; i++)
    {
        *(data_type *)(tmp_data + j) = d[i];
#ifdef DEBUG
        sprintf(tmp_str, "data_type: %d", *(data_type *)(tmp_data + j));
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
        j += sizeof(data_type);
        *(long *)(tmp_data + j) = count[i];
#ifdef DEBUG
        sprintf(tmp_str, "size: %d", *(long *)(tmp_data + j));
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
        j += sizeof(long);
    }
#ifdef DEBUG
    sprintf(tmp_str, "j: %d", j);
//print_comment(__LINE__, __FILE__, tmp_str);
#endif
}

void CtlMessage::init_list()
{
    char *tmpchptr = NULL;
    char **tmpcharlist;
    char *tmpname = NULL;
    char *tmptype = NULL;
    char *tmpstring = NULL;
    int no, i, j, sel, len;
    //char tmp_str[255];

    m_name = NULL;
    char *p = data;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        m_name = new char[strlen(tmpchptr) + 1];
        strcpy(m_name, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  m_name " << m_name << endl;

    inst_no = NULL;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        inst_no = new char[strlen(tmpchptr) + 1];
        strcpy(inst_no, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  inst_no " << inst_no << endl;

    h_name = NULL;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        h_name = new char[strlen(tmpchptr) + 1];
        strcpy(h_name, tmpchptr);
    }

    //   cerr << "   CtlMessage::init_list()  h_name " << h_name << endl;

    no_of_objects = 0;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        no_of_objects = atoi(tmpchptr);
    }
    else
    {
        if (m_name)
            no_of_objects = 1;
    }

    no_of_params_in = 0;
    tmpchptr = strsep(&p, "\n");
    if (tmpchptr)
    {
        no_of_params_in = atoi(tmpchptr);
    }

    port_names = new char *[no_of_objects];
    object_names = new char *[no_of_objects];
    port_connected = new int[no_of_objects];
    object_types = new char *[no_of_objects];
    params_in = new Param *[no_of_params_in];

    for (i = 0; i < no_of_objects; i++)
    {
        tmpchptr = strsep(&p, "\n");
        port_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(port_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", port_names[i]);
        ////print_comment(__LINE__, __FILE__, tmp_str);

        tmpchptr = strsep(&p, "\n");
        object_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_names[i]);
        ////print_comment(__LINE__, __FILE__, tmp_str);

        tmpchptr = strsep(&p, "\n");
        object_types[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_types[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_types[i]);
        ////print_comment(__LINE__, __FILE__, tmp_str);

        // new aw 25-07-2000 "CONNECTED" message
        tmpchptr = strsep(&p, "\n");
        port_connected[i] = (strcmp(tmpchptr, "CONNECTED") == 0);
    }

    for (i = 0; i < no_of_params_in; i++)
    {
        tmpname = strsep(&p, "\n");
        tmptype = strsep(&p, "\n");

        if (!strcmp(tmptype, "FloatScalar"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamFloatScalar(tmpname, tmpstring);
            delete[] tmpstring;
        }
        else if (!strcmp(tmptype, "IntScalar"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamIntScalar(tmpname, tmpstring);
            delete[] tmpstring;
        }
        else if (!strcmp(tmptype, "FloatSlider"))
        {
            no = atoi(strsep(&p, "\n"));
            if (no != 3)
            {
                //print_comment(__LINE__, __FILE__, "wrong number of slider-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamFloatSlider(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "IntSlider"))
        {
            no = atoi(strsep(&p, "\n"));
            if (no != 3)
            {
                //print_comment(__LINE__, __FILE__, "wrong number of slider-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamIntSlider(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "String"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamString(tmpname, tmpstring);
            delete[] tmpstring;
        }
        else if (!strcmp(tmptype, "Choice"))
        {
            no = atoi(strsep(&p, "\n"));
            sel = atoi(strsep(&p, "\n")); // the selected no
            tmpcharlist = new char *[no - 1];
            for (j = 0; j < no - 1; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamChoice(tmpname, no - 1, sel, tmpcharlist);
            for (j = 0; j < no - 1; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Cli"))
        {
            // command line interface
            no = atoi(strsep(&p, "\n"));
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamCli(tmpname, tmpstring);
            delete[] tmpstring;
        }
        else if (!strcmp(tmptype, "Colormap"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamColormap(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Browser"))
        {
            no = atoi(strsep(&p, "\n"));
            if (no != 2)
            {
                //print_comment(__LINE__, __FILE__, "wrong number of browser-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }

            int len = (int)strlen(tmpcharlist[0]);
            if (len > 1 && (tmpcharlist[0][0] == '"') && (tmpcharlist[0][len - 1] == '"')) // we have a string with quotes, so remove them
            {
                strcpy(tmpcharlist[0], tmpcharlist[0] + 1);
                tmpcharlist[0][len - 2] = '\0';
            }
            params_in[i] = (Param *)new ParamBrowser(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Boolean"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpchptr = strsep(&p, "\n");
            tmpstring = new char[strlen(tmpchptr) + 1];
            strcpy(tmpstring, tmpchptr);
            params_in[i] = (Param *)new ParamBoolean(tmpname, tmpstring);
            delete[] tmpstring;
        }
        else if (!strcmp(tmptype, "FloatVector"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamFloatVector(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "IntVector"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamIntVector(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "MMPanel"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamMMPanel(tmpname, no, tmpcharlist);
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Text"))
        {
            no = atoi(strsep(&p, "\n"));
            len = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no - 1];
            for (j = 0; j < no - 1; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamText(tmpname, tmpcharlist, no - 1, len);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Timer"))
        {
            no = atoi(strsep(&p, "\n"));
            if (no != 3)
            {
                //print_comment(__LINE__, __FILE__, "wrong number of timer-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamTimer(tmpname, no, tmpcharlist);
            for (j = 0; j < no; j++)
                delete[] tmpcharlist[j];
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Passwd"))
        {
            no = atoi(strsep(&p, "\n"));
            if (no != 3)
            {
                //print_comment(__LINE__, __FILE__, "wrong number of timer-parameters");
            }
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamPasswd(tmpname, no, tmpcharlist);
            delete[] tmpcharlist;
        }
        else if (!strcmp(tmptype, "Arrayset"))
        {
            no = atoi(strsep(&p, "\n"));
            tmpcharlist = new char *[no];
            for (j = 0; j < no; j++)
            {
                tmpchptr = strsep(&p, "\n");
                tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
                strcpy(tmpcharlist[j], tmpchptr);
            }
            params_in[i] = (Param *)new ParamArrayset(tmpname, no, tmpcharlist);
            delete[] tmpcharlist;
        }
        else
        {
            char buf[1024];
            sprintf(buf, "unkown type: %s", tmptype);
            //print_comment(__LINE__, __FILE__, buf);
        }
        // cerr << "got " << params_in[i]->name << endl;
    }
}

char *CtlMessage::get_object_name(const char *port)
{
    int i;

    for (i = 0; i < no_of_objects && strcmp(port_names[i], port); i++)
        ;
    if (i == no_of_objects)
        return 0L;
    else
        return object_names[i];
}

char *CtlMessage::get_object_type(const char *port)
{
    int i;

    for (i = 0; i < no_of_objects && strcmp(port_names[i], port); i++)
        ;
    if (i == no_of_objects)
        return 0L;
    else
        return object_types[i];
}

// check whether port is connected
int CtlMessage::is_port_connected(const char *port_name)
{
    int i = 0;

    // find port with this name
    while (i < no_of_objects && strcmp(port_names[i], port_name))
        i++;
    if (i == no_of_objects)
        return 0;
    else
        return port_connected[i];
}

//// ---------- GET PARAMETERS ----------

int CtlMessage::get_scalar_param(const char *param_name, long *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *value = atol(((ParamIntScalar *)params_in[i])->list);
        return ((ParamIntScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_scalar_param(const char *param_name, float *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *value = (float)atof(((ParamFloatScalar *)params_in[i])->list);
        return ((ParamFloatScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_vector_param(const char *param_name, int pos, long *list)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *list = atol(((ParamIntVector *)params_in[i])->list[pos]);
        return ((ParamIntScalar *)params_in[i])->no;
    }
}

int CtlMessage::get_vector_param(const char *param_name, int pos, float *list)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *list = (float)atof(((ParamFloatVector *)params_in[i])->list[pos]);
        return ((ParamFloatVector *)params_in[i])->no;
    }
}

int CtlMessage::get_string_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamString *)params_in[i])->list;
        return ((ParamString *)params_in[i])->no;
    }
}

int CtlMessage::get_text_param(const char *param_name, char ***text, int *line_num)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *text = ((ParamText *)params_in[i])->list;
        *line_num = ((ParamText *)params_in[i])->get_line_number();
        return ((ParamText *)params_in[i])->no;
    }
}

int CtlMessage::get_boolean_param(const char *param_name, int *value)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        if (strcmp(((ParamBoolean *)params_in[i])->list, "TRUE"))
            *value = 0;
        else
            *value = 1;

        return ((ParamBoolean *)params_in[i])->no;
    }
}

int CtlMessage::get_slider_param(const char *param_name, float *min, float *max, float *val)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = *val = (float)0.0;
        return 0;
    }
    else
    {
        *min = (float)atof(((ParamFloatSlider *)params_in[i])->list[0]);
        *max = (float)atof(((ParamFloatSlider *)params_in[i])->list[1]);
        *val = (float)atof(((ParamFloatSlider *)params_in[i])->list[2]);
        return ((ParamFloatSlider *)params_in[i])->no;
    }
}

int CtlMessage::get_slider_param(const char *param_name, long *min, long *max, long *val)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = *val = 0;
        return 0;
    }
    else
    {
        *min = atol(((ParamIntSlider *)params_in[i])->list[0]);
        *max = atol(((ParamIntSlider *)params_in[i])->list[1]);
        *val = atol(((ParamIntSlider *)params_in[i])->list[2]);
        return ((ParamIntSlider *)params_in[i])->no;
    }
}

int CtlMessage::get_choice_param(const char *param_name, int *sel)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *sel = ((ParamChoice *)params_in[i])->sel;
        return ((ParamChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_choice_param(const char *param_name, char **string)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *string = ((ParamChoice *)params_in[i])->list[(((ParamChoice *)params_in[i])->sel) - 1];
        return ((ParamChoice *)params_in[i])->no;
    }
}

int CtlMessage::get_browser_param(const char *param_name, char **file)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *file = ((ParamBrowser *)params_in[i])->list[0];
        return ((ParamBrowser *)params_in[i])->no;
    }
}

int CtlMessage::get_mmpanel_param(const char *param_name, char **text1, char **text2,
                                  long *min, long *max, long *val,
                                  int *b1, int *b2, int *b3, int *b4,
                                  int *b5, int *b6, int *b7)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *text1 = ((ParamMMPanel *)params_in[i])->list[0];
        *text2 = ((ParamMMPanel *)params_in[i])->list[1];
        *min = atol(((ParamMMPanel *)params_in[i])->list[2]);
        *max = atol(((ParamMMPanel *)params_in[i])->list[3]);
        *val = atol(((ParamMMPanel *)params_in[i])->list[4]);
        if (strcmp(((ParamMMPanel *)params_in[i])->list[5], "TRUE"))
            *b1 = 0;
        else
            *b1 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[6], "TRUE"))
            *b2 = 0;
        else
            *b2 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[7], "TRUE"))
            *b3 = 0;
        else
            *b3 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[8], "TRUE"))
            *b4 = 0;
        else
            *b4 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[9], "TRUE"))
            *b5 = 0;
        else
            *b5 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[10], "TRUE"))
            *b6 = 0;
        else
            *b6 = 1;
        if (strcmp(((ParamMMPanel *)params_in[i])->list[11], "TRUE"))
            *b7 = 0;
        else
            *b7 = 1;

        return ((ParamMMPanel *)params_in[i])->no;
    }
}

int CtlMessage::get_timer_param(const char *param_name, long *start, long *delta, long *state)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *start = *delta = *state = 0;
        return 0;
    }
    else
    {
        *start = atol(((ParamTimer *)params_in[i])->list[0]);
        *delta = atol(((ParamTimer *)params_in[i])->list[1]);
        *state = atol(((ParamTimer *)params_in[i])->list[2]);
        return ((ParamTimer *)params_in[i])->no;
    }
}

int CtlMessage::get_passwd_param(const char *param_name, char **host, char **user,
                                 char **passwd)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *host = ((ParamPasswd *)params_in[i])->list[0];
        *user = ((ParamPasswd *)params_in[i])->list[1];
        *passwd = ((ParamPasswd *)params_in[i])->list[2];
        return ((ParamPasswd *)params_in[i])->no;
    }
}

int CtlMessage::get_cli_param(const char *param_name, char **command)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
        return 0;
    else
    {
        *command = ((ParamCli *)params_in[i])->list;
        return ((ParamCli *)params_in[i])->no;
    }
}

int CtlMessage::get_arrayset_param(const char *param_name, char **buf)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL)
           || strcmp(params_in[i]->name, param_name))
        i++;

    if (i == no_of_params_in)
    {
        return 0;
    }
    else
    {
        // return pointer to the arrayset buffer
        *buf = ((ParamArrayset *)params_in[i])->list[0];
        return ((ParamArrayset *)params_in[i])->no;
    }
}

int CtlMessage::get_colormap_param(const char *param_name, float *min, float *max, int *len, colormap_type *type)
{
    int i = 0;
    while (i < no_of_params_in
           && (params_in[i]->name == NULL
               || strcmp(params_in[i]->name, param_name)))
        i++;

    if (i == no_of_params_in)
    {
        *min = *max = (float)0.0;
        *len = 0;
        *type = RGBAX;
        return 0;
    }
    else
    {
        *min = (float)atof(((ParamColormap *)params_in[i])->list[0]);
        *max = (float)atof(((ParamColormap *)params_in[i])->list[1]);
        if (!strcmp(((ParamColormap *)params_in[i])->list[2], "RGBAX"))
        {
            *len = atoi(((ParamColormap *)params_in[i])->list[3]);
            *type = RGBAX;
        }
        else if (!strcmp(((ParamColormap *)params_in[i])->list[2], "VIRVO"))
        {
            *type = VIRVO;
        }
        else
        {
            cerr << "invalid colormap type " << ((ParamColormap *)params_in[i])->list[3] << endl;
            *type = RGBAX;
        }
        return ((ParamColormap *)params_in[i])->no;
    }
}

int CtlMessage::set_browser_param(const char *pname, char *file, char *wildcard)
{
    int i = 0;
    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;

    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamBrowser(pname, file, wildcard);
    no_of_params_out++;
    return ((ParamBrowser *)params_out[i])->no;
}

int CtlMessage::set_vector_param(const char *pname, int len, long *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntVector(pname, len, val);
    no_of_params_out++;
    return ((ParamIntVector *)params_out[i])->no;
}

int CtlMessage::set_vector_param(const char *pname, int len, float *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatVector(pname, len, val);
    no_of_params_out++;
    return ((ParamFloatVector *)params_out[i])->no;
}

int CtlMessage::set_boolean_param(const char *pname, int val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamBoolean(pname, val);
    no_of_params_out++;
    return ((ParamBoolean *)params_out[i])->no;
}

int CtlMessage::set_scalar_param(const char *pname, long val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntScalar(pname, val);
    no_of_params_out++;
    return ((ParamIntScalar *)params_out[i])->no;
}

int CtlMessage::set_scalar_param(const char *pname, float val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatScalar(pname, val);
    no_of_params_out++;
    return ((ParamFloatScalar *)params_out[i])->no;
}

int CtlMessage::set_choice_param(const char *pname, int len, char **list, int pos)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamChoice(pname, len, list, pos);
    no_of_params_out++;
    return ((ParamChoice *)params_out[i])->no;
}

int CtlMessage::set_string_param(const char *pname, char *val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamString(pname, val);
    no_of_params_out++;
    return ((ParamString *)params_out[i])->no;
}

int CtlMessage::set_text_param(const char *pname, char *text, int linenum)
{
    int i, j;
    char *tmpchptr;
    char **tmpcharlist;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    tmpcharlist = new char *[linenum];

    char *p = text;
    for (j = 0; j < linenum - 1; j++)
    {
        tmpchptr = strsep(&p, "\n");
        tmpcharlist[j] = new char[strlen(tmpchptr) + 1];
        strcpy(tmpcharlist[j], tmpchptr);
    }
    params_out[i] = (Param *)new ParamText(pname, tmpcharlist, linenum, (int)strlen(text));
    delete[] tmpcharlist;
    no_of_params_out++;
    return ((ParamText *)params_out[i])->no;
}

int CtlMessage::set_slider_param(const char *pname, float min, float max, float val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamFloatSlider(pname, min, max, val);
    no_of_params_out++;
    return ((ParamFloatSlider *)params_out[i])->no;
}

int CtlMessage::set_slider_param(const char *pname, long min, long max, long val)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamIntSlider(pname, min, max, val);
    no_of_params_out++;
    return ((ParamIntSlider *)params_out[i])->no;
}

int CtlMessage::set_mmpanel_param(const char *pname, char *text1, char *text2,
                                  long min, long max, long val, int b1, int b2,
                                  int b3, int b4, int b5, int b6, int b7)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    char buf1[7], buf2[7], buf3[7], buf4[7], buf5[7], buf6[7], buf7[7];
    if (b1 == 0)
        strcpy(buf1, "FALSE");
    else
        strcpy(buf1, "TRUE");
    if (b2 == 0)
        strcpy(buf2, "FALSE");
    else
        strcpy(buf2, "TRUE");
    if (b3 == 0)
        strcpy(buf3, "FALSE");
    else
        strcpy(buf3, "TRUE");
    if (b4 == 0)
        strcpy(buf4, "FALSE");
    else
        strcpy(buf4, "TRUE");
    if (b5 == 0)
        strcpy(buf5, "FALSE");
    else
        strcpy(buf5, "TRUE");
    if (b6 == 0)
        strcpy(buf6, "FALSE");
    else
        strcpy(buf6, "TRUE");
    if (b7 == 0)
        strcpy(buf7, "FALSE");
    else
        strcpy(buf7, "TRUE");
    params_out[i] = (Param *)new ParamMMPanel(pname, text1, text2, min, max, val,
                                              buf1, buf2, buf3, buf4, buf5, buf6, buf7);
    no_of_params_out++;
    return ((ParamMMPanel *)params_out[i])->no;
}

int CtlMessage::set_timer_param(const char *pname, long start, long delta, long state)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamTimer(pname, start, delta, state);
    no_of_params_out++;
    return ((ParamTimer *)params_out[i])->no;
}

int CtlMessage::set_passwd_param(const char *pname, char *host, char *user, char *passwd)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamPasswd(pname, host, user, passwd);
    no_of_params_out++;
    return ((ParamPasswd *)params_out[i])->no;
}

int CtlMessage::set_cli_param(const char *pname, char *result)
{
    int i;

    for (i = 0; i < no_of_params_out && strcmp(params_out[i]->name, pname); i++)
        ;
    if (i != no_of_params_out || i > MAX_OUT_PAR)
        return 0; // already set

    params_out[i] = (Param *)new ParamCli(pname, result);
    no_of_params_out++;
    return ((ParamCli *)params_out[i])->no;
}

int CtlMessage::set_save_object(const char *sname)
{
    char **tmpptr;
    int i;

    if (no_of_save_objects % 10 == 0)
    {
        tmpptr = new char *[no_of_save_objects + 10];
        for (i = 0; i < no_of_save_objects; i++)
            tmpptr[i] = save_names[i];
        save_names = tmpptr;
        // delete [] save_names;
    }

    save_names[no_of_save_objects] = new char[strlen(sname) + 1];
    strcpy(save_names[no_of_save_objects], sname);
    no_of_save_objects++;
    return 1;
}

int CtlMessage::set_release_object(const char *sname)
{
    char **tmpptr;
    int i;

    if (no_of_release_objects % 10 == 0)
    {
        tmpptr = new char *[no_of_release_objects + 10];
        for (i = 0; i < no_of_release_objects; i++)
            tmpptr[i] = release_names[i];
        release_names = tmpptr;
        // delete [] release_names;
    }

    release_names[no_of_release_objects] = new char[strlen(sname) + 1];
    strcpy(release_names[no_of_release_objects], sname);
    no_of_release_objects++;
    return 1;
}

int CtlMessage::create_finall_message()
{
    int size, i, j;
    char numbuf[15];

    size = 1; // final '\0'
    size += (int)strlen(m_name) + 1;
    size += (int)strlen(inst_no) + 1;
    size += (int)strlen(h_name) + 1;
    size += 5; // no_of_params_out
    size += 5; // no_of_save_objects
    size += 5; // no_of_release_objects

    for (i = 0; i < no_of_params_out; i++)
    {
        switch (params_out[i]->type)
        {

        case STRING:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("String") + 1 + 5 + 1 + strlen(((ParamString *)params_out[i])->list) + 1);
            break;

        case TEXT:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Text") + 1 + 5 + 1 + 10 + 1);
            for (j = 0; j < (params_out[i]->no) - 1; j++)
                size += (int)strlen(((ParamText *)params_out[i])->list[j]) + 1;
            break;

        case COVISE_BOOLEAN:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Boolean") + 1 + 5 + 1 + strlen(((ParamBoolean *)params_out[i])->list) + 1);
            break;

        case BROWSER:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Browser") + 1 + 5 + 1 + strlen(((ParamBrowser *)params_out[i])->list[0]) + 1 + strlen(((ParamBrowser *)params_out[i])->list[1]) + 1);
            break;

        case FLOAT_SLIDER:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("FloatSlider") + 1 + 5 + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[0]) + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[1]) + 1 + strlen(((ParamFloatSlider *)params_out[i])->list[2]) + 1);
            break;

        case INT_SLIDER:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("IntSlider") + 1 + 5 + 1 + strlen(((ParamIntSlider *)params_out[i])->list[0]) + 1 + strlen(((ParamIntSlider *)params_out[i])->list[1]) + 1 + strlen(((ParamIntSlider *)params_out[i])->list[2]) + 1);
            break;

        case COLORMAP_MSG:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Colormap") + 1 + 5 + 1);
            for (j = 0; j < params_out[i]->no; j++)
                size += (int)(strlen(((ParamColormap *)params_out[i])->list[j]) + 1);
            break;

        case CHOICE:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Choice") + 1 + 5 + 1 + 5 + 1);
            for (j = 0; j < (params_out[i]->no); j++)
                size += (int)(strlen(((ParamChoice *)params_out[i])->list[j]) + 1);
            break;

        case FLOAT_VECTOR:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("FloatVector") + 1 + 5 + 1);
            for (j = 0; j < params_out[i]->no; j++)
                size += (int)(strlen(((ParamFloatVector *)params_out[i])->list[j]) + 1);
            break;

        case INT_VECTOR:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("IntVector") + 1 + 5 + 1);
            for (j = 0; j < params_out[i]->no; j++)
                size += (int)(strlen(((ParamIntVector *)params_out[i])->list[j]) + 1);
            break;

        case FLOAT_SCALAR:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("FloatScalar") + 1 + 5 + 1 + strlen(((ParamFloatScalar *)params_out[i])->list) + 1);
            break;

        case INT_SCALAR:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("IntScalar") + 1 + 5 + 1 + strlen(((ParamIntScalar *)params_out[i])->list) + 1);
            break;

        case MMPANEL:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("MMPANEL") + 1 + 5 + 1 + strlen(((ParamMMPanel *)params_out[i])->list[0]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[1]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[2]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[3]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[4]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[5]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[6]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[7]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[8]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[9]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[10]) + 1 + strlen(((ParamMMPanel *)params_out[i])->list[11]) + 1);
            break;

        case TIMER:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Timer") + 1 + 5 + 1 + strlen(((ParamTimer *)params_out[i])->list[0]) + 1 + strlen(((ParamTimer *)params_out[i])->list[1]) + 1 + strlen(((ParamTimer *)params_out[i])->list[2]) + 1);
            break;

        case PASSWD:
            size += (int)(strlen(params_out[i]->name) + 1 + strlen("Passwd") + 1 + 5 + 1 + strlen(((ParamString *)params_out[0])->list) + 1 + strlen(((ParamString *)params_out[1])->list) + 1 + strlen(((ParamString *)params_out[2])->list) + 1);
            break;
        case CLI:
            size += (int)strlen(params_out[i]->name) + 1 + (int)strlen("Cli") + 1 + 5 + 1 + (int)strlen(((ParamCli *)params_out[i])->list) + 1;
            break;

        case ARRAYSET:
            size += (int)strlen(params_out[i]->name) + 1 + (int)strlen("Arrayset") + 1 + 5 + 1;
            for (j = 0; j < params_out[i]->no; j++)
                size += (int)strlen(((ParamArrayset *)params_out[i])->list[j]) + 1;
            break;
        }
    }
    for (i = 0; i < no_of_save_objects; i++)
        size += (int)strlen(save_names[i]) + 1;
    for (i = 0; i < no_of_release_objects; i++)
        size += (int)strlen(release_names[i]) + 1;

    if (data)
    {
        delete[] data;
    }
    data = new char[size];

    strcpy(data, m_name);
    strcat(data, "\n");

    strcat(data, inst_no);
    strcat(data, "\n");

    strcat(data, h_name);
    strcat(data, "\n");

    sprintf(numbuf, "%d\n", no_of_params_out);
    strcat(data, numbuf);
    sprintf(numbuf, "%d\n", no_of_save_objects);
    strcat(data, numbuf);
    sprintf(numbuf, "%d\n", no_of_release_objects);
    strcat(data, numbuf);

    for (i = 0; i < no_of_params_out; i++)
    {
        strcat(data, params_out[i]->name);
        strcat(data, "\n");

        switch (params_out[i]->type)
        {

        case STRING:
            strcat(data, "String");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamString *)params_out[i])->list);
            strcat(data, "\n");
            break;

        case TEXT:
            strcpy(data, "Text");
            strcat(data, "\n");
            // no of tokens
            sprintf(numbuf, "%d\n", (params_out[i]->no) + 1);
            strcat(data, numbuf);
            // string length
            sprintf(numbuf, "%d\n", ((ParamText *)params_out[i])->length);
            strcat(data, numbuf);
            for (j = 0; j < ((ParamText *)params_out[i])->line_num; j++)
            {
                strcat(data, ((ParamText *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case COVISE_BOOLEAN:
            strcat(data, "Boolean");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamBoolean *)params_out[i])->list);
            strcat(data, "\n");
            break;

        case BROWSER:
            strcat(data, "Browser");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamBrowser *)params_out[i])->list[0]);
            strcat(data, "\n");
            strcat(data, ((ParamBrowser *)params_out[i])->list[1]);
            strcat(data, "\n");
            break;

        case FLOAT_SLIDER:
            strcat(data, "FloatSlider");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamFloatSlider *)params_out[i])->list[0]);
            strcat(data, "\n");
            strcat(data, ((ParamFloatSlider *)params_out[i])->list[1]);
            strcat(data, "\n");
            strcat(data, ((ParamFloatSlider *)params_out[i])->list[2]);
            strcat(data, "\n");
            break;

        case INT_SLIDER:
            strcat(data, "IntSlider");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamIntSlider *)params_out[i])->list[0]);
            strcat(data, "\n");
            strcat(data, ((ParamIntSlider *)params_out[i])->list[1]);
            strcat(data, "\n");
            strcat(data, ((ParamIntSlider *)params_out[i])->list[2]);
            strcat(data, "\n");
            break;

        case COLORMAP_MSG:
            strcat(data, "Colormap");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamColormap *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case CHOICE:
            strcat(data, "Choice");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", (params_out[i]->no) + 1);
            strcat(data, numbuf);
            sprintf(numbuf, "%d\n", ((ParamChoice *)params_out[i])->sel);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamChoice *)params_out[i])->list[j]);
                strcat(data, "\n");
            }

            break;

        case FLOAT_VECTOR:
            strcat(data, "FloatVector");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamFloatVector *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case INT_VECTOR:
            strcat(data, "IntVector");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamIntVector *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case FLOAT_SCALAR:
            strcat(data, "FloatScalar");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamFloatScalar *)params_out[i])->list);
            strcat(data, "\n");
            break;

        case INT_SCALAR:
            strcat(data, "IntScalar");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamIntScalar *)params_out[i])->list);
            strcat(data, "\n");
            break;

        case MMPANEL:
            strcat(data, "MMPanel");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamMMPanel *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case TIMER:
            strcat(data, "Timer");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamTimer *)params_out[i])->list[0]);
            strcat(data, "\n");
            strcat(data, ((ParamTimer *)params_out[i])->list[1]);
            strcat(data, "\n");
            strcat(data, ((ParamTimer *)params_out[i])->list[2]);
            strcat(data, "\n");
            break;

        case PASSWD:
            strcat(data, "Passwd");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamPasswd *)params_out[i])->list[0]);
            strcat(data, "\n");
            strcat(data, ((ParamPasswd *)params_out[i])->list[1]);
            strcat(data, "\n");
            strcat(data, ((ParamPasswd *)params_out[i])->list[2]);
            strcat(data, "\n");
            break;

        case ARRAYSET:
            strcat(data, "Arrayset");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            for (j = 0; j < params_out[i]->no; j++)
            {
                strcat(data, ((ParamArrayset *)params_out[i])->list[j]);
                strcat(data, "\n");
            }
            break;

        case CLI:
            strcat(data, "Cli");
            strcat(data, "\n");
            sprintf(numbuf, "%d\n", params_out[i]->no);
            strcat(data, numbuf);
            strcat(data, ((ParamCli *)params_out[i])->list);
            strcat(data, "\n");
            break;
        }
    }
    for (i = 0; i < no_of_save_objects; i++)
    {
        strcat(data, save_names[i]);
        strcat(data, "\n");
    }
    for (i = 0; i < no_of_release_objects; i++)
    {
        strcat(data, release_names[i]);
        strcat(data, "\n");
    }

    type = FINALL;
    length = (int)strlen(data) + 1;

    return 1;
}

int CtlMessage::create_finpart_message()
{
    int size;

    size = 1; // final '\0'
    size += (int)strlen(m_name) + 1;
    size += (int)strlen(inst_no) + 1;
    size += (int)strlen(h_name) + 1;

    if (data)
    {
        delete[] data;
    }
    data = new char[size];

    strcpy(data, m_name);
    strcat(data, "\n");

    strcat(data, inst_no);
    strcat(data, "\n");

    strcat(data, h_name);
    strcat(data, "\n");

    type = FINPART;
    length = (int)strlen(data) + 1;

    return 1;
}

CtlMessage::~CtlMessage()
{
    int i;

    delete[] m_name;
    delete[] h_name;

    for (i = 0; i < no_of_objects; i++)
    {
        delete[] port_names[i];
        delete[] object_names[i];
        delete[] object_types[i];
    }
    delete[] port_names;
    delete[] object_names;
    delete[] object_types;

    for (i = 0; i < no_of_save_objects; i++)
    {
        delete[] save_names[i];
    }
    delete[] save_names;

    for (i = 0; i < no_of_release_objects; i++)
    {
        delete[] release_names[i];
    }
    delete[] release_names;

    for (i = 0; i < no_of_params_in; i++)
    {
        delete params_in[i];
    }
    delete[] params_in;

    for (i = 0; i < no_of_params_out; i++)
    {
        delete params_out[i];
    }
    delete[] params_out;

    delete[] inst_no;
}

void RenderMessage::init_list()
{
    char *tmpchptr;
    int i;
    //char tmp_str[255];

    m_name = get_part(data);
    inst_no = get_part();
    h_name = get_part();
    no_of_objects = atoi(get_part());

    object_names = new char *[no_of_objects];

    for (i = 0; i < no_of_objects; i++)
    {
        tmpchptr = get_part();
        object_names[i] = new char[strlen(tmpchptr) + 1];
        strcpy(object_names[i], tmpchptr);
        //sprintf(tmp_str, "getting %s", object_names[i]);
        ////print_comment(__LINE__, __FILE__, tmp_str);
    }
}

RenderMessage::~RenderMessage()
{
    int i;

    for (i = 0; i < no_of_objects; i++)
    {
        delete[] object_names[i];
    }
    delete[] object_names;
}
*/
