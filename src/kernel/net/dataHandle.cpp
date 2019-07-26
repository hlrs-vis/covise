/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "dataHandle.h"
#include <cstddef>
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <cassert>

using namespace std;
namespace covise
{
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
    } else
    {
        m_dataPtr = data;
    }
}

DataHandle::DataHandle(size_t size)
{
    m_ManagedData.reset(new char[size]);
    m_dataPtr = m_ManagedData.get();
    m_length = static_cast<int>(size);
}


const char* DataHandle::data() const
{
    if (m_ManagedData.get() && m_dataPtr)
    {
        checkPtr();
        return m_dataPtr;
    } else if (m_dataPtr && !m_ManagedData.get())
    {
        return m_dataPtr;
    } else if (!m_dataPtr && m_ManagedData.get())
    {
        return m_ManagedData.get();
    } else
    {
        return nullptr;
    }
}
char* DataHandle::accessData()
{
    return const_cast<char *>(data());
}

const int DataHandle::length() const
{
	return m_length;
}


const char *DataHandle::end() const
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

}
