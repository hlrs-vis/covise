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
namespace vrb
{
vrb::DataHandle::DataHandle()
	:m_data(nullptr)
	, m_length(0)
	,m_ownsData(false)
{
}

DataHandle::DataHandle(char* data, int length)
	: m_data(data)
	, m_length(length)
	, m_ownsData(true)
{

}
vrb::DataHandle::DataHandle(covise::TokenBuffer& tb)
	:m_length(tb.get_length())
	,m_data(nullptr)
	,m_ownsData(false)
{
	try
	{
		m_data = tb.take_data();
		m_ownsData = true;
	}
	catch (const std::exception&)
	{
		m_data = tb.get_data();
		m_ownsData = false;
	}
}
vrb::DataHandle::DataHandle(covise::MessageBase& m)
	: m_data(nullptr)
	, m_length(m.length)
{
	try
	{
		m_data = m.takeData();
		m_ownsData = true;
	}
	catch (const std::exception&)
	{
		m_data = m.data;
		m_ownsData = false;
	}
}
DataHandle::~DataHandle()
{
	deleteMe();
}

const char* vrb::DataHandle::data() const
{
	return m_data;
}

const int vrb::DataHandle::length() const
{
	return m_length;
}

DataHandle& vrb::DataHandle::operator=(const DataHandle& other)
{
	m_data = other.m_data;
	m_length = other.m_length;
	m_ownsData = other.m_ownsData;
	other.m_others.insert(this);
	m_others.insert(&other);
	return *this;
}

void DataHandle::copyTokenBuffer(const covise::TokenBuffer& tb)
{
	deleteMe();
	m_length = tb.get_length();
	char* n = new char [m_length];
	memcpy(n, tb.get_data(), m_length);
	m_data = n;
}


void vrb::DataHandle::deleteMe(void)
{
	if (!m_ownsData)
	{
		m_data = nullptr;
		m_length = 0;
		return;
	}
	for (auto other : m_others)
	{
		other->m_others.erase(this);
	}
	if (m_others.size() == 0)
	{
		delete[] m_data;
		m_data = nullptr;
	}
}

template<>
void serialize<DataHandle>(covise::TokenBuffer& tb, const DataHandle& value)
{
	covise::TokenBuffer n(value.data(), value.length());
	tb << n;
}
//!copies the data 
template<>
void deserialize<DataHandle>(covise::TokenBuffer& tb, DataHandle& value)
{
	covise::TokenBuffer n;
	tb >> n;
	value = DataHandle(n);

}
}