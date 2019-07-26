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
	:m_data(nullptr)
	, m_length(0)
{
}

DataHandle::DataHandle() = default;

DataHandle::DataHandle(char* data, const int length)
	: m_length(length)
{
	m_data.reset(data, [](char* c) {delete[]c; });
}


const char* DataHandle::data() const
{
	return m_data.get();
}

const int DataHandle::length() const
{
	return m_length;
}

}
