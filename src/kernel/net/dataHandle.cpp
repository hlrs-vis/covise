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

DataHandle::DataHandle() = default;

DataHandle::DataHandle(char* data, const int length)
	: m_length(length)
{
	m_data.reset(data, [](char* c) {delete[]c; });
}


const char* vrb::DataHandle::data() const
{
	return m_data.get();
}

int vrb::DataHandle::length() const
{
	return m_length;
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
    auto l = n.get_length();
    value = DataHandle(n.take_data(), l);

}
}
