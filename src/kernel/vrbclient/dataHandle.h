/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H
#include <memory>
#include "SharedStateSerializer.h"
#include <util/coExport.h>


namespace vrb
{
	//controlls the lifetime of data
class VRBEXPORT DataHandle
{
public:
    DataHandle();
    DataHandle(char* data, const int length);

	const char* data() const;
    int length() const;

private:

	std::shared_ptr<char> m_data;
	int m_length = 0;
};

template<>
void VRBEXPORT serialize<DataHandle>(covise::TokenBuffer& tb, const DataHandle& value);


template<>
void VRBEXPORT deserialize<DataHandle>(covise::TokenBuffer& tb, DataHandle& value);


}
#endif // !DATA_HANDLE_H

