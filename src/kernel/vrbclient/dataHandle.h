/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H
#include <set>
#include <functional>
#include "SharedStateSerializer.h"
#include <util/coExport.h>
namespace covise
{
	class TokenBuffer;
	class MessageBase;
}

namespace vrb
{
	//controlls the lifetime of data
class VRBEXPORT DataHandle
{
public:
	DataHandle();
	DataHandle(char* data, int length);
	//takes conrol over the TokenBuffer data if tb owns its data
	DataHandle(covise::TokenBuffer &tb);
	DataHandle(covise::MessageBase& m);
	~DataHandle();

	const char* data() const;
	const int length() const;
	DataHandle& operator=(const DataHandle& other);
	//copies the data of tb
	void copyTokenBuffer(const covise::TokenBuffer& tb);
private:
	const char* m_data;
	int m_length;
	bool m_ownsData;
	mutable std::set<const DataHandle*> m_others;

	void deleteMe(void);
};
template<>
void VRBEXPORT serialize<DataHandle>(covise::TokenBuffer& tb, const DataHandle& value);


template<>
void VRBEXPORT deserialize<DataHandle>(covise::TokenBuffer& tb, DataHandle& value);


}
#endif // !DATA_HANDLE_H

