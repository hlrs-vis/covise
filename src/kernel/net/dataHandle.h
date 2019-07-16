/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATA_HANDLE_H
#define DATA_HANDLE_H
#include <memory>
#include <util/coExport.h>


namespace covise
{
	//controlls the lifetime of data
class NETEXPORT DataHandle
{
public:
    DataHandle();
    DataHandle(char* data, const int length);

	const char* data() const;
    int length() const;

private:

	std::shared_ptr<const char> m_data;
	int m_length = 0;
};
}
#endif // !DATA_HANDLE_H

