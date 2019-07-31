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
    virtual ~DataHandle();
	explicit DataHandle(char* data, const size_t length, bool doDelete = true);
    explicit DataHandle(char* data, const int length, bool doDelete = true);
    DataHandle(size_t size);
	const char* data() const;

    char* accessData();

	const int length() const;
    //pointer to the last char
    const char* end() const;
    char* end();
    void setLength(const int l);
    void incLength(const int inc);
    void movePtr(int amount);
protected:
    //char* m_dataSection = nullptr;
	std::shared_ptr<char> m_ManagedData;
    char* m_dataPtr = nullptr;
    int m_length = 0;
    //check that m_dataPtr points in the the managed memory
    void checkPtr() const;
};
}
#endif // !DATA_HANDLE_H

