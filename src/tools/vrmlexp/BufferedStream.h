#ifndef BASE_WRITER_H
#define BASE_WRITER_H
//#include <varargs.h>
#include <stdarg.h>
#include <maxstring.h>
#include <stdio.h>
#define LINELENGTH 10000
/**
* This implements a buffered high performance version of the max stream
*/
class BufferedStream
{
public:
	BufferedStream();
	virtual ~BufferedStream();
	/**
	* Open a file
	*
	* \param fileName File name to open
	* \return			true if successful, false otherwize
	*/
	bool Open(const MCHAR* fileName, bool unused1, int unused2);

	/**
	* Open a file
	*
	* \param fileName File name to open
	* \return			true if successful, false otherwize
	*/
	//bool Open(const MaxSDK::Util::MaxString& fileName);


	/**
	* Close the file
	**/
	void Close();

	/**
	* Write a string
	* \param string	String to write
	* \param nchars	Number of character to write. Default is size of string in characters
	* \return			Number of characters written
	*/
	//virtual size_t Write(const char* string, size_t nchars = (size_t)-1);

	/**
	* Write an UTF-16 string
	* \param string	String to write
	* \param nchars	Number of character to write. Default is size of string in characters
	* \return			Number of characters written
	*/
	//virtual size_t Write(const wchar_t*string, size_t nchars = (size_t)-1);

	/**
	* Write an MaxString
	* \param String	String to write
	* \return			Number of characters written
	*/
	//virtual size_t Write(const MaxSDK::Util::MaxString&String);



	/**
	* Returns true if file is open
	*/
	virtual bool IsFileOpen() const;

	/**
	* Make sure that all the buffer are synced with the OS native objects.
	*/
	virtual void Flush();

	inline void appendToBuffer(const char *buf, size_t numBytes) { prepareBuffer(numBytes); memcpy(buffer + writePosition, lineBuffer, numBytes), writePosition += numBytes; };
	inline void prepareBuffer(size_t numBytes) { if ((writePosition + numBytes) > bufferSize) Flush(); };


#ifdef UNICODE
	size_t Printf(const wchar_t*, ...);
	//size_t Vprintf(const wchar_t*, va_list);
#endif
	size_t Printf(const char*, ...);
	//size_t Vprintf(const char*, va_list);
private:

	FILE *fp;
	char *buffer;
	size_t bufferSize;
	size_t writePosition;
	char lineBuffer[LINELENGTH];
	wchar_t wLineBuffer[LINELENGTH];

};
#endif

