#include "BufferedStream.h"


BufferedStream::BufferedStream()
{

	fp=NULL;
	buffer=NULL;
    bufferSize=1024*1024*50;
	writePosition=0;
}
BufferedStream::~BufferedStream()
{
	delete[] buffer;
}

bool BufferedStream::Open(const MCHAR* fileName, bool append, int unused2)
{

	errno_t err=0;
	if(append)
	    err = _tfopen_s(&fp, fileName, _T("a"));
	else
		err = _tfopen_s(&fp, fileName, _T("w"));
	if (err == 0)
	{
		buffer = new char[bufferSize];
		writePosition = 0;
		return true;
	}
	else
	{
		fp = NULL;
		buffer = NULL;
		return false;
	}
	return false;
}

/*bool BufferedStream::Open(const MaxSDK::Util::MaxString& fileName)
{
}*/

/**
* Close the file
**/
void BufferedStream::Close()
{
	Flush();
	delete[] buffer;
	buffer = NULL;
	fclose(fp);
}

/*size_t BufferedStream::Write(const char* string, size_t nchars = (size_t)-1)
{
}


size_t BufferedStream::Write(const wchar_t*string, size_t nchars = (size_t)-1)
{
}
size_t BufferedStream::Write(const MaxSDK::Util::MaxString&String)
{
}
*/
bool BufferedStream::IsFileOpen() const
{
	if (fp != NULL)
		return true;
	else
		return false;
}

/**
* Make sure that all the buffer are synced with the OS native objects.
*/
void BufferedStream::Flush()
{
	size_t written = fwrite(buffer, 1, writePosition, fp);
	if (written < writePosition)
	{
		fprintf(stderr, "short write\n");
	}
	writePosition = 0;
}
#ifdef UNICODE
size_t BufferedStream::Printf(const wchar_t* format, ...)
{
	va_list arglist;
	wLineBuffer[0] = '\0';
	va_start(arglist, format);
	int ret = _vsnwprintf(wLineBuffer, LINELENGTH, format, arglist);
	va_end(arglist);
	if (ret > 0)
	{
		int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wLineBuffer[0], ret, NULL, 0, NULL, NULL);
		WideCharToMultiByte(CP_UTF8, 0, &wLineBuffer[0], ret, &lineBuffer[0], size_needed, NULL, NULL);
		size_t len = strlen(lineBuffer);
		appendToBuffer(lineBuffer, size_needed);
		return size_needed;
    }
	return ret;
}
//size_t BufferedStream::Vprintf(const wchar_t*, va_list)
//{
//}
#endif
size_t BufferedStream::Printf(const char* format , ...)
{
	va_list arglist;
	lineBuffer[0] = '\0';
	va_start(arglist, format);
	int ret = vsnprintf(lineBuffer,LINELENGTH,format, arglist);
	va_end(arglist);
	size_t len = strlen(lineBuffer);
	appendToBuffer(lineBuffer, len);
	return len;
}
//size_t BufferedStream::Vprintf(const char*, va_list)
//{
//}


