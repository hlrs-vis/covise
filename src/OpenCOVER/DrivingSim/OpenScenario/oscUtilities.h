/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_UTILITIES_H
#define OSC_UTILITIES_H

#include <string>

#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/util/XercesDefs.hpp>
XERCES_CPP_NAMESPACE_BEGIN
class SAXParseException;
XERCES_CPP_NAMESPACE_END

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>


namespace bf = boost::filesystem;


namespace OpenScenario
{

//
class ParserErrorHandler: public xercesc::ErrorHandler
{
public:
    void warning(const xercesc::SAXParseException &spExept);
    void error(const xercesc::SAXParseException &spExept);
    void fatalError(const xercesc::SAXParseException &spExept);
	void releaseErrorMessage();
    void resetErrors();
	char *getErrorMessage()
	{
		return errorMsg;
	}

private:
    void reportParseException(const xercesc::SAXParseException &spExept);

	char *errorMsg;
};


//
bf::path getEnvVariable(const std::string &envVar);
std::string generateRandomString(const size_t numOfChars);
}

#endif /* OSC_UTILITIES_H */
