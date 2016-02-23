/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_UTILITIES_H
#define OSC_UTILITIES_H

#include <string>

#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXParseException.hpp>


namespace OpenScenario
{

//
struct fileNamePath
{
    fileNamePath() { };
    ~fileNamePath() { };

    std::string fileName;
    std::string path;
};


//
class ParserErrorHandler: public xercesc::ErrorHandler
{
public:
    void warning(const xercesc::SAXParseException &spExept);
    void error(const xercesc::SAXParseException &spExept);
    void fatalError(const xercesc::SAXParseException &spExept);
    void resetErrors();

private:
    void reportParseException(const xercesc::SAXParseException &spExept);
};


//
std::string generateRandomString(const size_t numOfChars);

}

#endif /* OSC_UTILITIES_H */
