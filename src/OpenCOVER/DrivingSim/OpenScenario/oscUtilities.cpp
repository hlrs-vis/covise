/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscUtilities.h"

#include <iostream>
#include <cstdlib>

#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/util/TransService.hpp>


using namespace OpenScenario;


//////
// class parserErrorHandler
//
void ParserErrorHandler::warning(const xercesc::SAXParseException &spExept)
{
    std::cerr << "Warning:" << std::endl;
    reportParseException(spExept);
}

void ParserErrorHandler::error(const xercesc::SAXParseException &spExept)
{
    std::cerr << "Error:" << std::endl;
	reportParseException(spExept);

    //throw any kind of exception so that we have something to catch
    throw 98;
}

void ParserErrorHandler::fatalError(const xercesc::SAXParseException &spExept)
{
    std::cerr << "Fatal Error:" << std::endl;
    reportParseException(spExept);

    //throw any kind of exception so that we have something to catch
    throw 99;
}

void ParserErrorHandler::resetErrors()
{

}

void ParserErrorHandler::reportParseException(const xercesc::SAXParseException &spExept)
{
    char *message = XMLChTranscodeUtf(spExept.getMessage());
	errorMsg = message;

    std::cerr << " at line " << spExept.getLineNumber() << ", column " << spExept.getColumnNumber() <<":\n" << message <<std::endl;

//	xercesc::XMLString::release(&message);
}

void ParserErrorHandler::releaseErrorMessage()
{
	xercesc::XMLString::release(&errorMsg);
}



//////
//

//get environment variable
//
bf::path OpenScenario::getEnvVariable(const std::string &envVar)
{
    //getenv aus <cstdlib>
    const char *val = std::getenv(envVar.c_str());

    return val == NULL ? "" : bf::path(val);
}

//random string
//
std::string OpenScenario::generateRandomString(const size_t numOfChars)
{
    //output string
    std::string str(numOfChars, 0);
    //set of chars to use in string
    const char alphaNum[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
    //length of array
    const size_t maxLength = (sizeof(alphaNum) / sizeof(char)) - 1;

    for (size_t i = 0; i <= numOfChars; i++)
    {
        str[i] = alphaNum[rand() % maxLength];
    }

    return str;
}

char *
OpenScenario::XMLChTranscodeUtf(const XMLCh *in)
{
	return  (char *)xercesc::TranscodeToStr(in, xercesc::XMLString::stringLen(in), "utf-8").adopt(); 
}

XMLCh *
OpenScenario::XMLChTranscodeUtf(const char *s)
{
    return  xercesc::TranscodeFromStr((unsigned char*)s, strlen(s), "utf-8").adopt();
}
