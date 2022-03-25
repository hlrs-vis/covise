#include "coConfigXercesConverter.h"
#include <iostream>
#include <string>
#include <xercesc/util/XMLString.hpp>
#include <locale>
#include <boost/locale/encoding.hpp>

std::string
covise::xercescToStdString(const XMLCh *str)
{
    char *c = xercesc::XMLString::transcode(str);
    std::string retval = c;
    xercesc::XMLString::release(&c);
    return retval;
}

void Deleter::operator()(XMLCh *d)
{
    xercesc::XMLString::release(&d);
}

std::unique_ptr<XMLCh, Deleter> covise::stringToXexcesc(const std::string &s)
{
    return std::unique_ptr<XMLCh, Deleter>(xercesc::XMLString::transcode(s.c_str()), Deleter());
}
