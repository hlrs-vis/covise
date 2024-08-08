#ifndef COCONFIG_XERCES_CONVERTER_H
#define COCONFIG_XERCES_CONVERTER_H

#include <string>
#include <memory>
#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/util/Xerces_autoconf_config.hpp>

struct Deleter
{
    void operator()(XMLCh *d);
};

namespace covise
{
    std::string xercescToStdString(const XMLCh *str);
    std::unique_ptr<XMLCh, Deleter> stringToXexcesc(const std::string &s);
}

#endif // COCONFIG_XERCES_CONVERTER_H
