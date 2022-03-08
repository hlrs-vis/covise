#include "coConfigXercesConverter.h"
#include <locale>
#include <iostream>
#include <xercesc/util/XMLString.hpp>

// utility wrapper to adapt locale-bound facets for wstring/wbuffer convert
template <class Facet>
struct deletable_facet : Facet
{
    template <class... Args>
    deletable_facet(Args &&...args) : Facet(std::forward<Args>(args)...) {}
    ~deletable_facet() {}
};

std::string covise::xercescToStdString(const XMLCh *str)
{
    if (!str)
        return "";
    static std::wstring_convert<deletable_facet<std::codecvt<XMLCh, char, std::mbstate_t>>, XMLCh> convert;
    return convert.to_bytes(str);
}

void Deleter::operator()(XMLCh *d)
{
    xercesc::XMLString::release(&d);
}

std::unique_ptr<XMLCh, Deleter> covise::stringToXexcesc(const std::string &s)
{
    return std::unique_ptr<XMLCh, Deleter>(xercesc::XMLString::transcode(s.c_str()), Deleter());
}