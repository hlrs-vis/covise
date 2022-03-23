#include "coConfigXercesConverter.h"
#include <iostream>
#include <string>
#include <xercesc/util/XMLString.hpp>

#ifdef HAVE_UNICODE_LITERALS
#include <locale>
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
    const wchar_t *s;
    static std::wstring_convert<deletable_facet<std::codecvt<XMLCh, char, std::mbstate_t>>, XMLCh> convert;
    return convert.to_bytes(str);
}
#else
#include <boost/locale/encoding.hpp>
std::string covise::xercescToStdString(const XMLCh *str)
{
    if (!str)
        return "";
    return boost::locale::conv::from_utf<XMLCh>(str, "Latin1");
}

#endif
void Deleter::operator()(XMLCh *d)
{
    xercesc::XMLString::release(&d);
}

std::unique_ptr<XMLCh, Deleter> covise::stringToXexcesc(const std::string &s)
{
    return std::unique_ptr<XMLCh, Deleter>(xercesc::XMLString::transcode(s.c_str()), Deleter());
}